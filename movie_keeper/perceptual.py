"""Perceptual duplicate detection using frame-level pHash signatures."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:  # pragma: no cover - optional dependency import
    import imagehash
    from PIL import Image
except ImportError:  # pragma: no cover - handled at runtime
    imagehash = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]

from .utils import get_video_duration

PathLike = Union[str, Path]

log = logging.getLogger(__name__)


class UnionFind:
    """Minimal union-find / disjoint-set structure keyed by hashable items."""

    def __init__(self, items: Optional[Iterable] = None) -> None:
        self._parent: Dict = {}
        self._rank: Dict = {}
        if items:
            for item in items:
                self.add(item)

    def add(self, item) -> None:
        if item not in self._parent:
            self._parent[item] = item
            self._rank[item] = 0

    def find(self, item):
        self.add(item)
        # Path compression.
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        cur = item
        while self._parent[cur] != root:
            self._parent[cur], cur = root, self._parent[cur]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def groups(self) -> List[List]:
        buckets: Dict = {}
        for item in self._parent:
            buckets.setdefault(self.find(item), []).append(item)
        return list(buckets.values())


def extract_frames(
    file_path: PathLike,
    num_frames: int = 5,
    temp_dir: Optional[PathLike] = None,
) -> List[Path]:
    """Extract ``num_frames`` evenly spaced frames from ``file_path``.

    The first and last 10% of the timeline are skipped to avoid intros and
    credits. Frames are written as JPEGs in ``temp_dir`` (a fresh temp
    directory is created when not supplied). Caller is responsible for
    cleaning up temp directories they passed in.
    """
    src = Path(file_path)
    duration = get_video_duration(src)
    if not duration or duration <= 0:
        log.warning("Cannot determine duration for %s; skipping frame extraction", src)
        return []
    if num_frames <= 0:
        return []

    if temp_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="movie_keeper_frames_"))
        owns_dir = True
    else:
        out_dir = Path(temp_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        owns_dir = False

    start = duration * 0.10
    end = duration * 0.90
    if end <= start:
        start, end = 0.0, duration

    if num_frames == 1:
        timestamps = [(start + end) / 2.0]
    else:
        step = (end - start) / (num_frames - 1)
        timestamps = [start + step * i for i in range(num_frames)]

    frame_paths: List[Path] = []
    for index, ts in enumerate(timestamps):
        frame_path = out_dir / f"{src.stem}_{index:02d}.jpg"
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{ts:.3f}",
            "-i",
            str(src),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(frame_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except FileNotFoundError:
            log.error("ffmpeg not found on PATH")
            break
        except subprocess.CalledProcessError as exc:
            log.warning(
                "ffmpeg failed extracting frame %d of %s: %s",
                index,
                src,
                exc.stderr.decode("utf-8", errors="replace").strip(),
            )
            continue

        if frame_path.exists() and frame_path.stat().st_size > 0:
            frame_paths.append(frame_path)

    if owns_dir and not frame_paths:
        # Nothing extracted — clean up our private temp dir.
        shutil.rmtree(out_dir, ignore_errors=True)

    return frame_paths


def compute_video_signature(
    file_path: PathLike,
    num_frames: int = 5,
) -> Optional[List]:
    """Compute a list of perceptual hashes for ``file_path``.

    Returns ``None`` if the signature can't be computed (e.g. ffmpeg failure
    or missing ``imagehash`` dependency).
    """
    if imagehash is None or Image is None:
        log.error("imagehash / Pillow not installed; cannot compute perceptual hash")
        return None

    src = Path(file_path)
    temp_dir = Path(tempfile.mkdtemp(prefix="movie_keeper_sig_"))
    try:
        frames = extract_frames(src, num_frames=num_frames, temp_dir=temp_dir)
        if not frames:
            return None

        hashes = []
        for frame in frames:
            try:
                with Image.open(frame) as img:
                    hashes.append(imagehash.phash(img))
            except (OSError, ValueError) as exc:
                log.warning("Failed to hash frame %s: %s", frame, exc)

        return hashes or None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def compute_similarity(sig1: Sequence, sig2: Sequence) -> float:
    """Average Hamming distance between two perceptual signatures.

    Returns ``float('inf')`` when either signature is empty so that mismatched
    inputs cannot accidentally be considered duplicates.
    """
    if not sig1 or not sig2:
        return float("inf")

    pairs = min(len(sig1), len(sig2))
    total = 0
    for i in range(pairs):
        total += sig1[i] - sig2[i]  # imagehash defines __sub__ as Hamming dist
    return total / pairs


def find_perceptual_duplicates(
    file_paths: Iterable[PathLike],
    threshold: int = 10,
    duration_tolerance: float = 60.0,
) -> List[List[Path]]:
    """Find perceptual duplicate groups across ``file_paths``.

    Two files are considered duplicates when their average frame Hamming
    distance is ``<= threshold`` and their durations differ by no more than
    ``duration_tolerance`` seconds. Groups are formed transitively via
    union-find and only groups with more than one member are returned.
    """
    paths = [Path(p) for p in file_paths]
    print(f"Computing perceptual signatures for {len(paths)} file(s) ...")

    signatures: Dict[Path, List] = {}
    durations: Dict[Path, Optional[float]] = {}
    for index, path in enumerate(paths, start=1):
        print(f"  [{index}/{len(paths)}] signature for {path.name}")
        sig = compute_video_signature(path)
        if sig is None:
            log.info("No signature for %s; skipping", path)
            continue
        signatures[path] = sig
        durations[path] = get_video_duration(path)

    keys: List[Path] = list(signatures.keys())
    uf: UnionFind = UnionFind(keys)

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            dur_a, dur_b = durations.get(a), durations.get(b)
            if dur_a is not None and dur_b is not None:
                if abs(dur_a - dur_b) > duration_tolerance:
                    continue
            distance = compute_similarity(signatures[a], signatures[b])
            if distance <= threshold:
                uf.union(a, b)

    groups: List[List[Path]] = [g for g in uf.groups() if len(g) > 1]
    print(f"Found {len(groups)} perceptual-duplicate group(s).")
    return groups


__all__ = [
    "UnionFind",
    "extract_frames",
    "compute_video_signature",
    "compute_similarity",
    "find_perceptual_duplicates",
]


# Convenience tuple type for callers that want both keep + remove information.
PerceptualGroup = Tuple[Path, List[Path]]
