"""Perceptual duplicate detection using frame-level pHash signatures."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    import imagehash
    from PIL import Image
except ImportError:
    imagehash = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from .index import VideoIndex

from .index import DEFAULT_PERCEPTUAL_FRAMES, compute_content_fingerprint
from .utils import get_video_duration, get_video_metadata

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
    num_frames: int = 15,
    temp_dir: Optional[PathLike] = None,
) -> List[Path]:
    """Extract evenly spaced frames from ``file_path``."""
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
        shutil.rmtree(out_dir, ignore_errors=True)

    return frame_paths


def compute_video_signature(
    file_path: PathLike,
    num_frames: int = DEFAULT_PERCEPTUAL_FRAMES,
) -> Optional[List]:
    """Compute a list of perceptual hashes for ``file_path``."""
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
    if not sig1 or not sig2:
        return float("inf")

    if len(sig1) <= len(sig2):
        query, target = sig1, sig2
    else:
        query, target = sig2, sig1

    total_best = 0
    for q_hash in query:
        best = min(q_hash - t_hash for t_hash in target)
        total_best += best
    return total_best / len(query)


def _best_match_details(sig1: Sequence, sig2: Sequence) -> Tuple[float, float]:
    if not sig1 or not sig2:
        return float("inf"), 0.0

    if len(sig1) <= len(sig2):
        query, target = sig1, sig2
    else:
        query, target = sig2, sig1

    best_distances: List[int] = []
    for q_hash in query:
        best = min(q_hash - t_hash for t_hash in target)
        best_distances.append(best)

    avg_distance = sum(best_distances) / len(best_distances)
    good_matches = sum(1 for d in best_distances if d <= 8)
    good_ratio = good_matches / len(best_distances)
    return avg_distance, good_ratio


def _serialize_phash(page_hash) -> str:
    return str(page_hash)


def _deserialize_phash(value: str):
    if imagehash is None:
        return None
    return imagehash.hex_to_hash(value)


def find_perceptual_duplicates(
    file_paths: Iterable[PathLike],
    threshold: int = 10,
    duration_tolerance: float = 60.0,
    index: Optional["VideoIndex"] = None,
    *,
    use_cache: bool = True,
    num_frames: int = DEFAULT_PERCEPTUAL_FRAMES,
) -> List[List[Path]]:
    """Find perceptual duplicate groups across ``file_paths``."""
    paths = [Path(p) for p in file_paths]
    if len(paths) < 2:
        return []

    print(f"Computing perceptual signatures for {len(paths)} file(s) ...")
    signatures: Dict[Path, List] = {}
    durations: Dict[Path, Optional[float]] = {}
    cache_hits = 0

    for step, path in enumerate(paths, start=1):
        sig: Optional[List] = None
        record = index.get(path) if index is not None else None

        if (
            use_cache
            and index is not None
            and record
            and record.perceptual_hashes
            and record.perceptual_num_frames == num_frames
            and index.is_content_cache_hit(path)
        ):
            sig = [
                deserialized
                for value in record.perceptual_hashes
                if (deserialized := _deserialize_phash(value)) is not None
            ]
            if sig:
                cache_hits += 1
                print(f"  [{step}/{len(paths)}] cached signature {path.name}")
                durations[path] = record.duration_sec
                signatures[path] = sig
                continue

        print(f"  [{step}/{len(paths)}] signature for {path.name}")
        sig = compute_video_signature(path, num_frames=num_frames)
        meta = get_video_metadata(path) or {}
        durations[path] = meta.get("duration_sec")

        if sig:
            signatures[path] = sig
            if index is not None:
                index.upsert(
                    path,
                    content_fingerprint=compute_content_fingerprint(path),
                    duration_sec=meta.get("duration_sec"),
                    width=meta.get("width"),
                    height=meta.get("height"),
                    bit_rate=meta.get("bit_rate"),
                    video_codec=meta.get("video_codec"),
                    audio_codec=meta.get("audio_codec"),
                    perceptual_hashes=[_serialize_phash(value) for value in sig],
                    perceptual_num_frames=num_frames,
                )

    if cache_hits:
        print(f"Reused perceptual cache for {cache_hits} video(s).")

    if len(signatures) < 2:
        print("Not enough signatures to compare.")
        return []

    keys: List[Path] = list(signatures.keys())
    uf: UnionFind = UnionFind(keys)

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            dur_a, dur_b = durations.get(a), durations.get(b)
            if dur_a is not None and dur_b is not None:
                if abs(dur_a - dur_b) > duration_tolerance:
                    continue
            avg_distance, good_ratio = _best_match_details(
                signatures[a], signatures[b]
            )
            if avg_distance <= threshold or good_ratio >= 0.6:
                uf.union(a, b)

    groups: List[List[Path]] = [g for g in uf.groups() if len(g) > 1]
    print(f"Found {len(groups)} perceptual-duplicate group(s).")
    return groups
