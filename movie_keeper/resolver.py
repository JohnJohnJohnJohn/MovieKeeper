"""Quality-based duplicate resolution."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

from .utils import get_video_metadata

PathLike = Union[str, Path]

log = logging.getLogger(__name__)

QualityScore = Tuple[int, int]


def get_quality_score(file_path: PathLike) -> QualityScore:
    """Return ``(pixel_count, bitrate)`` for ranking. Missing fields become 0."""
    path = Path(file_path)
    meta = get_video_metadata(path)
    if not meta:
        return (0, 0)

    width = meta.get("width") or 0
    height = meta.get("height") or 0
    bit_rate = meta.get("bit_rate") or 0
    return (int(width) * int(height), int(bit_rate))


def resolve_duplicates(
    duplicate_groups: Iterable[Sequence[PathLike]],
) -> List[Tuple[Path, List[Path]]]:
    """For each duplicate group, pick the highest quality file to keep.

    Quality is ranked by pixel count first, then bitrate. Ties fall back to
    file size, and finally to lexicographic path order, so the result is
    deterministic.
    """
    resolved: List[Tuple[Path, List[Path]]] = []

    for raw_group in duplicate_groups:
        group: List[Path] = [Path(p) for p in raw_group]
        if not group:
            continue
        if len(group) == 1:
            resolved.append((group[0], []))
            continue

        scored = []
        for path in group:
            try:
                size = path.stat().st_size if path.exists() else 0
            except OSError:
                size = 0
            score = get_quality_score(path)
            scored.append((score, size, path))

        # Sort highest-quality first; ties broken by larger size, then path.
        scored.sort(key=lambda t: (-t[0][0], -t[0][1], -t[1], str(t[2]).lower()))
        keep = scored[0][2]
        remove = [s[2] for s in scored[1:]]
        log.info(
            "Group of %d -> keep %s (%dpx, %dbps)",
            len(group),
            keep,
            scored[0][0][0],
            scored[0][0][1],
        )
        resolved.append((keep, remove))

    return resolved
