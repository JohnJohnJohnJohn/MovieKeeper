"""Quality-based duplicate resolution."""

from __future__ import annotations

import logging
import statistics
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

from .utils import get_video_metadata

PathLike = Union[str, Path]

log = logging.getLogger(__name__)

QualityScore = Tuple[int, int, int]


def get_quality_score(file_path: PathLike) -> QualityScore:
    """Return ``(pixel_count, bitrate, duration_sec)`` for ranking."""
    path = Path(file_path)
    meta = get_video_metadata(path) or {}

    width = meta.get("width") or 0
    height = meta.get("height") or 0
    bit_rate = meta.get("bit_rate") or 0
    duration = int(meta.get("duration_sec") or 0)
    return (int(width) * int(height), int(bit_rate), duration)


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size if path.exists() else 0
    except OSError:
        return 0


def rank_videos_for_keep(
    group: Sequence[PathLike],
    *,
    quality_score_for: Callable[[Path], QualityScore] = get_quality_score,
    duration_for: Optional[Callable[[Path], Optional[float]]] = None,
) -> List[Path]:
    """Rank duplicate candidates: pixels, bitrate, duration proximity, size, path."""
    paths = [Path(path) for path in group]
    if not paths:
        return []

    if duration_for is None:
        duration_for = lambda path: (get_video_metadata(path) or {}).get("duration_sec")

    durations = [duration_for(path) for path in paths]
    valid_durations = [value for value in durations if value is not None and value > 0]
    median_duration = statistics.median(valid_durations) if valid_durations else None

    scored = []
    for path, duration in zip(paths, durations):
        quality = quality_score_for(path)
        if median_duration is None or duration is None or duration <= 0:
            proximity = float("inf")
        else:
            proximity = abs(duration - median_duration)
        scored.append((quality, proximity, _safe_size(path), path))

    scored.sort(
        key=lambda item: (
            -item[0][0],
            -item[0][1],
            item[1],
            -item[2],
            str(item[3]).lower(),
        )
    )
    return [path for *_, path in scored]


def resolve_duplicates(
    duplicate_groups: Iterable[Sequence[PathLike]],
) -> List[Tuple[Path, List[Path]]]:
    """For each duplicate group, pick the highest quality file to keep."""
    resolved: List[Tuple[Path, List[Path]]] = []

    for raw_group in duplicate_groups:
        group: List[Path] = [Path(p) for p in raw_group]
        if not group:
            continue
        if len(group) == 1:
            resolved.append((group[0], []))
            continue

        ranked = rank_videos_for_keep(group)
        keep = ranked[0]
        remove = [path for path in group if path != keep]
        score = get_quality_score(keep)
        log.info(
            "Group of %d -> keep %s (%dpx, %dbps)",
            len(group),
            keep,
            score[0],
            score[1],
        )
        resolved.append((keep, remove))

    return resolved
