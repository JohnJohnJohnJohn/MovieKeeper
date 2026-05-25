"""Detect and merge contiguous split-part video files."""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .naming import standardize_stem
from .utils import get_video_duration, move_to_trash, video_is_readable

log = logging.getLogger(__name__)

_ALREADY_MERGED = re.compile(
    r"(?:_part_|_pt_|_cd|_disc|\s)(\d+)to(\d+)\s*$",
    re.IGNORECASE,
)

_PART_PATTERNS = (
    re.compile(r"^(?P<prefix>.+?_part_)(?P<num>\d+)$", re.IGNORECASE),
    re.compile(r"^(?P<prefix>.+?_pt_)(?P<num>\d+)$", re.IGNORECASE),
    re.compile(r"^(?P<prefix>.+?_cd)(?P<num>\d+)$", re.IGNORECASE),
    re.compile(r"^(?P<prefix>.+?-cd)(?P<num>\d+)$", re.IGNORECASE),
    re.compile(r"^(?P<prefix>.+?_disc)(?P<num>\d+)$", re.IGNORECASE),
    re.compile(r"^(?P<prefix>.+ )(?P<num>\d+)$"),
    re.compile(r"^(?P<prefix>.+-)(?P<num>\d+)$"),
)


@dataclass(frozen=True)
class PartNameParts:
    prefix: str
    suffix: str
    part: int


@dataclass(frozen=True)
class PartInfo:
    file: Path
    prefix: str
    suffix: str
    part: int

    @property
    def parent(self) -> Path:
        return self.file.parent

    @property
    def group_key(self) -> Tuple[str, str, str]:
        return (str(self.parent.resolve()), self.prefix, self.suffix)


@dataclass(frozen=True)
class PartMergeGroup:
    parts: Tuple[PartInfo, ...]

    @property
    def parent(self) -> Path:
        return self.parts[0].parent

    @property
    def start(self) -> int:
        return self.parts[0].part

    @property
    def end(self) -> int:
        return self.parts[-1].part

    @property
    def source_files(self) -> List[Path]:
        return [info.file for info in self.parts]

    def raw_merged_stem(self) -> str:
        prefix = self.parts[0].prefix
        suffix = self.parts[0].suffix
        return f"{prefix}{self.start}to{self.end}{suffix}"

    def merged_stem(self) -> str:
        return standardize_stem(self.raw_merged_stem())

    def total_duration(self) -> float:
        total = 0.0
        for info in self.parts:
            duration = get_video_duration(info.file)
            if duration:
                total += duration
        return total


def parse_part_name(stem: str) -> Optional[PartNameParts]:
    if _ALREADY_MERGED.search(stem):
        return None

    for pattern in _PART_PATTERNS:
        match = pattern.match(stem)
        if not match:
            continue
        suffix = match.groupdict().get("suffix") or ""
        return PartNameParts(
            prefix=match.group("prefix"),
            suffix=suffix,
            part=int(match.group("num")),
        )
    return None


def parse_part_file(file_path: Path) -> Optional[PartInfo]:
    parsed = parse_part_name(file_path.stem)
    if parsed is None:
        return None
    return PartInfo(
        file=file_path.resolve(),
        prefix=parsed.prefix,
        suffix=parsed.suffix,
        part=parsed.part,
    )


def _find_contiguous_runs(parts: Sequence[PartInfo]) -> List[List[PartInfo]]:
    if len(parts) < 2:
        return []

    ordered = sorted(parts, key=lambda item: item.part)
    runs: List[List[PartInfo]] = []
    current = [ordered[0]]

    for item in ordered[1:]:
        if item.part == current[-1].part + 1:
            current.append(item)
            continue
        if len(current) >= 2:
            runs.append(current)
        current = [item]

    if len(current) >= 2:
        runs.append(current)
    return runs


def _durations_compatible(parts: Sequence[PartInfo], tolerance: float = 0.15) -> bool:
    durations = [get_video_duration(info.file) for info in parts]
    if any(duration is None or duration <= 0 for duration in durations):
        return True
    if len(parts) == 2:
        return True
    average = sum(durations) / len(durations)
    return all(abs(duration - average) / average <= tolerance for duration in durations)


def find_part_merge_groups(
    files: Iterable[Path],
    *,
    duration_tolerance: float = 0.15,
) -> List[PartMergeGroup]:
    grouped: dict[Tuple[str, str, str], List[PartInfo]] = {}

    for file_path in files:
        info = parse_part_file(Path(file_path))
        if info is None:
            continue
        grouped.setdefault(info.group_key, []).append(info)

    merge_groups: List[PartMergeGroup] = []
    for parts in grouped.values():
        for run in _find_contiguous_runs(parts):
            if _durations_compatible(run, tolerance=duration_tolerance):
                merge_groups.append(PartMergeGroup(parts=tuple(run)))

    merge_groups.sort(
        key=lambda group: (
            str(group.parent).lower(),
            group.parts[0].prefix.lower(),
            group.parts[0].suffix.lower(),
            group.start,
        )
    )
    return merge_groups


def unreadable_parts(group: PartMergeGroup) -> List[Path]:
    """Return source files that ffprobe/ffmpeg cannot read."""
    return [
        info.file
        for info in group.parts
        if not video_is_readable(info.file, quiet=True)
    ]


def unique_file_path(parent: Path, filename: str) -> Path:
    candidate = parent / filename
    counter = 2
    while candidate.exists():
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        candidate = parent / f"{stem}__{counter}{suffix}"
        counter += 1
    return candidate


def _merge_with_ffmpeg(
    sources: Sequence[Path],
    output: Path,
    *,
    reencode: bool = False,
) -> bool:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    ) as list_file:
        for source in sources:
            escaped = str(source.resolve()).replace("'", "'\\''")
            list_file.write(f"file '{escaped}'\n")
        list_path = Path(list_file.name)

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
    ]
    if reencode:
        cmd.extend(
            [
                "-map",
                "0:v:0?",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "medium",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
            ]
        )
    else:
        cmd.extend(["-c", "copy"])
    cmd.append(str(output))

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        log.error("ffmpeg not found on PATH; cannot merge into %s", output.name)
        list_path.unlink(missing_ok=True)
        return False
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        log.warning(
            "ffmpeg concat failed (%s): %s",
            output.name,
            stderr or exc,
        )
        list_path.unlink(missing_ok=True)
        return False
    finally:
        list_path.unlink(missing_ok=True)

    return output.exists() and output.stat().st_size > 0


def merge_part_group(
    group: PartMergeGroup,
    *,
    trash_dir: Path,
    library_root: Optional[Path] = None,
    dry_run: bool = False,
) -> Optional[Path]:
    merged_stem = group.merged_stem()
    output = unique_file_path(group.parent, f"{merged_stem}.mp4")

    if dry_run:
        return output

    bad_parts = unreadable_parts(group)
    if bad_parts:
        for path in bad_parts:
            log.error("Cannot merge unreadable part: %s", path)
        return None

    if _merge_with_ffmpeg(group.source_files, output, reencode=False):
        pass
    elif not _merge_with_ffmpeg(group.source_files, output, reencode=True):
        if output.exists():
            output.unlink(missing_ok=True)
        return None

    for source in group.source_files:
        moved = move_to_trash(
            source,
            trash_dir=trash_dir,
            library_root=library_root,
        )
        if moved is None:
            log.error("Failed to trash merged source: %s", source)

    return output
