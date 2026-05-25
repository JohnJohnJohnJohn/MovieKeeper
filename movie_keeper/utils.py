"""Shared utility helpers for movie_keeper."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Union

PathLike = Union[str, Path]

DEFAULT_TRASH_DIR_NAME = ".movie_keeper_trash"


def setup_logging(log_file: Optional[PathLike] = None) -> logging.Logger:
    """Configure Python logging to the console and optionally a file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def move_to_trash(
    file_path: PathLike,
    trash_dir: Optional[PathLike] = None,
    *,
    library_root: Optional[PathLike] = None,
) -> Optional[Path]:
    """Move a file to a trash directory rather than deleting it."""
    src = Path(file_path)
    log = logging.getLogger(__name__)

    if not src.exists():
        log.warning("Cannot move to trash, file not found: %s", src)
        return None

    if trash_dir is None:
        trash_path = src.parent / DEFAULT_TRASH_DIR_NAME
    else:
        trash_path = Path(trash_dir)

    if library_root is not None:
        try:
            relative = src.resolve().relative_to(Path(library_root).resolve())
            target = trash_path / relative
        except ValueError:
            target = trash_path / src.name
    else:
        target = trash_path / src.name

    target.parent.mkdir(parents=True, exist_ok=True)

    counter = 1
    while target.exists():
        target = target.parent / f"{target.stem}__{counter}{target.suffix}"
        counter += 1

    try:
        shutil.move(str(src), str(target))
        log.info("Moved %s -> %s", src, target)
        return target
    except OSError as exc:
        log.error("Failed to move %s to trash: %s", src, exc)
        return None


def _run_ffprobe(
    args: list,
    file_path: PathLike,
    *,
    quiet: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run ffprobe with the supplied args and parse JSON output."""
    log = logging.getLogger(__name__)
    cmd = ["ffprobe", "-v", "error", *args, "-of", "json", str(file_path)]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        log.error("ffprobe not found on PATH")
        return None
    except subprocess.CalledProcessError as exc:
        if not quiet:
            log.warning("ffprobe failed for %s: %s", file_path, exc.stderr.strip())
        return None

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        if not quiet:
            log.warning("Could not parse ffprobe output for %s: %s", file_path, exc)
        return None


def video_is_readable(file_path: PathLike, *, quiet: bool = True) -> bool:
    """Return True when ffprobe can read at least one video stream."""
    data = _run_ffprobe(
        ["-select_streams", "v:0", "-show_entries", "stream=codec_type"],
        file_path,
        quiet=quiet,
    )
    streams = (data or {}).get("streams") or []
    return any(stream.get("codec_type") == "video" for stream in streams)


def get_video_duration(file_path: PathLike) -> Optional[float]:
    """Return the duration in seconds for ``file_path`` via ffprobe."""
    data = _run_ffprobe(
        ["-show_entries", "format=duration"],
        file_path,
    )
    if not data:
        return None

    fmt = data.get("format") or {}
    raw = fmt.get("duration")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def get_video_metadata(
    file_path: PathLike,
    *,
    quiet: bool = False,
) -> Optional[Dict[str, Any]]:
    """Return video metadata including codecs and duration."""
    data = _run_ffprobe(
        [
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,bit_rate,codec_name,codec_type:format=bit_rate,duration",
        ],
        file_path,
        quiet=quiet,
    )
    if not data:
        return None

    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    if not streams:
        return None

    video_stream = streams[0]
    width = video_stream.get("width")
    height = video_stream.get("height")

    bit_rate_raw = video_stream.get("bit_rate") or fmt.get("bit_rate")
    try:
        bit_rate = int(bit_rate_raw) if bit_rate_raw is not None else None
    except (TypeError, ValueError):
        bit_rate = None

    duration_raw = fmt.get("duration")
    try:
        duration_sec = float(duration_raw) if duration_raw is not None else None
    except (TypeError, ValueError):
        duration_sec = None

    audio_codec: Optional[str] = None
    for stream in data.get("streams") or []:
        if stream.get("codec_type") == "audio" and audio_codec is None:
            audio_codec = stream.get("codec_name")

    return {
        "width": int(width) if width is not None else None,
        "height": int(height) if height is not None else None,
        "bit_rate": bit_rate,
        "duration_sec": duration_sec,
        "video_codec": video_stream.get("codec_name"),
        "audio_codec": audio_codec,
    }


def format_size(num_bytes: float) -> str:
    """Return a human-readable representation of a byte count."""
    if num_bytes is None:
        return "0 B"

    size = float(num_bytes)
    sign = "-" if size < 0 else ""
    size = abs(size)

    units = ("B", "KB", "MB", "GB", "TB", "PB", "EB")
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{sign}{int(size)} {unit}"
            return f"{sign}{size:.2f} {unit}"
        size /= 1024.0

    return f"{sign}{size:.2f} {units[-1]}"


def format_duration(seconds: Optional[float]) -> str:
    """Return a compact human-readable duration."""
    if seconds is None or seconds <= 0:
        return "unknown duration"

    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_bitrate(bps: Optional[int]) -> str:
    if not bps:
        return "unknown bitrate"
    if bps >= 1_000_000:
        return f"{bps / 1_000_000:.1f} Mbps"
    return f"{bps / 1000:.0f} Kbps"
