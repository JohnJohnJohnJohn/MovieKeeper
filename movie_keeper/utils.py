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
    """Configure Python logging to the console and optionally a file.

    Args:
        log_file: Optional path to a file. If supplied, logs are also written
            to this file in addition to stderr.

    Returns:
        The configured root logger.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers to keep configuration idempotent.
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


def move_to_trash(file_path: PathLike, trash_dir: Optional[PathLike] = None) -> Optional[Path]:
    """Move a file to a trash directory rather than deleting it.

    The trash directory is created if it does not exist. If a file with the
    same name already lives in the trash dir, a numeric suffix is appended.

    Args:
        file_path: Path of the file to move.
        trash_dir: Destination trash directory. Defaults to
            ``<file parent>/.movie_keeper_trash``.

    Returns:
        The new path inside the trash directory, or ``None`` if the source
        file does not exist.
    """
    src = Path(file_path)
    log = logging.getLogger(__name__)

    if not src.exists():
        log.warning("Cannot move to trash, file not found: %s", src)
        return None

    if trash_dir is None:
        trash_path = src.parent / DEFAULT_TRASH_DIR_NAME
    else:
        trash_path = Path(trash_dir)

    trash_path.mkdir(parents=True, exist_ok=True)

    target = trash_path / src.name
    counter = 1
    while target.exists():
        target = trash_path / f"{src.stem}__{counter}{src.suffix}"
        counter += 1

    try:
        shutil.move(str(src), str(target))
        log.info("Moved %s -> %s", src, target)
        return target
    except OSError as exc:
        log.error("Failed to move %s to trash: %s", src, exc)
        return None


def _run_ffprobe(args: list, file_path: PathLike) -> Optional[Dict[str, Any]]:
    """Run ffprobe with the supplied args and parse JSON output."""
    log = logging.getLogger(__name__)
    cmd = ["ffprobe", "-v", "error", *args, "-of", "json", str(file_path)]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        log.error("ffprobe not found on PATH")
        return None
    except subprocess.CalledProcessError as exc:
        log.warning("ffprobe failed for %s: %s", file_path, exc.stderr.strip())
        return None

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        log.warning("Could not parse ffprobe output for %s: %s", file_path, exc)
        return None


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


def get_video_metadata(file_path: PathLike) -> Optional[Dict[str, Any]]:
    """Return a dict with ``width``, ``height``, and ``bit_rate``.

    Resolution comes from the first video stream; bit_rate falls back to
    ``format.bit_rate`` if the stream does not report it.
    """
    data = _run_ffprobe(
        [
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,bit_rate:format=bit_rate",
        ],
        file_path,
    )
    if not data:
        return None

    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    if not streams:
        return None

    stream = streams[0]
    width = stream.get("width")
    height = stream.get("height")

    bit_rate_raw = stream.get("bit_rate") or fmt.get("bit_rate")
    try:
        bit_rate = int(bit_rate_raw) if bit_rate_raw is not None else None
    except (TypeError, ValueError):
        bit_rate = None

    return {
        "width": int(width) if width is not None else None,
        "height": int(height) if height is not None else None,
        "bit_rate": bit_rate,
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
