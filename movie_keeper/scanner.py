"""Filesystem scanning for video files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

PathLike = Union[str, Path]

VIDEO_EXTENSIONS = frozenset(
    {
        ".mp4",
        ".mkv",
        ".avi",
        ".mov",
        ".wmv",
        ".flv",
        ".webm",
        ".m4v",
        ".ts",
        ".mpg",
        ".mpeg",
    }
)


def _is_hidden(path: Path) -> bool:
    """True if any segment of the path begins with a dot."""
    return any(part.startswith(".") for part in path.parts if part not in ("/", ""))


def scan_directory(root_path: PathLike) -> List[Path]:
    """Recursively scan ``root_path`` for supported video files.

    Hidden files and directories (prefixed with ``.``) are skipped. Returns a
    list of absolute paths.
    """
    log = logging.getLogger(__name__)
    root = Path(root_path).expanduser().resolve()

    if not root.exists():
        log.error("Scan root does not exist: %s", root)
        return []
    if not root.is_dir():
        log.error("Scan root is not a directory: %s", root)
        return []

    print(f"Scanning {root} ...")
    results: List[Path] = []
    total_seen = 0

    # Manual walk so we can prune hidden directories.
    stack: List[Path] = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except (PermissionError, OSError) as exc:
            log.warning("Cannot read %s: %s", current, exc)
            continue

        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_symlink():
                # Avoid traversing symlinks to keep scans bounded.
                continue
            if entry.is_dir():
                stack.append(entry)
            elif entry.is_file():
                total_seen += 1
                if entry.suffix.lower() in VIDEO_EXTENSIONS:
                    results.append(entry.resolve())

    print(f"Found {len(results)} video file(s) (out of {total_seen} files inspected).")
    return results
