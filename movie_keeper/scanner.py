"""Filesystem scanning for video files."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from .index import VideoIndex

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

_BAR_WIDTH = 28
_UPDATE_INTERVAL_S = 0.1
_LOG_INTERVAL_DIRS = 500


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts if part not in ("/", ""))


class _ScanProgress:
    """Live scan progress for TTY consoles; periodic logs otherwise."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.dirs_scanned = 0
        self.videos_found = 0
        self.files_seen = 0
        self.cache_hits = 0
        self._interactive = sys.stdout.isatty()
        self._last_render = 0.0
        self._last_logged_dirs = 0
        self._log = logging.getLogger(__name__)

    def visit_dir(self) -> None:
        self.dirs_scanned += 1
        self._maybe_update()

    def add_video(self, *, cached: bool = False) -> None:
        self.videos_found += 1
        if cached:
            self.cache_hits += 1
        self._maybe_update(force=True)

    def add_files(self, count: int) -> None:
        if count:
            self.files_seen += count
            self._maybe_update()

    def _bar(self) -> str:
        position = self.dirs_scanned % _BAR_WIDTH
        return f"[{'=' * position}>{' ' * (_BAR_WIDTH - position - 1)}]"

    def _message(self) -> str:
        parts = [
            self._bar(),
            f"{self.dirs_scanned:,} folders",
            f"{self.videos_found:,} videos",
            f"{self.files_seen:,} files",
        ]
        if self.cache_hits:
            parts.append(f"{self.cache_hits:,} cached")
        return " | ".join(parts)

    def _maybe_update(self, force: bool = False) -> None:
        now = time.monotonic()
        if self._interactive:
            if force or now - self._last_render >= _UPDATE_INTERVAL_S:
                self._last_render = now
                sys.stdout.write(f"\rScanning {self.root} ... {self._message()}")
                sys.stdout.flush()
            return

        if force or self.dirs_scanned - self._last_logged_dirs >= _LOG_INTERVAL_DIRS:
            self._last_logged_dirs = self.dirs_scanned
            self._log.info("Scan progress: %s", self._message())

    def finish(self) -> None:
        if self._interactive:
            sys.stdout.write(f"\rScanning {self.root} ... {self._message()}\n")
            sys.stdout.flush()


def scan_directory(
    root_path: PathLike,
    index: Optional["VideoIndex"] = None,
    *,
    use_cache: bool = True,
) -> List[Path]:
    """Recursively scan ``root_path`` for supported video files."""
    log = logging.getLogger(__name__)
    root = Path(root_path).expanduser().resolve()

    if not root.exists():
        log.error("Scan root does not exist: %s", root)
        return []
    if not root.is_dir():
        log.error("Scan root is not a directory: %s", root)
        return []

    results: List[Path] = []
    total_seen = 0
    progress = _ScanProgress(root)

    stack: List[Path] = [root]
    while stack:
        current = stack.pop()
        progress.visit_dir()

        try:
            entries = list(current.iterdir())
        except (PermissionError, OSError) as exc:
            log.warning("Cannot read %s: %s", current, exc)
            continue

        dir_files = 0
        for entry in entries:
            if entry.name.startswith(".") or entry.is_symlink():
                continue
            if entry.is_dir():
                stack.append(entry)
            elif entry.is_file():
                dir_files += 1
                if entry.suffix.lower() in VIDEO_EXTENSIONS:
                    resolved = entry.resolve()
                    cached = bool(
                        use_cache
                        and index is not None
                        and index.is_scan_cache_hit(resolved)
                    )
                    results.append(resolved)
                    progress.add_video(cached=cached)

        total_seen += dir_files
        progress.add_files(dir_files)

    progress.finish()
    print(
        f"Found {len(results)} video file(s) "
        f"(out of {total_seen} files inspected)."
    )
    if progress.cache_hits:
        print(f"Reused scan cache for {progress.cache_hits} video(s).")

    if index is not None:
        for video in results:
            index.ensure_metadata(video)
        pruned = index.prune_to(results)
        if pruned:
            log.info("Pruned %d stale index record(s).", pruned)

    return results
