"""Command line entry point for movie_keeper.

Wires together scanning, exact-duplicate detection, perceptual duplicate
resolution, and MP4 conversion into a single interactive workflow.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Set

from movie_keeper.scanner import scan_directory
from movie_keeper.hasher import find_exact_duplicates, select_file_to_keep
from movie_keeper.perceptual import find_perceptual_duplicates
from movie_keeper.resolver import resolve_duplicates
from movie_keeper.converter import needs_conversion, convert_all
from movie_keeper.utils import setup_logging, move_to_trash, format_size


# --- Lightweight ANSI helpers --------------------------------------------------

_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}


def _color(text: str, *styles: str) -> str:
    """Wrap ``text`` in ANSI styles when stdout is a TTY."""
    if not sys.stdout.isatty():
        return text
    prefix = "".join(_ANSI[s] for s in styles if s in _ANSI)
    if not prefix:
        return text
    return f"{prefix}{text}{_ANSI['reset']}"


def _header(title: str) -> None:
    """Print a clearly demarcated section header."""
    bar = "=" * 3
    print()
    print(_color(f"{bar} {title} {bar}", "bold", "cyan"))


def _info(text: str) -> None:
    print(_color(text, "blue"))


def _ok(text: str) -> None:
    print(_color(text, "green"))


def _warn(text: str) -> None:
    print(_color(text, "yellow"))


def _err(text: str) -> None:
    print(_color(text, "red"))


def _confirm(prompt: str) -> bool:
    """Ask the user a y/N question. Returns True only on explicit yes."""
    try:
        answer = input(f"{prompt} ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


# --- Argument parsing ----------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="movie-keeper",
        description=(
            "Scan a directory of videos, deduplicate them (exact + perceptual), "
            "and convert anything that is not already an H.264/AAC MP4."
        ),
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Root directory to scan for movies.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without deleting or converting anything.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Perceptual hash similarity threshold (lower = stricter). Default: 10.",
    )
    parser.add_argument(
        "--keep-originals",
        action="store_true",
        help="Keep the original files after MP4 conversion (do not move to trash).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to write a detailed operation log.",
    )
    return parser


# --- Helpers for size accounting ----------------------------------------------

def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size if path.exists() else 0
    except OSError:
        return 0


def _trash_files(
    files_to_remove: Iterable[Path],
    trash_dir: Path,
    log: logging.Logger,
) -> int:
    """Move each file to trash. Returns total bytes reclaimed."""
    reclaimed = 0
    for victim in files_to_remove:
        size = _safe_size(victim)
        try:
            moved = move_to_trash(victim, trash_dir=trash_dir)
        except Exception as exc:  # pragma: no cover - defensive
            log.error("Unexpected error trashing %s: %s", victim, exc)
            continue
        if moved is not None:
            reclaimed += size
            print(f"  {_color('trashed', 'yellow')} {victim.name}")
        else:
            _err(f"  failed to trash {victim.name}")
    return reclaimed


# --- Phase implementations -----------------------------------------------------

def _phase_exact_duplicates(
    files: List[Path],
    trash_dir: Path,
    dry_run: bool,
    log: logging.Logger,
) -> tuple[List[Path], int, int]:
    """Returns (remaining_files, removed_count, bytes_reclaimed)."""
    _header("Phase 2: Removing Exact Duplicates")
    try:
        groups = find_exact_duplicates(files)
    except Exception as exc:
        log.error("Exact duplicate detection failed: %s", exc)
        return files, 0, 0

    if not groups:
        _ok("No exact duplicates found.")
        return files, 0, 0

    plan: List[tuple[Path, List[Path]]] = []
    for digest, group in groups.items():
        try:
            keep, remove = select_file_to_keep(group)
        except ValueError:
            continue
        plan.append((keep, remove))

    total_dupes = sum(len(r) for _, r in plan)
    print(_color(f"Found {len(plan)} duplicate group(s) "
                 f"covering {total_dupes} extra file(s):", "bold"))
    for keep, remove in plan:
        print(f"  {_color('KEEP', 'green')}   {keep}")
        for victim in remove:
            print(f"  {_color('REMOVE', 'red')} {victim}")

    if dry_run:
        _warn("Dry run: no files were moved.")
        # In dry-run, downstream phases should still see the originals so the
        # user gets a complete preview.
        return files, 0, 0

    if total_dupes == 0:
        return files, 0, 0

    if not _confirm(
        _color(f"Found {total_dupes} exact duplicates. Remove them? [y/N]", "yellow")
    ):
        _warn("Skipped exact duplicate removal.")
        return files, 0, 0

    removed: Set[Path] = set()
    reclaimed = 0
    for _, remove in plan:
        reclaimed += _trash_files(remove, trash_dir, log)
        removed.update(remove)

    remaining = [p for p in files if p not in removed]
    _ok(f"Removed {len(removed)} exact duplicate(s); reclaimed {format_size(reclaimed)}.")
    return remaining, len(removed), reclaimed


def _phase_perceptual_duplicates(
    files: List[Path],
    threshold: int,
    trash_dir: Path,
    dry_run: bool,
    log: logging.Logger,
) -> tuple[List[Path], int, int]:
    _header("Phase 3: Removing Perceptual Duplicates")
    try:
        groups = find_perceptual_duplicates(files, threshold=threshold)
    except Exception as exc:
        log.error("Perceptual duplicate detection failed: %s", exc)
        return files, 0, 0

    if not groups:
        _ok("No perceptual duplicates found.")
        return files, 0, 0

    try:
        resolved = resolve_duplicates(groups)
    except Exception as exc:
        log.error("Quality resolution failed: %s", exc)
        return files, 0, 0

    plan = [(keep, remove) for keep, remove in resolved if remove]
    total_victims = sum(len(r) for _, r in plan)
    print(_color(f"Found {len(plan)} perceptual group(s) "
                 f"covering {total_victims} lower-quality file(s):", "bold"))
    for keep, remove in plan:
        print(f"  {_color('KEEP', 'green')}   {keep}")
        for victim in remove:
            print(f"  {_color('REMOVE', 'red')} {victim}")

    if dry_run:
        _warn("Dry run: no files were moved.")
        return files, 0, 0

    if total_victims == 0:
        return files, 0, 0

    if not _confirm(
        _color(
            f"Found {total_victims} perceptual duplicates. Remove the lower-quality "
            f"copies? [y/N]",
            "yellow",
        )
    ):
        _warn("Skipped perceptual duplicate removal.")
        return files, 0, 0

    removed: Set[Path] = set()
    reclaimed = 0
    for _, remove in plan:
        reclaimed += _trash_files(remove, trash_dir, log)
        removed.update(remove)

    remaining = [p for p in files if p not in removed]
    _ok(
        f"Removed {len(removed)} perceptual duplicate(s); "
        f"reclaimed {format_size(reclaimed)}."
    )
    return remaining, len(removed), reclaimed


def _phase_conversion(
    files: List[Path],
    trash_dir: Path,
    dry_run: bool,
    keep_originals: bool,
    log: logging.Logger,
) -> tuple[int, int]:
    """Returns (converted_count, bytes_delta)."""
    _header("Phase 4: Converting to MP4")

    candidates: List[Path] = []
    for path in files:
        try:
            if needs_conversion(path):
                candidates.append(path)
        except Exception as exc:
            log.error("needs_conversion failed for %s: %s", path, exc)

    if not candidates:
        _ok("Every remaining file is already H.264/AAC MP4.")
        return 0, 0

    print(_color(f"{len(candidates)} file(s) need conversion:", "bold"))
    for path in candidates:
        print(f"  {_color('CONVERT', 'magenta')} {path}")

    if dry_run:
        _warn("Dry run: no conversions performed.")
        return 0, 0

    if not _confirm(
        _color(f"Convert {len(candidates)} file(s) to H.264/AAC MP4? [y/N]", "yellow")
    ):
        _warn("Skipped conversion.")
        return 0, 0

    # Track sizes for space-saved accounting.
    pre_sizes = {p: _safe_size(p) for p in candidates}

    try:
        results = convert_all(
            candidates,
            dry_run=False,
            keep_originals=keep_originals,
            trash_dir=trash_dir,
        )
    except Exception as exc:
        log.error("Conversion batch failed: %s", exc)
        return 0, 0

    converted = 0
    bytes_delta = 0  # positive = saved, negative = grew
    for src, output in results:
        if output is None:
            continue
        converted += 1
        new_size = _safe_size(output)
        bytes_delta += pre_sizes.get(src, 0) - new_size

    if converted:
        _ok(
            f"Converted {converted} file(s); net size change: "
            f"{format_size(bytes_delta)} {'saved' if bytes_delta >= 0 else 'grew'}."
        )
    else:
        _warn("No files were successfully converted.")
    return converted, bytes_delta


# --- Main ----------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    log = setup_logging(args.log_file)

    root = Path(args.path).expanduser().resolve()
    if not root.exists():
        _err(f"Path does not exist: {root}")
        return 2
    if not root.is_dir():
        _err(f"Path is not a directory: {root}")
        return 2

    trash_dir = root / ".movie_keeper_trash"

    print(_color("movie_keeper", "bold", "magenta"))
    print(f"  root:      {root}")
    print(f"  trash:     {trash_dir}")
    print(f"  threshold: {args.threshold}")
    print(f"  dry-run:   {args.dry_run}")
    print(f"  keep-originals: {args.keep_originals}")

    # Phase 1: scan
    _header("Phase 1: Scanning Directory")
    try:
        files = scan_directory(root)
    except Exception as exc:
        log.error("Scan failed: %s", exc)
        _err(f"Scan failed: {exc}")
        return 1

    initial_count = len(files)
    print(_color(f"Discovered {initial_count} video file(s).", "bold"))
    if initial_count == 0:
        _warn("Nothing to do.")
        return 0

    exact_removed = perceptual_removed = 0
    exact_bytes = perceptual_bytes = 0
    converted = converted_bytes_delta = 0

    try:
        files, exact_removed, exact_bytes = _phase_exact_duplicates(
            files, trash_dir, args.dry_run, log
        )

        files, perceptual_removed, perceptual_bytes = _phase_perceptual_duplicates(
            files, args.threshold, trash_dir, args.dry_run, log
        )

        converted, converted_bytes_delta = _phase_conversion(
            files, trash_dir, args.dry_run, args.keep_originals, log
        )
    except KeyboardInterrupt:
        print()
        _warn("Interrupted by user. Exiting cleanly.")
        return 130

    # Final summary
    _header("Summary")
    total_reclaimed = exact_bytes + perceptual_bytes + max(converted_bytes_delta, 0)
    print(f"  Files discovered:         {initial_count}")
    print(f"  Exact duplicates removed: {exact_removed} "
          f"({format_size(exact_bytes)})")
    print(f"  Perceptual duplicates:    {perceptual_removed} "
          f"({format_size(perceptual_bytes)})")
    print(f"  Files converted:          {converted} "
          f"(net {format_size(converted_bytes_delta)})")
    print(_color(f"  Approx. space saved:      {format_size(total_reclaimed)}", "green", "bold"))
    if args.dry_run:
        _warn("This was a dry run — no files were modified.")
    elif (exact_removed + perceptual_removed) and not trash_dir.exists():
        # Defensive: unlikely, but informative.
        pass
    else:
        print(f"  Trash directory:          {trash_dir}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        _warn("Interrupted by user. Exiting cleanly.")
        sys.exit(130)
