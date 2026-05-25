"""Command line entry point for movie_keeper."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Set

from movie_keeper.converter import (
    ConversionSettings,
    needs_container_or_codec_work,
    needs_conversion,
    standardize_video,
)
from movie_keeper.utils import video_is_readable
from movie_keeper.hasher import find_exact_duplicates, select_file_to_keep
from movie_keeper.index import VideoIndex, index_file_for
from movie_keeper.parts import (
    PartMergeGroup,
    find_part_merge_groups,
    merge_part_group,
    unreadable_parts,
)
from movie_keeper.perceptual import find_perceptual_duplicates
from movie_keeper.resolver import get_quality_score, rank_videos_for_keep
from movie_keeper.scanner import scan_directory
from movie_keeper.tagger import (
    build_tag_profiles,
    collect_video_signatures,
    extract_leading_tag,
    is_untagged_video,
    proposed_tagged_name,
    suggest_tags_for_untagged,
)
from movie_keeper.utils import (
    format_bitrate,
    format_duration,
    format_size,
    get_video_metadata,
    move_to_trash,
    setup_logging,
)

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
    if not sys.stdout.isatty():
        return text
    prefix = "".join(_ANSI[s] for s in styles if s in _ANSI)
    if not prefix:
        return text
    return f"{prefix}{text}{_ANSI['reset']}"


def _header(title: str) -> None:
    bar = "=" * 3
    print()
    print(_color(f"{bar} {title} {bar}", "bold", "cyan"))


def _ok(text: str) -> None:
    print(_color(text, "green"))


def _warn(text: str) -> None:
    print(_color(text, "yellow"))


def _err(text: str) -> None:
    print(_color(text, "red"))


def _confirm(prompt: str, *, default: bool = True) -> bool:
    try:
        answer = input(f"{prompt} ").strip().lower()
    except EOFError:
        return default
    if not answer:
        return default
    if answer in ("n", "no"):
        return False
    return answer in ("y", "yes")


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size if path.exists() else 0
    except OSError:
        return 0


def _unique_file_path(parent: Path, filename: str) -> Path:
    candidate = parent / filename
    counter = 2
    while candidate.exists():
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        candidate = parent / f"{stem}__{counter}{suffix}"
        counter += 1
    return candidate


def _format_video_label(path: Path, *, recommended: bool = False) -> str:
    meta = get_video_metadata(path) or {}
    width = meta.get("width") or 0
    height = meta.get("height") or 0
    resolution = f"{width}x{height}" if width and height else "unknown resolution"
    bitrate = format_bitrate(meta.get("bit_rate"))
    duration = format_duration(meta.get("duration_sec"))
    size = format_size(_safe_size(path))
    suffix = "  (recommended)" if recommended else ""
    return f"{path} ({resolution}, {bitrate}, {duration}, {size}){suffix}"


def _rank_video_group(group: List[Path]) -> List[Path]:
    return rank_videos_for_keep(group, quality_score_for=get_quality_score)


def _prompt_keep_choice(group: List[Path]) -> Optional[Path]:
    ordered = _rank_video_group(group)
    if not ordered:
        return None

    print("  Choose which version to keep:")
    for number, path in enumerate(ordered, start=1):
        print(f"    {number}. {_format_video_label(path, recommended=(number == 1))}")

    default_choice = 1
    while True:
        try:
            raw = input(
                _color(f"  Enter number to keep [{default_choice}]: ", "yellow")
            ).strip()
        except EOFError:
            return ordered[default_choice - 1]

        if not raw:
            return ordered[default_choice - 1]

        try:
            choice = int(raw)
        except ValueError:
            _warn("  Invalid choice; enter a number from the list.")
            continue

        if 1 <= choice <= len(ordered):
            return ordered[choice - 1]

        _warn(f"  Invalid choice; enter a number between 1 and {len(ordered)}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="movie-keeper",
        description=(
            "Scan a directory of movies, deduplicate them (exact + perceptual), "
            "and standardize filenames and MP4 formats."
        ),
    )
    parser.add_argument("--path", required=True, help="Root directory to scan for movies.")
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
        help="Keep the original files after standardization (do not move to trash).",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 CRF for phase 4 conversion (lower = higher quality). Default: 18.",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="ffmpeg x264 preset for phase 4 conversion. Default: medium.",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=0.0,
        metavar="MB",
        help="Optional output size cap. 0 disables size limits.",
    )
    parser.add_argument("--log-file", default=None, help="Path to write a detailed operation log.")
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore and rebuild the local scan/hash index.",
    )
    parser.add_argument(
        "--suggest-tags",
        action="store_true",
        help="Suggest bracket tags for untagged movies using tagged library styles.",
    )
    parser.add_argument(
        "--tags-only",
        action="store_true",
        help="Scan the library and run tag suggestions only.",
    )
    parser.add_argument(
        "--standardize-only",
        action="store_true",
        help="Scan the library and run filename/MP4 standardization (phase 4) only.",
    )
    parser.add_argument(
        "--skip-combine-parts",
        action="store_true",
        help="Skip merging contiguous split-part files (phase 5) during the normal pipeline.",
    )
    parser.add_argument(
        "--combine-parts-only",
        action="store_true",
        help="Scan the library and merge contiguous split-part files only.",
    )
    parser.add_argument(
        "--tag-min-samples",
        type=int,
        default=3,
        help="Minimum tagged movies required to learn a tag profile. Default: 3.",
    )
    parser.add_argument(
        "--tag-threshold",
        type=int,
        default=12,
        help="Visual similarity threshold for tag suggestions. Default: 12.",
    )
    parser.add_argument(
        "--apply-tags",
        action="store_true",
        help="Prompt to rename untagged movies when a tag match is suggested.",
    )
    return parser


def _trash_files(
    files_to_remove: Iterable[Path],
    trash_dir: Path,
    log: logging.Logger,
    *,
    library_root: Optional[Path] = None,
) -> int:
    reclaimed = 0
    for victim in files_to_remove:
        size = _safe_size(victim)
        try:
            moved = move_to_trash(
                victim,
                trash_dir=trash_dir,
                library_root=library_root,
            )
        except Exception as exc:
            log.error("Unexpected error trashing %s: %s", victim, exc)
            continue
        if moved is not None:
            reclaimed += size
            print(f"  {_color('trashed', 'yellow')} {victim.name}")
        else:
            _err(f"  failed to trash {victim.name}")
    return reclaimed


def _phase_exact_duplicates(
    files: List[Path],
    trash_dir: Path,
    dry_run: bool,
    log: logging.Logger,
    index: Optional[VideoIndex] = None,
    *,
    library_root: Optional[Path] = None,
    use_cache: bool = True,
) -> tuple[List[Path], int, int]:
    _header("Phase 2: Removing Exact Duplicates")
    try:
        groups = find_exact_duplicates(files, index=index, use_cache=use_cache)
    except Exception as exc:
        log.error("Exact duplicate detection failed: %s", exc)
        return files, 0, 0

    if not groups:
        _ok("No exact duplicates found.")
        return files, 0, 0

    removed: Set[Path] = set()
    reclaimed = 0
    group_count = 0

    for digest, group in groups.items():
        del digest
        try:
            keep, remove = select_file_to_keep(group)
        except ValueError:
            continue

        if not remove:
            continue

        group_count += 1
        print()
        print(_color(f"Exact duplicate group {group_count}:", "bold"))
        print(f"  {_color('KEEP', 'green')}   {keep}")
        for victim in remove:
            print(f"  {_color('REMOVE', 'red')} {victim}")

        if dry_run:
            _warn("  Dry run: would prompt to remove this group.")
            continue

        if not _confirm(_color("  Remove the duplicate(s) in this group? [Y/n]", "yellow")):
            _warn("  Skipped this group.")
            continue

        reclaimed += _trash_files(remove, trash_dir, log, library_root=library_root)
        removed.update(remove)
        if index is not None:
            index.remove_many(remove)

    if group_count == 0:
        _ok("No exact duplicates found.")
    elif dry_run:
        _warn("Dry run: no files were moved.")
    elif removed:
        _ok(
            f"Removed {len(removed)} exact duplicate(s); "
            f"reclaimed {format_size(reclaimed)}."
        )
    else:
        _warn("No exact duplicates were removed.")

    remaining = [p for p in files if p not in removed]
    return remaining, len(removed), reclaimed


def _phase_perceptual_duplicates(
    files: List[Path],
    threshold: int,
    trash_dir: Path,
    dry_run: bool,
    log: logging.Logger,
    index: Optional[VideoIndex] = None,
    *,
    library_root: Optional[Path] = None,
    use_cache: bool = True,
) -> tuple[List[Path], int, int]:
    _header("Phase 3: Removing Perceptual Duplicates")
    try:
        groups = find_perceptual_duplicates(
            files,
            threshold=threshold,
            index=index,
            use_cache=use_cache,
        )
    except Exception as exc:
        log.error("Perceptual duplicate detection failed: %s", exc)
        return files, 0, 0

    if not groups:
        _ok("No perceptual duplicates found.")
        return files, 0, 0

    removed: Set[Path] = set()
    reclaimed = 0
    group_count = 0

    for group in groups:
        paths = [Path(path) for path in group]
        if len(paths) < 2:
            continue

        group_count += 1
        print()
        print(_color(f"Perceptual duplicate group {group_count}:", "bold"))

        if dry_run:
            for number, path in enumerate(_rank_video_group(paths), start=1):
                print(
                    f"    {number}. {_format_video_label(path, recommended=(number == 1))}"
                )
            _warn("  Dry run: would prompt to choose which version to keep.")
            continue

        keep = _prompt_keep_choice(paths)
        if keep is None:
            _warn("  Skipped this group.")
            continue

        remove = [path for path in paths if path != keep]
        print(f"  {_color('KEEP', 'green')}   {_format_video_label(keep)}")
        for victim in remove:
            print(f"  {_color('REMOVE', 'red')} {_format_video_label(victim)}")

        if not _confirm(
            _color("  Remove the other duplicate(s) in this group? [Y/n]", "yellow")
        ):
            _warn("  Skipped this group.")
            continue

        reclaimed += _trash_files(remove, trash_dir, log, library_root=library_root)
        removed.update(remove)
        if index is not None:
            index.remove_many(remove)

    if group_count == 0:
        _ok("No perceptual duplicates found.")
    elif dry_run:
        _warn("Dry run: no files were moved.")
    elif removed:
        _ok(
            f"Removed {len(removed)} perceptual duplicate(s); "
            f"reclaimed {format_size(reclaimed)}."
        )
    else:
        _warn("No perceptual duplicates were removed.")

    remaining = [p for p in files if p not in removed]
    return remaining, len(removed), reclaimed


def _phase_standardization(
    files: List[Path],
    trash_dir: Path,
    dry_run: bool,
    keep_originals: bool,
    log: logging.Logger,
    video_index: Optional[VideoIndex] = None,
    *,
    library_root: Optional[Path] = None,
    conversion_settings: ConversionSettings,
) -> tuple[List[Path], int, int]:
    _header("Phase 4: Standardizing Movies")

    candidates: List[Path] = []
    for path in files:
        try:
            if needs_conversion(path, conversion_settings):
                candidates.append(path)
        except Exception as exc:
            log.error("needs_conversion failed for %s: %s", path, exc)

    if not candidates:
        _ok("Every remaining file already matches the standard filename and MP4 format.")
        return files, 0, 0

    budget_note = (
        f", max {conversion_settings.max_file_size_mb:g} MB/file"
        if conversion_settings.uses_size_budget()
        else ""
    )
    print(
        _color(
            f"Standardizing {len(candidates)} file(s) one at a time "
            f"(rename + H.264/AAC MP4 at CRF {conversion_settings.crf}, "
            f"preset {conversion_settings.preset}{budget_note}; no prompts) ...",
            "bold",
        )
    )

    if dry_run:
        for step, path in enumerate(candidates, start=1):
            _warn(f"  [{step}/{len(candidates)}] Dry run: would standardize {path.name}")
        _warn("Dry run: no standardizations performed.")
        return files, 0, 0

    path_map = {path: path for path in files}
    changed = 0
    bytes_delta = 0

    for step, src in enumerate(candidates, start=1):
        if needs_container_or_codec_work(src) and not video_is_readable(src, quiet=True):
            _warn(
                f"  [{step}/{len(candidates)}] Skipped unreadable file: {src.name} "
                "(ffprobe cannot read it; file may be incomplete or corrupt)"
            )
            continue

        print(f"  [{step}/{len(candidates)}] {src.name} ...", flush=True)
        try:
            output = standardize_video(
                src,
                trash_dir=trash_dir,
                library_root=library_root,
                keep_originals=keep_originals,
                settings=conversion_settings,
            )
        except Exception as exc:
            log.error("Standardization failed for %s: %s", src, exc)
            _err(f"  [{step}/{len(candidates)}] Failed to standardize {src.name}")
            continue

        if output is None:
            _err(f"  [{step}/{len(candidates)}] Failed to standardize {src.name}")
            continue

        pre_size = _safe_size(src)
        changed += 1
        bytes_delta += pre_size - _safe_size(output)

        for original, current in list(path_map.items()):
            if current == src:
                path_map[original] = output

        if output.name != src.name:
            _ok(f"  [{step}/{len(candidates)}] {src.name} -> {output.name}")
        else:
            _ok(f"  [{step}/{len(candidates)}] {src.name}")

        if video_index is not None:
            video_index.remove(src)
            video_index.ensure_metadata(output)

    updated_files = list(dict.fromkeys(path_map.values()))

    if changed:
        _ok(
            f"Standardized {changed} file(s); net size change: "
            f"{format_size(bytes_delta)} "
            f"{'saved' if bytes_delta >= 0 else 'grew'}."
        )
    else:
        _warn("No files were standardized.")

    return updated_files, changed, bytes_delta


def _format_part_merge_group(group: PartMergeGroup) -> str:
    parts = []
    for info in group.parts:
        if not video_is_readable(info.file, quiet=True):
            parts.append(f"{info.file.name} (unreadable/corrupt)")
            continue
        meta = get_video_metadata(info.file, quiet=True) or {}
        duration = format_duration(meta.get("duration_sec"))
        parts.append(f"{info.file.name} ({duration})")
    return " + ".join(parts)


def _phase_combine_parts(
    files: List[Path],
    trash_dir: Path,
    dry_run: bool,
    log: logging.Logger,
    video_index: Optional[VideoIndex] = None,
    *,
    library_root: Optional[Path] = None,
) -> tuple[List[Path], int]:
    _header("Phase 5: Combining Split Parts")

    groups = find_part_merge_groups(files)
    if not groups:
        _ok("No contiguous split-part groups found.")
        return files, 0

    merged_paths: List[Path] = []
    removed: Set[Path] = set()
    group_count = 0

    for group in groups:
        group_count += 1
        merged_name = f"{group.merged_stem()}.mp4"
        total_duration = format_duration(group.total_duration())

        print()
        print(_color(f"Split-part merge group {group_count}:", "bold"))
        print(f"  {_format_part_merge_group(group)}")
        print(
            f"  {_color('MERGE', 'magenta')} -> {merged_name} "
            f"({total_duration} total)"
        )

        bad_parts = unreadable_parts(group)
        if bad_parts:
            for path in bad_parts:
                _err(f"  Unreadable/corrupt part: {path.name}")
            _warn(
                "  Skipping merge. Re-download or repair the bad part(s) "
                "before trying again."
            )
            continue

        if dry_run:
            _warn("  Dry run: would prompt to merge this group.")
            continue

        if not _confirm(_color("  Merge these contiguous parts? [Y/n]", "yellow")):
            _warn("  Skipped this group.")
            continue

        try:
            output = merge_part_group(
                group,
                trash_dir=trash_dir,
                library_root=library_root,
                dry_run=False,
            )
        except Exception as exc:
            log.error("Part merge failed for %s: %s", merged_name, exc)
            _err(f"  Failed to merge into {merged_name}")
            continue

        if output is None:
            _err(f"  Failed to merge into {merged_name}")
            continue

        merged_paths.append(output)
        removed.update(group.source_files)
        if video_index is not None:
            video_index.remove_many(group.source_files)
            video_index.ensure_metadata(output)

        _ok(f"  Merged -> {output}")

    if group_count == 0:
        _ok("No contiguous split-part groups found.")
    elif dry_run:
        _warn("Dry run: no split parts were merged.")
    elif merged_paths:
        _ok(f"Merged {len(merged_paths)} split-part group(s).")
    else:
        _warn("No split-part groups were merged.")

    remaining = [path for path in files if path not in removed]
    remaining.extend(merged_paths)
    return remaining, len(merged_paths)


def _phase_tag_suggestions(
    files: List[Path],
    dry_run: bool,
    log: logging.Logger,
    video_index: Optional[VideoIndex] = None,
    *,
    use_cache: bool = True,
    min_samples: int = 3,
    threshold: int = 12,
    apply_tags: bool = False,
) -> int:
    _header("Phase 6: Suggesting Bracket Tags")

    tagged = [path for path in files if extract_leading_tag(path.name)]
    untagged = [path for path in files if is_untagged_video(path)]
    print(
        f"Tagged movies: {len(tagged)} | Untagged movies: {len(untagged)} | "
        f"Learning from tags with at least {min_samples} tagged work(s)."
    )

    if len(tagged) < min_samples:
        _warn("Not enough tagged movies to learn tag profiles.")
        return 0

    try:
        signatures = collect_video_signatures(
            files,
            video_index,
            use_cache=use_cache,
        )
    except Exception as exc:
        log.error("Tag signature collection failed: %s", exc)
        _err(f"Tag signature collection failed: {exc}")
        return 0

    profiles = build_tag_profiles(signatures, min_samples=min_samples)
    if not profiles:
        _warn("No tag profiles met the minimum sample count.")
        return 0

    print(_color(f"Learned {len(profiles)} tag profile(s).", "bold"))
    suggestions = suggest_tags_for_untagged(
        untagged,
        signatures,
        profiles,
        threshold=threshold,
    )

    if not suggestions:
        _ok("No confident tag matches found for untagged movies.")
        return 0

    applied = 0
    for path, suggestion in suggestions:
        proposed = proposed_tagged_name(path.name, suggestion.tag)
        print()
        print(_color("Untagged movie:", "bold"))
        print(f"  {path.name}")
        print(
            f"  {_color('SUGGEST', 'magenta')} [{suggestion.tag}] "
            f"(confidence: {suggestion.confidence}, "
            f"distance: {suggestion.avg_distance:.1f}, "
            f"metadata: {suggestion.metadata_score:.2f}, "
            f"from {suggestion.sample_count} tagged work(s), "
            f"margin: {suggestion.margin:.1f})"
        )
        print(f"  proposed: {proposed}")

        if dry_run:
            _warn("  Dry run: would suggest this tag.")
            continue

        if not apply_tags:
            continue

        if not _confirm(_color(f"  Rename to [{suggestion.tag}] ...? [Y/n]", "yellow")):
            _warn("  Skipped.")
            continue

        destination = _unique_file_path(path.parent, proposed)
        try:
            path.rename(destination)
        except OSError as exc:
            log.error("Failed renaming %s -> %s: %s", path, destination, exc)
            _err(f"  Failed to rename {path.name}")
            continue

        applied += 1
        if video_index is not None:
            video_index.remove(path)
            video_index.ensure_metadata(destination)
        _ok(f"  Renamed -> {destination.name}")

    if dry_run:
        _warn("Dry run: no files were renamed.")
    elif apply_tags:
        _ok(f"Applied {applied} tag(s).")
    else:
        _ok(
            f"Found {len(suggestions)} tag suggestion(s). "
            "Re-run with --apply-tags to rename matched files."
        )

    return len(suggestions)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    log = setup_logging(args.log_file)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    exclusive_modes = sum(
        bool(flag)
        for flag in (
            args.tags_only,
            args.standardize_only,
            args.combine_parts_only,
        )
    )
    if exclusive_modes > 1:
        _err(
            "Choose only one of --tags-only, --standardize-only, "
            "or --combine-parts-only."
        )
        return 2

    root = Path(args.path).expanduser().resolve()
    if not root.exists():
        _err(f"Path does not exist: {root}")
        return 2
    if not root.is_dir():
        _err(f"Path is not a directory: {root}")
        return 2
    if args.max_file_size_mb < 0:
        _err("--max-file-size-mb must be 0 or greater.")
        return 2
    if not 1 <= args.crf <= 51:
        _err("--crf must be between 1 and 51.")
        return 2

    conversion_settings = ConversionSettings(
        crf=args.crf,
        preset=args.preset,
        max_file_size_mb=args.max_file_size_mb,
    )

    trash_dir = root / ".movie_keeper_trash"
    index_path = index_file_for(root)
    use_cache = not args.rebuild_cache
    index = VideoIndex.load(root, rebuild=not use_cache)

    print(_color("movie_keeper", "bold", "magenta"))
    print(f"  root:      {root}")
    print(f"  trash:     {trash_dir}")
    print(f"  index:     {index_path}")
    print(f"  threshold: {args.threshold}")
    print(f"  dry-run:   {args.dry_run}")
    print(f"  keep-originals: {args.keep_originals}")
    print(f"  crf:       {conversion_settings.crf}")
    print(f"  preset:    {conversion_settings.preset}")
    if conversion_settings.uses_size_budget():
        print(f"  max-file-size: {conversion_settings.max_file_size_mb:g} MB")
    print(f"  cache:     {'rebuild' if not use_cache else f'{len(index)} record(s)'}")
    if args.suggest_tags or args.tags_only:
        print(
            f"  tags:      min-samples={args.tag_min_samples}, "
            f"threshold={args.tag_threshold}"
        )

    _header("Phase 1: Scanning Directory")
    try:
        files = scan_directory(root, index=index, use_cache=use_cache)
    except Exception as exc:
        log.error("Scan failed: %s", exc)
        _err(f"Scan failed: {exc}")
        return 1

    initial_count = len(files)
    print(_color(f"Discovered {initial_count} video file(s).", "bold"))
    if initial_count == 0:
        if not args.dry_run:
            index.save()
        _warn("Nothing to do.")
        return 0

    exact_removed = perceptual_removed = 0
    exact_bytes = perceptual_bytes = 0
    standardized = standardized_bytes_delta = 0
    parts_merged = 0
    tag_suggestions = 0
    run_combine_parts = args.combine_parts_only or (
        not args.skip_combine_parts and not args.tags_only
    )

    try:
        if args.combine_parts_only:
            files, parts_merged = _phase_combine_parts(
                files,
                trash_dir,
                args.dry_run,
                log,
                video_index=index,
                library_root=root,
            )
        elif args.standardize_only:
            files, standardized, standardized_bytes_delta = _phase_standardization(
                files,
                trash_dir,
                args.dry_run,
                args.keep_originals,
                log,
                video_index=index,
                library_root=root,
                conversion_settings=conversion_settings,
            )
            if run_combine_parts:
                files, parts_merged = _phase_combine_parts(
                    files,
                    trash_dir,
                    args.dry_run,
                    log,
                    video_index=index,
                    library_root=root,
                )
        elif not args.tags_only:
            files, exact_removed, exact_bytes = _phase_exact_duplicates(
                files,
                trash_dir,
                args.dry_run,
                log,
                index,
                library_root=root,
                use_cache=use_cache,
            )

            files, perceptual_removed, perceptual_bytes = _phase_perceptual_duplicates(
                files,
                args.threshold,
                trash_dir,
                args.dry_run,
                log,
                index,
                library_root=root,
                use_cache=use_cache,
            )

            files, standardized, standardized_bytes_delta = _phase_standardization(
                files,
                trash_dir,
                args.dry_run,
                args.keep_originals,
                log,
                video_index=index,
                library_root=root,
                conversion_settings=conversion_settings,
            )

            if run_combine_parts:
                files, parts_merged = _phase_combine_parts(
                    files,
                    trash_dir,
                    args.dry_run,
                    log,
                    video_index=index,
                    library_root=root,
                )

        if args.suggest_tags or args.tags_only:
            tag_suggestions = _phase_tag_suggestions(
                files,
                args.dry_run,
                log,
                index,
                use_cache=use_cache,
                min_samples=args.tag_min_samples,
                threshold=args.tag_threshold,
                apply_tags=args.apply_tags,
            )
    except KeyboardInterrupt:
        print()
        _warn("Interrupted by user. Saving index before exit.")
        if not args.dry_run:
            index.save()
        return 130

    if not args.dry_run:
        index.save()

    _header("Summary")
    total_reclaimed = (
        exact_bytes + perceptual_bytes + max(standardized_bytes_delta, 0)
    )
    print(f"  Files discovered:         {initial_count}")
    print(f"  Exact duplicates removed: {exact_removed} ({format_size(exact_bytes)})")
    print(
        f"  Perceptual duplicates:    {perceptual_removed} "
        f"({format_size(perceptual_bytes)})"
    )
    print(
        f"  Files standardized:       {standardized} "
        f"(net {format_size(standardized_bytes_delta)})"
    )
    if run_combine_parts:
        print(f"  Split-part groups merged: {parts_merged}")
    if args.suggest_tags or args.tags_only:
        print(f"  Tag suggestions:          {tag_suggestions}")
    print(
        _color(f"  Approx. space saved:      {format_size(total_reclaimed)}", "green", "bold")
    )
    if args.dry_run:
        _warn("This was a dry run — no files were modified.")
    else:
        print(f"  Trash directory:          {trash_dir}")
        print(f"  Index file:               {index_path}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        _warn("Interrupted by user. Exiting cleanly.")
        sys.exit(130)
