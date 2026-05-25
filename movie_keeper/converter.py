"""Convert assorted video formats to a unified H.264/AAC MP4."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

from .naming import file_name_is_standard, standard_file_name
from .utils import _run_ffprobe, move_to_trash, video_is_readable

PathLike = Union[str, Path]

log = logging.getLogger(__name__)

TARGET_VIDEO_CODEC = "h264"
TARGET_AUDIO_CODEC = "aac"
TARGET_EXTENSION = ".mp4"
DEFAULT_CRF = 18
DEFAULT_PRESET = "medium"


@dataclass(frozen=True)
class ConversionSettings:
    crf: int = DEFAULT_CRF
    preset: str = DEFAULT_PRESET
    max_file_size_mb: float = 0.0
    scale_factor: float = 1.0

    def uses_size_budget(self) -> bool:
        return self.max_file_size_mb > 0


DEFAULT_CONVERSION_SETTINGS = ConversionSettings()


def _stream_codecs(file_path: PathLike) -> Tuple[Optional[str], Optional[str]]:
    data = _run_ffprobe(
        ["-show_entries", "stream=codec_name,codec_type"],
        file_path,
    )
    if not data:
        return None, None

    v_codec: Optional[str] = None
    a_codec: Optional[str] = None
    for stream in data.get("streams", []) or []:
        ctype = stream.get("codec_type")
        cname = stream.get("codec_name")
        if ctype == "video" and v_codec is None:
            v_codec = cname
        elif ctype == "audio" and a_codec is None:
            a_codec = cname

    return v_codec, a_codec


def _normalize_codec(codec: Optional[str]) -> Optional[str]:
    return codec.lower() if codec else None


def _codecs_are_target(file_path: PathLike) -> bool:
    v_codec, a_codec = _stream_codecs(file_path)
    if v_codec is None:
        return False
    if _normalize_codec(v_codec) != TARGET_VIDEO_CODEC:
        return False
    if a_codec is not None and _normalize_codec(a_codec) != TARGET_AUDIO_CODEC:
        return False
    return True


def needs_codec_conversion(file_path: PathLike) -> bool:
    path = Path(file_path)
    if not video_is_readable(path):
        return False

    if path.suffix.lower() != TARGET_EXTENSION:
        return True

    return not _codecs_are_target(path)


def needs_container_or_codec_work(file_path: PathLike) -> bool:
    """True when the file must be remuxed/transcoded, not just renamed."""
    path = Path(file_path)
    if path.suffix.lower() != TARGET_EXTENSION:
        return True
    return needs_codec_conversion(path)


def needs_rename(file_path: PathLike) -> bool:
    return not file_name_is_standard(file_path, extension=TARGET_EXTENSION)


def needs_conversion(
    file_path: PathLike,
    settings: ConversionSettings = DEFAULT_CONVERSION_SETTINGS,
) -> bool:
    del settings
    return needs_rename(file_path) or needs_codec_conversion(file_path)


def _unique_output_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    counter = 2
    while candidate.exists():
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        candidate = directory / f"{stem}__{counter}{suffix}"
        counter += 1
    return candidate


def _temporary_output_path(directory: Path, suffix: str) -> Path:
    candidate = directory / f".movie_keeper_convert{suffix}"
    counter = 1
    while candidate.exists():
        candidate = directory / f".movie_keeper_convert_{counter}{suffix}"
        counter += 1
    return candidate


def _conversion_work_path(
    src: Path,
    target: Path,
    target_name: str,
    *,
    keep_originals: bool,
) -> Path:
    """Pick a writable ffmpeg output path that avoids clobbering the source."""
    if target.resolve() == src.resolve():
        if keep_originals:
            return _unique_output_path(src.parent, target_name)
        return _temporary_output_path(src.parent, target.suffix)

    if target.exists():
        return _unique_output_path(src.parent, target_name)
    return target


def _build_ffmpeg_cmd(
    src: Path,
    output: Path,
    *,
    settings: ConversionSettings,
    remux_only: bool = False,
) -> List[str]:
    if remux_only:
        return [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-map",
            "0:v:0?",
            "-map",
            "0:a?",
            "-map",
            "0:s?",
            "-c",
            "copy",
            "-c:s",
            "mov_text",
            str(output),
        ]

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-map",
        "0:v:0?",
        "-map",
        "0:a?",
        "-map",
        "0:s?",
        "-c:v",
        "libx264",
        "-crf",
        str(settings.crf),
        "-preset",
        settings.preset,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-c:s",
        "mov_text",
    ]
    if settings.scale_factor < 0.999:
        cmd.extend(["-vf", f"scale=trunc(iw*{settings.scale_factor}/2)*2:-2"])
    cmd.append(str(output))
    return cmd


def _run_ffmpeg(cmd: List[str], src: Path) -> bool:
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
        log.error("ffmpeg not found on PATH; cannot convert %s", src)
        return False
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        log.error("ffmpeg conversion failed for %s: %s", src, stderr or exc)
        return False
    return True


def _fits_size_budget(path: Path, settings: ConversionSettings) -> bool:
    if not settings.uses_size_budget():
        return True
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
    except OSError:
        return False
    return size_mb <= settings.max_file_size_mb


def convert_to_mp4(
    input_path: PathLike,
    output_path: PathLike,
    *,
    settings: ConversionSettings = DEFAULT_CONVERSION_SETTINGS,
) -> Optional[Path]:
    """Convert or remux ``input_path`` to ``output_path``."""
    src = Path(input_path)
    output = Path(output_path)
    if not src.exists() or not src.is_file():
        log.warning("Cannot convert missing file: %s", src)
        return None

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.resolve() == src.resolve():
        log.error(
            "convert_to_mp4 output must differ from input for %s; "
            "use standardize_video for in-place conversion",
            src,
        )
        return None

    remux_only = src.suffix.lower() != TARGET_EXTENSION and _codecs_are_target(src)

    working_settings = settings
    max_attempts = 10 if settings.uses_size_budget() else 1
    for attempt in range(max_attempts):
        cmd = _build_ffmpeg_cmd(
            src,
            output,
            settings=working_settings,
            remux_only=remux_only and attempt == 0,
        )
        if _run_ffmpeg(cmd, src) and output.exists() and output.stat().st_size > 0:
            if _fits_size_budget(output, settings):
                return output
        if output.exists():
            try:
                output.unlink()
            except OSError:
                pass
        if not settings.uses_size_budget():
            break
        if working_settings.crf < 32:
            working_settings = ConversionSettings(
                crf=min(working_settings.crf + 2, 32),
                preset=working_settings.preset,
                max_file_size_mb=working_settings.max_file_size_mb,
                scale_factor=working_settings.scale_factor,
            )
        elif working_settings.scale_factor > 0.5:
            working_settings = ConversionSettings(
                crf=working_settings.crf,
                preset=working_settings.preset,
                max_file_size_mb=working_settings.max_file_size_mb,
                scale_factor=working_settings.scale_factor * 0.9,
            )
        else:
            break
        remux_only = False

    return None


def standardize_video(
    input_path: PathLike,
    *,
    trash_dir: Path,
    library_root: Optional[Path] = None,
    keep_originals: bool = False,
    settings: ConversionSettings = DEFAULT_CONVERSION_SETTINGS,
) -> Optional[Path]:
    """Rename and/or convert a video to the standard MP4 convention."""
    src = Path(input_path)
    if not src.exists() or not src.is_file():
        return None

    target_name = standard_file_name(src, extension=TARGET_EXTENSION)
    target = src.parent / target_name

    if not needs_rename(src) and not needs_container_or_codec_work(src):
        return src

    if needs_container_or_codec_work(src):
        if not video_is_readable(src, quiet=True):
            log.warning("Cannot read %s; skipping standardization", src)
            return None

        in_place = target.resolve() == src.resolve()
        work_path = _conversion_work_path(
            src,
            target,
            target_name,
            keep_originals=keep_originals,
        )

        output = convert_to_mp4(src, work_path, settings=settings)
        if output is None:
            return None

        if in_place and not keep_originals:
            moved = move_to_trash(src, trash_dir=trash_dir, library_root=library_root)
            if moved is None:
                log.error("Failed to trash original before replacing %s", src)
                try:
                    output.unlink()
                except OSError:
                    pass
                return None
            try:
                output.rename(target)
            except OSError as exc:
                log.error("Failed replacing %s with converted file: %s", target, exc)
                return None
            return target

        if not keep_originals and src.resolve() != output.resolve():
            move_to_trash(src, trash_dir=trash_dir, library_root=library_root)
        return output

    if src.name == target_name:
        return src

    if target.exists() and target.resolve() != src.resolve():
        target = _unique_output_path(src.parent, target_name)
    try:
        src.rename(target)
    except OSError as exc:
        log.error("Failed renaming %s -> %s: %s", src, target, exc)
        return None
    return target


def _standardize_one(
    path: Path,
    *,
    trash_dir: Path,
    library_root: Optional[Path],
    keep_originals: bool,
    settings: ConversionSettings,
) -> Tuple[Path, Optional[Path]]:
    try:
        output = standardize_video(
            path,
            trash_dir=trash_dir,
            library_root=library_root,
            keep_originals=keep_originals,
            settings=settings,
        )
    except Exception as exc:
        log.error("Standardization failed for %s: %s", path, exc)
        return path, None
    return path, output


def standardize_all(
    file_paths: Iterable[PathLike],
    *,
    trash_dir: Path,
    library_root: Optional[Path] = None,
    keep_originals: bool = False,
    settings: ConversionSettings = DEFAULT_CONVERSION_SETTINGS,
) -> List[Tuple[Path, Optional[Path]]]:
    """Standardize videos one at a time."""
    results: List[Tuple[Path, Optional[Path]]] = []
    for path in file_paths:
        results.append(
            _standardize_one(
                Path(path),
                trash_dir=trash_dir,
                library_root=library_root,
                keep_originals=keep_originals,
                settings=settings,
            )
        )
    return results


def convert_all(
    file_paths: Iterable[PathLike],
    dry_run: bool = False,
    keep_originals: bool = False,
    trash_dir: Optional[PathLike] = None,
    *,
    settings: ConversionSettings = DEFAULT_CONVERSION_SETTINGS,
    library_root: Optional[Path] = None,
) -> List[Tuple[Path, Optional[Path]]]:
    """Iterate through files and standardize any that need it."""
    if dry_run:
        return [(Path(p), None) for p in file_paths]

    return standardize_all(
        file_paths,
        trash_dir=Path(trash_dir) if trash_dir is not None else Path("."),
        library_root=library_root,
        keep_originals=keep_originals,
        settings=settings,
    )
