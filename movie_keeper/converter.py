"""Convert assorted video formats to a unified H.264/AAC MP4."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

from .utils import _run_ffprobe, move_to_trash

PathLike = Union[str, Path]

log = logging.getLogger(__name__)

TARGET_VIDEO_CODEC = "h264"
TARGET_AUDIO_CODEC = "aac"
TARGET_EXTENSION = ".mp4"


def _stream_codecs(file_path: PathLike) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(video_codec, audio_codec)`` for the first streams of each."""
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


def needs_conversion(file_path: PathLike) -> bool:
    """True if the file is not already an H.264/AAC MP4."""
    path = Path(file_path)
    if path.suffix.lower() != TARGET_EXTENSION:
        return True

    v_codec, a_codec = _stream_codecs(path)
    if v_codec is None:
        # Couldn't probe — leave it alone rather than risk a destructive convert.
        log.warning("Cannot determine codecs for %s; not converting", path)
        return False

    if v_codec.lower() != TARGET_VIDEO_CODEC:
        return True
    if a_codec is not None and a_codec.lower() != TARGET_AUDIO_CODEC:
        return True
    return False


def convert_to_mp4(
    input_path: PathLike,
    output_dir: Optional[PathLike] = None,
) -> Optional[Path]:
    """Convert ``input_path`` to an MP4 (H.264 + AAC, optional mov_text subs).

    Returns the output path on success or ``None`` on failure. The output is
    written next to the source unless ``output_dir`` is supplied.
    """
    src = Path(input_path)
    if not src.exists() or not src.is_file():
        log.warning("Cannot convert missing file: %s", src)
        return None

    target_dir = Path(output_dir) if output_dir is not None else src.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    base_name = src.stem
    output = target_dir / f"{base_name}{TARGET_EXTENSION}"
    if output.resolve() == src.resolve():
        # In-place conversion would clobber the source; pick a safe name.
        output = target_dir / f"{base_name}.converted{TARGET_EXTENSION}"

    counter = 1
    while output.exists():
        output = target_dir / f"{base_name}.converted_{counter}{TARGET_EXTENSION}"
        counter += 1

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-stats",
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
        "18",
        "-preset",
        "slow",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-c:s",
        "mov_text",
        str(output),
    ]

    print(f"Converting: {src.name} -> {output.name}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        log.error("ffmpeg not found on PATH; cannot convert %s", src)
        return None
    except subprocess.CalledProcessError as exc:
        log.error("ffmpeg conversion failed for %s: %s", src, exc)
        if output.exists():
            try:
                output.unlink()
            except OSError:
                pass
        return None

    if not output.exists() or output.stat().st_size == 0:
        log.error("Conversion produced empty output for %s", src)
        return None

    return output


def convert_all(
    file_paths: Iterable[PathLike],
    dry_run: bool = False,
    keep_originals: bool = False,
    trash_dir: Optional[PathLike] = None,
) -> List[Tuple[Path, Optional[Path]]]:
    """Iterate through files and convert any that need it.

    Returns a list of ``(source, output_or_None)`` tuples. When ``dry_run`` is
    true no conversion happens and the output is reported as ``None``. After a
    successful conversion the original is moved to trash unless
    ``keep_originals`` is set.
    """
    results: List[Tuple[Path, Optional[Path]]] = []
    paths = [Path(p) for p in file_paths]
    print(f"Evaluating {len(paths)} file(s) for conversion ...")

    for index, path in enumerate(paths, start=1):
        prefix = f"[{index}/{len(paths)}]"
        if not path.exists():
            log.warning("%s missing, skipping: %s", prefix, path)
            continue

        if not needs_conversion(path):
            print(f"{prefix} OK (already H.264/AAC MP4): {path.name}")
            results.append((path, None))
            continue

        if dry_run:
            print(f"{prefix} DRY RUN — would convert: {path.name}")
            results.append((path, None))
            continue

        output = convert_to_mp4(path)
        if output is None:
            print(f"{prefix} FAILED to convert: {path.name}")
            results.append((path, None))
            continue

        print(f"{prefix} converted -> {output.name}")
        if not keep_originals:
            move_to_trash(path, trash_dir=trash_dir)
        results.append((path, output))

    return results
