"""Standard filename conventions for movie libraries."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')
TRAILING_COPY_NUMBER = re.compile(r"\(\s*(\d+)\s*\)\s*$")
STANDALONE_COPY = re.compile(r"(?:^|\s)copy(?:\s|$)", re.IGNORECASE)
STANDALONE_DUPLICATE = re.compile(r"(?:^|\s)duplicate(?:\s|$)", re.IGNORECASE)

_BRACKET_CHAR_MAP = str.maketrans(
    {
        "【": "[",
        "】": "]",
        "［": "[",
        "］": "]",
        "「": "[",
        "」": "]",
        "『": "[",
        "』": "]",
        "（": "(",
        "）": ")",
    }
)

_BRACKET_OPEN = "(["
_BRACKET_CLOSE = ")]"
_BRACKET_OPEN_RE = re.compile(
    rf"(?<=[^{re.escape(_BRACKET_OPEN)}\s])([{re.escape(_BRACKET_OPEN)}])"
)
_BRACKET_CLOSE_RE = re.compile(
    rf"([{re.escape(_BRACKET_CLOSE)}])(?=[^{re.escape(_BRACKET_CLOSE + _BRACKET_OPEN)}\s])"
)


def _normalize_bracket_characters(name: str) -> str:
    return name.translate(_BRACKET_CHAR_MAP)


def _normalize_bracket_spacing(name: str) -> str:
    spaced = _BRACKET_OPEN_RE.sub(r" \1", name)
    spaced = _BRACKET_CLOSE_RE.sub(r"\1 ", spaced)
    return spaced


def _remove_noise_tokens(name: str) -> str:
    cleaned = STANDALONE_COPY.sub(" ", name)
    return STANDALONE_DUPLICATE.sub(" ", cleaned)


def _collapse_spaces(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip(" .")


def _remove_trailing_copy_number(name: str) -> str:
    match = TRAILING_COPY_NUMBER.search(name)
    if not match:
        return name
    digits = match.group(1)
    if len(digits) == 4 and digits.startswith(("19", "20")):
        return name
    return name[: match.start()].strip()


def standardize_stem(name: str) -> str:
    """Return a cleaned, filesystem-safe filename stem."""
    cleaned = INVALID_FILENAME_CHARS.sub("", name.strip())
    cleaned = _remove_trailing_copy_number(cleaned).strip()
    cleaned = _remove_noise_tokens(cleaned)
    cleaned = _normalize_bracket_characters(cleaned)
    cleaned = _normalize_bracket_spacing(cleaned)
    cleaned = _collapse_spaces(cleaned)
    return cleaned or name.strip()


def standard_file_stem(path: PathLike) -> str:
    return standardize_stem(Path(path).stem)


def standard_file_name(path: PathLike, *, extension: str = ".mp4") -> str:
    return f"{standard_file_stem(path)}{extension}"


def file_name_is_standard(path: PathLike, *, extension: str = ".mp4") -> bool:
    target = Path(path)
    return target.name == standard_file_name(target, extension=extension)
