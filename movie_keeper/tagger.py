"""Bracket tag extraction and metadata-first tag suggestions."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .index import DEFAULT_PERCEPTUAL_FRAMES, compute_content_fingerprint
from .perceptual import (
    _best_match_details,
    _deserialize_phash,
    compute_video_signature,
)
from .utils import get_video_metadata

if TYPE_CHECKING:
    from .index import VideoIndex

PathLike = Union[str, Path]

_BRACKET_PAIRS = {
    "[": "]",
    "(": ")",
    "（": "）",
    "【": "】",
    "［": "］",
}

_METADATA_MARKERS = (
    "1080p",
    "720p",
    "2160p",
    "4k",
    "bluray",
    "webrip",
    "web-dl",
    "hdtv",
    "remux",
    "x264",
    "x265",
    "hevc",
    "aac",
    "dts",
    "english",
    "chinese",
    "subbed",
    "subtitles",
    "complete",
    "uncut",
    "director's cut",
    "directors cut",
)


@dataclass(frozen=True)
class TagSuggestion:
    tag: str
    avg_distance: float
    metadata_score: float
    good_ratio: float
    sample_count: int
    confidence: str
    margin: float


@dataclass
class TagProfile:
    tag: str
    durations: List[float] = field(default_factory=list)
    aspect_ratios: List[float] = field(default_factory=list)
    signatures: List[Sequence] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        return len(self.signatures)


def is_metadata_tag(tag: str) -> bool:
    lowered = tag.casefold()
    return any(marker.casefold() in lowered for marker in _METADATA_MARKERS)


def _extract_next_bracket_tag(name: str) -> tuple[Optional[str], str]:
    stripped = name.lstrip()
    if not stripped:
        return None, name

    opener = stripped[0]
    closer = _BRACKET_PAIRS.get(opener)
    if closer is None:
        return None, name

    close_index = stripped.find(closer, 1)
    if close_index == -1:
        return None, name

    tag = stripped[1:close_index].strip()
    remainder = stripped[close_index + 1 :].lstrip()
    return tag or None, remainder


def extract_leading_tag(name: str) -> Optional[str]:
    """Return the leading non-metadata bracket tag from a filename, if any."""
    remaining = Path(name).stem.strip()
    while remaining:
        tag, remaining = _extract_next_bracket_tag(remaining)
        if tag is None:
            return None
        if not is_metadata_tag(tag):
            return tag
    return None


def title_without_tag(name: str) -> str:
    remaining = Path(name).stem.strip()
    while remaining:
        tag, remaining = _extract_next_bracket_tag(remaining)
        if tag is None:
            break
    return remaining or Path(name).stem.strip()


def is_untagged_video(path: PathLike) -> bool:
    return extract_leading_tag(Path(path).name) is None


def _aspect_ratio(meta: dict) -> Optional[float]:
    width = meta.get("width") or 0
    height = meta.get("height") or 0
    if width and height:
        return float(width) / float(height)
    return None


def metadata_fit_score(
    *,
    duration_sec: Optional[float],
    aspect_ratio: Optional[float],
    profile: TagProfile,
) -> float:
    """Lower is better. Combines duration and aspect-ratio fit to a tag profile."""
    duration_penalty = 1.0
    if duration_sec and profile.durations:
        median_duration = statistics.median(profile.durations)
        if median_duration > 0:
            duration_penalty = abs(duration_sec - median_duration) / median_duration

    aspect_penalty = 1.0
    if aspect_ratio and profile.aspect_ratios:
        median_aspect = statistics.median(profile.aspect_ratios)
        aspect_penalty = abs(aspect_ratio - median_aspect)

    return duration_penalty + aspect_penalty


def collect_video_signatures(
    files: Iterable[PathLike],
    index: Optional["VideoIndex"] = None,
    *,
    use_cache: bool = True,
    num_frames: int = DEFAULT_PERCEPTUAL_FRAMES,
) -> Dict[Path, List]:
    signatures: Dict[Path, List] = {}
    for path in files:
        resolved = Path(path).resolve()
        record = index.get(resolved) if index is not None else None
        signature: Optional[List] = None

        if (
            use_cache
            and index is not None
            and record
            and record.perceptual_hashes
            and record.perceptual_num_frames == num_frames
            and index.is_content_cache_hit(resolved)
        ):
            signature = [
                deserialized
                for value in record.perceptual_hashes
                if (deserialized := _deserialize_phash(value)) is not None
            ]

        if not signature:
            signature = compute_video_signature(resolved, num_frames=num_frames)
            if signature and index is not None:
                meta = get_video_metadata(resolved) or {}
                index.upsert(
                    resolved,
                    content_fingerprint=compute_content_fingerprint(resolved),
                    duration_sec=meta.get("duration_sec"),
                    width=meta.get("width"),
                    height=meta.get("height"),
                    bit_rate=meta.get("bit_rate"),
                    video_codec=meta.get("video_codec"),
                    audio_codec=meta.get("audio_codec"),
                    perceptual_hashes=[str(value) for value in signature],
                    perceptual_num_frames=num_frames,
                )

        if signature:
            signatures[resolved] = signature

    return signatures


def build_tag_profiles(
    signatures: Dict[Path, Sequence],
    *,
    min_samples: int = 3,
) -> Dict[str, TagProfile]:
    profiles: Dict[str, TagProfile] = {}

    for path, signature in signatures.items():
        tag = extract_leading_tag(path.name)
        if not tag:
            continue

        meta = get_video_metadata(path) or {}
        profile = profiles.setdefault(tag, TagProfile(tag=tag))
        duration = meta.get("duration_sec")
        aspect = _aspect_ratio(meta)
        if duration:
            profile.durations.append(float(duration))
        if aspect:
            profile.aspect_ratios.append(aspect)
        profile.signatures.append(signature)

    return {
        tag: profile
        for tag, profile in profiles.items()
        if profile.sample_count >= min_samples
    }


def _perceptual_score(
    candidate: Sequence,
    profile_signatures: Sequence[Sequence],
) -> Tuple[float, float]:
    distances: List[float] = []
    ratios: List[float] = []
    for reference in profile_signatures:
        avg_distance, good_ratio = _best_match_details(candidate, reference)
        distances.append(avg_distance)
        ratios.append(good_ratio)
    return statistics.median(distances), statistics.median(ratios)


def _confidence_label(
    combined_score: float,
    metadata_score: float,
    good_ratio: float,
    sample_count: int,
    margin: float,
    threshold: int,
) -> str:
    if (
        combined_score <= max(8, threshold - 4)
        and metadata_score <= 0.25
        and good_ratio >= 0.6
        and sample_count >= 5
        and margin >= 2.0
    ):
        return "high"
    if combined_score <= threshold and metadata_score <= 0.45 and good_ratio >= 0.45:
        return "medium"
    if combined_score <= threshold and metadata_score <= 0.6 and good_ratio >= 0.35:
        return "low"
    return "none"


def suggest_tag_for_video(
    path: PathLike,
    signature: Sequence,
    profiles: Dict[str, TagProfile],
    *,
    threshold: int = 12,
) -> Optional[TagSuggestion]:
    if not profiles or not signature:
        return None

    meta = get_video_metadata(path) or {}
    duration = meta.get("duration_sec")
    aspect = _aspect_ratio(meta)

    scored: List[Tuple[str, float, float, float, float, int]] = []
    for tag, profile in profiles.items():
        metadata_score = metadata_fit_score(
            duration_sec=duration,
            aspect_ratio=aspect,
            profile=profile,
        )
        perceptual_distance, good_ratio = _perceptual_score(signature, profile.signatures)
        combined = perceptual_distance + (metadata_score * 10.0)
        scored.append(
            (tag, combined, metadata_score, good_ratio, perceptual_distance, profile.sample_count)
        )

    scored.sort(key=lambda item: (item[1], item[2], -item[3], -item[5], item[0].lower()))
    best = scored[0]
    second_combined = scored[1][1] if len(scored) > 1 else float("inf")
    margin = second_combined - best[1]

    confidence = _confidence_label(
        best[4],
        best[2],
        best[3],
        best[5],
        margin,
        threshold,
    )
    if confidence == "none":
        return None

    return TagSuggestion(
        tag=best[0],
        avg_distance=best[4],
        metadata_score=best[2],
        good_ratio=best[3],
        sample_count=best[5],
        confidence=confidence,
        margin=margin,
    )


def suggest_tags_for_untagged(
    files: Iterable[PathLike],
    signatures: Dict[Path, Sequence],
    profiles: Dict[str, TagProfile],
    *,
    threshold: int = 12,
) -> List[Tuple[Path, TagSuggestion]]:
    suggestions: List[Tuple[Path, TagSuggestion]] = []
    for path in files:
        resolved = Path(path).resolve()
        if not is_untagged_video(resolved):
            continue
        signature = signatures.get(resolved)
        if not signature:
            continue
        suggestion = suggest_tag_for_video(
            resolved,
            signature,
            profiles,
            threshold=threshold,
        )
        if suggestion is not None:
            suggestions.append((resolved, suggestion))

    suggestions.sort(
        key=lambda item: (
            {"high": 0, "medium": 1, "low": 2}.get(item[1].confidence, 3),
            item[1].avg_distance,
            item[0].name.lower(),
        )
    )
    return suggestions


def proposed_tagged_name(file_name: str, tag: str) -> str:
    stem = title_without_tag(file_name)
    suffix = Path(file_name).suffix
    return f"[{tag}] {stem}{suffix}"
