"""Persistent index for scanned and processed videos."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union

from .utils import get_video_metadata

PathLike = Union[str, Path]

INDEX_VERSION = 1
INDEX_DIR_NAME = ".movie_keeper"
INDEX_FILE_NAME = "index.json"
DEFAULT_PERCEPTUAL_FRAMES = 15

log = logging.getLogger(__name__)


def index_dir_for(root: PathLike) -> Path:
    return Path(root).resolve() / INDEX_DIR_NAME


def index_file_for(root: PathLike) -> Path:
    return index_dir_for(root) / INDEX_FILE_NAME


def compute_quick_fingerprint(path: PathLike) -> str:
    """Cheap change detector based on filesystem metadata only."""
    target = Path(path)
    try:
        stat = target.stat()
    except OSError:
        return ""
    payload = f"file:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_content_fingerprint(path: PathLike) -> str:
    """Stronger change detector used to validate cached hashes/signatures."""
    target = Path(path)
    if not target.exists() or not target.is_file():
        return ""
    try:
        stat = target.stat()
    except OSError:
        return ""
    return hashlib.sha256(f"{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")).hexdigest()


@dataclass
class VideoRecord:
    path: str
    quick_fingerprint: str
    content_fingerprint: str
    exact_hash: Optional[str] = None
    duration_sec: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    bit_rate: Optional[int] = None
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    perceptual_hashes: List[str] = field(default_factory=list)
    perceptual_num_frames: int = DEFAULT_PERCEPTUAL_FRAMES
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VideoRecord:
        return cls(
            path=data["path"],
            quick_fingerprint=data.get("quick_fingerprint", ""),
            content_fingerprint=data.get("content_fingerprint", ""),
            exact_hash=data.get("exact_hash"),
            duration_sec=data.get("duration_sec"),
            width=data.get("width"),
            height=data.get("height"),
            bit_rate=data.get("bit_rate"),
            video_codec=data.get("video_codec"),
            audio_codec=data.get("audio_codec"),
            perceptual_hashes=list(data.get("perceptual_hashes") or []),
            perceptual_num_frames=int(
                data.get("perceptual_num_frames") or DEFAULT_PERCEPTUAL_FRAMES
            ),
            updated_at=data.get("updated_at", ""),
        )


class VideoIndex:
    def __init__(self, root: Path, records: Optional[Dict[str, VideoRecord]] = None) -> None:
        self.root = root.resolve()
        self._records: Dict[str, VideoRecord] = records or {}

    @classmethod
    def load(cls, root: PathLike, *, rebuild: bool = False) -> VideoIndex:
        root_path = Path(root).resolve()
        if rebuild:
            return cls(root_path)

        index_path = index_file_for(root_path)
        if not index_path.exists():
            return cls(root_path)

        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not read index %s: %s", index_path, exc)
            return cls(root_path)

        if payload.get("version") != INDEX_VERSION:
            log.warning("Index version mismatch; rebuilding cache.")
            return cls(root_path)

        if payload.get("root") != str(root_path):
            log.warning("Index root mismatch; rebuilding cache.")
            return cls(root_path)

        records = {
            key: VideoRecord.from_dict(value)
            for key, value in (payload.get("videos") or {}).items()
        }
        return cls(root_path, records)

    def get(self, path: PathLike) -> Optional[VideoRecord]:
        return self._records.get(str(Path(path).resolve()))

    def upsert(
        self,
        path: PathLike,
        *,
        quick_fingerprint: Optional[str] = None,
        content_fingerprint: Optional[str] = None,
        exact_hash: Optional[str] = None,
        duration_sec: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        bit_rate: Optional[int] = None,
        video_codec: Optional[str] = None,
        audio_codec: Optional[str] = None,
        perceptual_hashes: Optional[List[str]] = None,
        perceptual_num_frames: Optional[int] = None,
    ) -> VideoRecord:
        resolved = Path(path).resolve()
        key = str(resolved)
        record = self._records.get(key)

        if record is None:
            record = VideoRecord(
                path=key,
                quick_fingerprint=quick_fingerprint or compute_quick_fingerprint(resolved),
                content_fingerprint=content_fingerprint
                or compute_content_fingerprint(resolved),
            )
            self._records[key] = record

        if quick_fingerprint is not None:
            record.quick_fingerprint = quick_fingerprint
        if content_fingerprint is not None:
            record.content_fingerprint = content_fingerprint
        if exact_hash is not None:
            record.exact_hash = exact_hash
        if duration_sec is not None:
            record.duration_sec = duration_sec
        if width is not None:
            record.width = width
        if height is not None:
            record.height = height
        if bit_rate is not None:
            record.bit_rate = bit_rate
        if video_codec is not None:
            record.video_codec = video_codec
        if audio_codec is not None:
            record.audio_codec = audio_codec
        if perceptual_hashes is not None:
            record.perceptual_hashes = perceptual_hashes
        if perceptual_num_frames is not None:
            record.perceptual_num_frames = perceptual_num_frames

        record.updated_at = datetime.now(timezone.utc).isoformat()
        return record

    def remove(self, path: PathLike) -> None:
        self._records.pop(str(Path(path).resolve()), None)

    def remove_many(self, paths: Iterable[PathLike]) -> None:
        for path in paths:
            self.remove(path)

    def prune_to(self, discovered: Iterable[PathLike]) -> int:
        keep = {str(Path(path).resolve()) for path in discovered}
        stale = [key for key in self._records if key not in keep]
        for key in stale:
            del self._records[key]
        return len(stale)

    def save(self) -> None:
        index_path = index_file_for(self.root)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_VERSION,
            "root": str(self.root),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "videos": {key: asdict(record) for key, record in self._records.items()},
        }
        temp_path = index_path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(index_path)

    def ensure_metadata(self, path: PathLike, *, force: bool = False) -> VideoRecord:
        resolved = Path(path).resolve()
        record = self.get(resolved)
        quick = compute_quick_fingerprint(resolved)
        if (
            not force
            and record
            and record.quick_fingerprint == quick
            and record.width is not None
        ):
            return record

        meta = get_video_metadata(resolved, quiet=True) or {}
        return self.upsert(
            resolved,
            quick_fingerprint=quick,
            content_fingerprint=compute_content_fingerprint(resolved),
            duration_sec=meta.get("duration_sec"),
            width=meta.get("width"),
            height=meta.get("height"),
            bit_rate=meta.get("bit_rate"),
            video_codec=meta.get("video_codec"),
            audio_codec=meta.get("audio_codec"),
        )

    def is_scan_cache_hit(self, path: PathLike) -> bool:
        record = self.get(path)
        if record is None:
            return False
        return record.quick_fingerprint == compute_quick_fingerprint(path)

    def is_content_cache_hit(self, path: PathLike) -> bool:
        record = self.get(path)
        if record is None:
            return False
        return record.content_fingerprint == compute_content_fingerprint(path)

    def cached_paths(self) -> Set[str]:
        return set(self._records.keys())

    def __len__(self) -> int:
        return len(self._records)
