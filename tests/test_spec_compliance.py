"""Spec compliance checks for MovieKeeper."""

from __future__ import annotations

import tempfile
import unittest
import unittest.mock
from pathlib import Path

from movie_keeper.converter import (
    ConversionSettings,
    _conversion_work_path,
    needs_conversion,
)
from movie_keeper.naming import file_name_is_standard, standardize_stem
from movie_keeper.parts import find_part_merge_groups, parse_part_name, unreadable_parts
from movie_keeper.parts import PartMergeGroup, PartInfo
from movie_keeper.resolver import rank_videos_for_keep
from movie_keeper.tagger import (
    TagProfile,
    build_tag_profiles,
    extract_leading_tag,
    is_untagged_video,
    metadata_fit_score,
    suggest_tag_for_video,
)


class NamingTests(unittest.TestCase):
    def test_bracket_and_noise_cleanup(self) -> None:
        self.assertEqual(standardize_stem("【A24】Dune  (2021) copy"), "[A24] Dune (2021)")
        self.assertEqual(standardize_stem("Movie (1)"), "Movie")

    def test_standard_filename(self) -> None:
        path = Path("[A24] Dune (2021).mp4")
        self.assertTrue(file_name_is_standard(path))


class PartsTests(unittest.TestCase):
    def test_parse_patterns(self) -> None:
        self.assertEqual(parse_part_name("LOTR_part_1").part, 1)
        self.assertEqual(parse_part_name("Movie-cd2").part, 2)
        self.assertIsNone(parse_part_name("Movie_part_1to2"))

    def test_contiguous_only(self) -> None:
        files = [
            Path("Movie_part_1.mkv"),
            Path("Movie_part_3.mkv"),
        ]
        self.assertEqual(find_part_merge_groups(files), [])


class PartsReadabilityTests(unittest.TestCase):
    def test_unreadable_parts_detects_bad_files(self) -> None:
        good = Path("Movie-1.mp4")
        bad = Path("Movie-2.mp4")
        group = PartMergeGroup(
            parts=(
                PartInfo(file=good, prefix="Movie-", suffix="", part=1),
                PartInfo(file=bad, prefix="Movie-", suffix="", part=2),
            )
        )

        def fake_readable(path: Path, *, quiet: bool = True) -> bool:
            del quiet
            return path == good

        with unittest.mock.patch(
            "movie_keeper.parts.video_is_readable",
            side_effect=fake_readable,
        ):
            self.assertEqual(unreadable_parts(group), [bad])


class TaggerTests(unittest.TestCase):
    def test_metadata_tags_are_ignored(self) -> None:
        self.assertIsNone(extract_leading_tag("[1080p] Dune (2021).mp4"))
        self.assertEqual(extract_leading_tag("[A24] Dune (2021).mp4"), "A24")
        self.assertTrue(is_untagged_video(Path("[1080p] Dune (2021).mp4")))

    def test_metadata_fit_prefers_similar_duration(self) -> None:
        profile = TagProfile(
            tag="A24",
            durations=[7200.0, 7300.0],
            aspect_ratios=[1.777, 1.777],
            signatures=[],
        )
        close = metadata_fit_score(duration_sec=7250.0, aspect_ratio=1.777, profile=profile)
        far = metadata_fit_score(duration_sec=3600.0, aspect_ratio=1.333, profile=profile)
        self.assertLess(close, far)


class ResolverTests(unittest.TestCase):
    def test_duration_proximity_breaks_ties(self) -> None:
        group = [Path("short.mp4"), Path("best.mp4"), Path("long.mp4")]
        scores = {
            group[0]: (2073600, 8_000_000, 7000),
            group[1]: (2073600, 8_000_000, 7190),
            group[2]: (2073600, 8_000_000, 7300),
        }
        durations = {group[0]: 7000.0, group[1]: 7190.0, group[2]: 7300.0}

        def fake_score(path: Path):
            return scores[path]

        ranked = rank_videos_for_keep(
            group,
            quality_score_for=fake_score,
            duration_for=lambda path: durations[path],
        )
        self.assertEqual(ranked[0], group[1])


class ConverterTests(unittest.TestCase):
    def test_needs_conversion_for_rename_or_codec(self) -> None:
        self.assertTrue(needs_conversion(Path("Dune (2021).mkv")))

    def test_in_place_conversion_uses_temp_not_suffix(self) -> None:
        src = Path("Dune (2021).mp4")
        target = src
        work = _conversion_work_path(
            src,
            target,
            "Dune (2021).mp4",
            keep_originals=False,
        )
        self.assertTrue(work.name.startswith(".movie_keeper_convert"))
        self.assertEqual(work.suffix, ".mp4")

    def test_in_place_keep_originals_uses_unique_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "Dune (2021).mp4"
            src.touch()
            work = _conversion_work_path(
                src,
                src,
                "Dune (2021).mp4",
                keep_originals=True,
            )
            self.assertEqual(work.name, "Dune (2021)__2.mp4")


if __name__ == "__main__":
    unittest.main()
