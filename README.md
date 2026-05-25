# MovieKeeper

A Python CLI tool to organize movie collections by removing duplicates and standardizing formats.

## Features

- **Exact duplicate detection** via SHA256 — files with identical content are identified and deduplicated
- **Perceptual duplicate detection** via frame hashing — different encodes of the same movie are detected using perceptual video hashing
- **Resolution-based quality selection** — when duplicates are found, the highest quality version (resolution + bitrate) is kept automatically
- **Standardized MP4 filenames + conversion** — files are renamed to a cleaned convention and converted to MP4 (H.264/AAC, CRF 18) for maximum compatibility
- **Split-part merging** — contiguous multi-part releases (`_part_1`, `_cd1`, etc.) can be merged into one file
- **Tag suggestions** — optional bracket-tag suggestions for untagged movies based on tagged library styles

## Prerequisites

- Python 3.9+
- ffmpeg and ffprobe (for video conversion, metadata extraction, and part merging)

## Installation

```bash
git clone https://github.com/JohnJohnJohnJohn/MovieKeeper.git
cd MovieKeeper

pip install -r requirements.txt
```

## Usage

### Dry-run mode (preview changes without making them)

```bash
python3 -m movie_keeper.cli --path /path/to/movies --dry-run
```

### Normal run

```bash
python3 -m movie_keeper.cli --path /path/to/movies
```

### With options

```bash
python3 -m movie_keeper.cli --path /path/to/movies --keep-originals --threshold 12
```

## How It Works

MovieKeeper processes your collection in six phases (split-part combining runs by default after standardization):

1. **Scan** — Recursively scans the target directory for video files (MP4, MKV, AVI, MOV, WMV, and more). Uses a local index cache for faster repeat runs.

2. **Exact Dedup** — Computes SHA256 hashes for all files. Files with identical hashes are grouped, and all but one copy is marked for removal.

3. **Perceptual Dedup** — Extracts representative frames from each video and computes perceptual hashes. Files with similar hashes (below the threshold) are grouped as duplicates of the same movie. You choose which version to keep.

4. **Standardize** — Renames files to a cleaned convention (bracket normalization, duplicate markers removed) and converts to **H.264/AAC MP4** automatically, one file at a time. Use `--crf`, `--preset`, and `--max-file-size-mb` to tune conversion.

5. **Combine split parts** — Detects contiguous split-part files (for example `movie_part_1.mkv` + `movie_part_2.mkv`) and merges them into a single MP4. Non-contiguous gaps are skipped. Prompts per group; Enter accepts the default. Use `--skip-combine-parts` to skip this phase.

6. **Suggest tags** *(optional)* — Learns style profiles from tagged files like `[A24] Dune (2021).mp4` and suggests likely tags for untagged movies based on visual similarity.

## Split-part merging

When a movie is split across multiple files, MovieKeeper can merge **contiguous** runs:

```bash
python3 -m movie_keeper.cli --path /path/to/movies --combine-parts-only --dry-run
python3 -m movie_keeper.cli --path /path/to/movies --skip-combine-parts
```

Supported naming patterns include `_part_N`, `_pt_N`, `_cdN`, `-cdN`, `_discN`, trailing ` N`, and `-N`. Already-merged names such as `_part_1to2` are skipped.

## Tag suggestions

If many files follow a tag pattern such as `[A24] Dune (2021).mp4`, MovieKeeper can compare untagged movies against learned tag profiles.

```bash
python3 -m movie_keeper.cli --path /path/to/movies --tags-only
python3 -m movie_keeper.cli --path /path/to/movies --suggest-tags --apply-tags
```

Requirements and caveats:

- A tag needs at least `--tag-min-samples` tagged works (default: 3) before MovieKeeper will suggest it
- Suggestions are heuristic visual-style matches, not proof — review before applying with `--apply-tags`

## Index cache

MovieKeeper stores a local index at `<library>/.movie_keeper/index.json` so repeat runs on the same parent directory are faster. The index remembers known files, exact content hashes, perceptual frame signatures, and metadata. Use `--rebuild-cache` to ignore the index and refresh everything.

## Safety Features

- **Confirmation prompts** — Phases 2, 3, and 5 ask before removing or merging (Enter accepts the default). Phase 3 lets you pick which version to keep, with resolution and bitrate shown for each copy. Phase 4 automatically renames and converts without prompting.
- **Trash instead of delete** — Removed duplicates, merged split parts, and converted originals are moved to `.movie_keeper_trash/` preserving their library-relative folder structure
- **Dry-run mode** — Preview all actions without making any changes to your files

## CLI Arguments

| Argument | Description | Default |
|---|---|---|
| `--path` | Path to the directory containing video files | Required |
| `--dry-run` | Preview changes without modifying any files | `False` |
| `--keep-originals` | Keep original files after standardization | `False` |
| `--threshold` | Perceptual hash distance threshold for duplicate detection | `10` |
| `--crf` | H.264 CRF for MP4 conversion (1–51) | `18` |
| `--preset` | ffmpeg x264 preset | `medium` |
| `--max-file-size-mb` | Optional output size cap; `0` disables | `0` |
| `--log-file` | Path to write a detailed operation log | None |
| `--rebuild-cache` | Ignore and rebuild the local scan/hash index | `False` |
| `--suggest-tags` | Suggest bracket tags for untagged movies after the normal pipeline | `False` |
| `--tags-only` | Scan and run tag suggestions only | `False` |
| `--standardize-only` | Scan and run filename/MP4 standardization (phase 4) only | `False` |
| `--skip-combine-parts` | Skip merging contiguous split-part files (phase 5) | `False` |
| `--combine-parts-only` | Scan and merge contiguous split-part files only | `False` |
| `--tag-min-samples` | Minimum tagged works required to learn a tag profile | `3` |
| `--tag-threshold` | Visual similarity threshold for tag suggestions | `12` |
| `--apply-tags` | Prompt to rename untagged files when a match is suggested | `False` |

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
