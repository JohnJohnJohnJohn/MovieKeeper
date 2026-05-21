# MovieKeeper

A Python CLI tool to organize movie collections by removing duplicates and standardizing formats.

## Features

- **Exact duplicate detection** via SHA256 — files with identical content are identified and deduplicated
- **Perceptual duplicate detection** via frame hashing — different encodes of the same movie are detected using perceptual video hashing
- **Resolution-based quality selection** — when duplicates are found, the highest quality version (resolution + bitrate) is kept automatically
- **MP4 conversion with ffmpeg** — all remaining files are converted to MP4 (H.264/AAC, CRF 18) for maximum compatibility

## Prerequisites

- Python 3.9+
- ffmpeg and ffprobe (for video conversion and metadata extraction)

## Installation

```bash
# Clone the repository
git clone https://github.com/JohnJohnJohnJohn/MovieKeeper.git
cd MovieKeeper

# Install Python dependencies
pip install -r requirements.txt

# Install ffmpeg (macOS)
brew install ffmpeg

# Install ffmpeg (Ubuntu/Debian)
sudo apt install ffmpeg
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

MovieKeeper processes your collection in four phases:

1. **Scan** — Recursively scans the target directory for video files (supports 11 formats including mkv, avi, mp4, mov, wmv, and more).

2. **Exact Dedup** — Computes SHA256 hashes for all files. Files with identical hashes are grouped, and all but one copy is marked for removal.

3. **Perceptual Dedup** — Extracts representative frames from each video and computes perceptual hashes. Videos with similar hashes (below the threshold) are grouped as duplicates of the same movie. The highest quality version (based on resolution and bitrate) is kept.

4. **Convert** — Converts all remaining video files to MP4 format using H.264 video (CRF 18) and AAC audio for maximum playback compatibility.

## Safety Features

- **Confirmation prompts** — Before deleting or converting files, you are asked to confirm
- **Trash instead of delete** — Removed files are moved to `.movie_keeper_trash/` rather than permanently deleted
- **Dry-run mode** — Preview all actions without making any changes to your files

## CLI Arguments

| Argument | Description | Default |
|---|---|---|
| `--path` | Path to the directory containing video files | Required |
| `--dry-run` | Preview changes without modifying any files | `False` |
| `--keep-originals` | Keep original files after conversion instead of deleting them | `False` |
| `--threshold` | Perceptual hash distance threshold for duplicate detection | `8` |

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
