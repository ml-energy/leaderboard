#!/usr/bin/env python3
"""Migrate existing text-to-video results.json files to include num_frames and fps.

This script reads the MODEL_CONFIGS from workloads.py to get the correct num_frames
and fps values for each video model, then updates all existing results.json files
in the text-to-video results directory.

Usage:
    uv run scripts/migrate_video_num_frames.py --results-dir /path/to/results
"""

import argparse
import json
import sys
from pathlib import Path


# Model configurations - copied from benchmark/mlenergy/diffusion/workloads.py
# Only video models (those with num_frames != None)
VIDEO_MODEL_CONFIGS = {
    "zai-org/CogVideoX1.5-5B": {
        "num_frames": 81,
        "fps": 8,
    },
    "zai-org/CogVideoX-2b": {
        "num_frames": 49,
        "fps": 8,
    },
    "BestWishYsh/ConsisID-preview": {
        "num_frames": 49,
        "fps": 8,
    },
    "maxin-cn/Latte-1": {
        "num_frames": 16,
        "fps": 8,
    },
    "tencent/HunyuanVideo": {
        "num_frames": 129,
        "fps": 15,
    },
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": {
        "num_frames": 81,
        "fps": 15,
    },
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": {
        "num_frames": 81,
        "fps": 15,
    },
}


class MigrationError(Exception):
    """Raised when migration encounters an unexpected condition."""
    pass


def migrate_results_file(results_path: Path, dry_run: bool = False) -> bool:
    """Update a single results.json file with num_frames and fps.

    Returns True if the file was updated, False if already migrated.
    Raises MigrationError on unexpected conditions.
    """
    try:
        with open(results_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise MigrationError(f"Malformed JSON in {results_path}: {e}")
    except OSError as e:
        raise MigrationError(f"Cannot read {results_path}: {e}")

    # Check if already has num_frames
    if "num_frames" in data and "fps" in data:
        print(f"  Skipping {results_path} - already has num_frames and fps")
        return False

    # Get model_id from the results
    model_id = data.get("model_id")
    if not model_id:
        raise MigrationError(f"No model_id found in {results_path}")

    # Look up num_frames and fps
    if model_id not in VIDEO_MODEL_CONFIGS:
        raise MigrationError(
            f"Unknown model_id '{model_id}' in {results_path}. "
            f"Add it to VIDEO_MODEL_CONFIGS in this script."
        )

    config = VIDEO_MODEL_CONFIGS[model_id]
    num_frames = config["num_frames"]
    fps = config["fps"]

    print(f"  Updating {results_path}")
    print(f"    model_id: {model_id}")
    print(f"    num_frames: {num_frames}, fps: {fps}")

    if dry_run:
        print(f"    (dry run - not writing)")
        return True

    # Update the data
    data["num_frames"] = num_frames
    data["fps"] = fps

    # Write back
    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate text-to-video results.json files to include num_frames and fps"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Path to results directory containing diffusion/text-to-video results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making changes",
    )
    args = parser.parse_args()

    results_root = Path(args.results_dir)

    # Find all text-to-video results.json files
    video_results_dir = results_root / "diffusion" / "text-to-video"

    if not video_results_dir.exists():
        print(f"ERROR: No text-to-video directory found at {video_results_dir}")
        sys.exit(1)

    print(f"Scanning for results.json files in {video_results_dir}...")

    results_files = list(video_results_dir.rglob("results.json"))
    print(f"Found {len(results_files)} results.json files")

    if not results_files:
        print("ERROR: No results.json files found")
        sys.exit(1)

    updated = 0
    errors = []
    for results_path in results_files:
        try:
            if migrate_results_file(results_path, dry_run=args.dry_run):
                updated += 1
        except MigrationError as e:
            errors.append(str(e))
            print(f"  ERROR: {e}")

    print(f"\nUpdated {updated} files" + (" (dry run)" if args.dry_run else ""))

    if errors:
        print(f"\nFailed with {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
