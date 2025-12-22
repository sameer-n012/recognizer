import argparse
import json
import logging
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_MAX_FRAMES = 150
DEFAULT_MAX_FRAME_RATE = 2.0  # frames per second


def extract_frames(
    video_path: Path, out_dir: Path, max_frames: int, max_frame_rate: float
):
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "json",
            str(video_path),
        ]
    )

    probe_json = json.loads(probe)
    if "streams" not in probe_json or len(probe_json["streams"]) == 0:
        raise ValueError(f"No video streams found in file: {video_path}")
    elif "duration" not in probe_json["streams"][0]:
        raise ValueError(f"No duration found for video stream in file: {video_path}")

    duration = float(json.loads(probe)["streams"][0]["duration"])

    if duration <= 0:
        raise ValueError(f"Invalid video duration ({duration}) for file: {video_path}")

    interval = max(duration / max_frames, 1 / max_frame_rate)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval}",
        str(out_dir / "frame_%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)

    return len(list(out_dir.glob("frame_*.jpg")))


def main(index_path: Path, frames_root: Path, max_frames: int, max_frame_rate: float):
    df = pd.read_parquet(index_path)

    videos = df[df["type"] == "video"]

    failures = 0

    for _, row in tqdm(
        videos.iterrows(), total=len(videos), desc="Extracting frames", unit="video"
    ):
        out_dir = frames_root / row["asset_id"]
        row["num_frames"] = 0
        if out_dir.exists():
            n = len(list(out_dir.iterdir()))
            if n > 0:
                row["num_frames"] = n
                continue
        try:
            n = extract_frames(Path(row["path"]), out_dir, max_frames, max_frame_rate)
            row["num_frames"] = n
            logger.info(
                f"Extracted frames for video: {row['path']} (ID: {row['asset_id']})"
            )
        except Exception:
            failures += 1
            logger.error(
                f"Failed to extract frames for video: {row['path']} (ID: {row['asset_id']})",
                exc_info=False,
            )

    df.to_parquet(index_path, index=False)

    print(
        f"Frame extraction complete for {len(videos) - failures} videos ({failures} failures)"
    )
    logger.info(
        f"Frame extraction complete for {len(videos) - failures} videos ({failures} failures)"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("cache/file_index.parquet"))
    ap.add_argument("--frames-dir", type=Path, default=Path("data/frames"))
    ap.add_argument("--log-file", type=Path, default=Path("logs/extract_frames.log"))
    ap.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    ap.add_argument("--max-frame-rate", type=float, default=DEFAULT_MAX_FRAME_RATE)
    args = ap.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logger.info("Starting frame extraction process")

    main(args.index, args.frames_dir, args.max_frames, args.max_frame_rate)

    logger.info("Frame extraction process completed")
