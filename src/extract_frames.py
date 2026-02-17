import argparse
import json
import logging
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import get_section, load_config, resolve, resolve_path

logger = logging.getLogger(__name__)


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
            "format=duration:stream=duration",
            "-of",
            "json",
            str(video_path),
        ]
    )

    probe_json = json.loads(probe)
    duration = 0
    if "streams" not in probe_json or len(probe_json["streams"]) == 0:
        raise ValueError(f"No video streams found in file: {video_path}")

    if "duration" in probe_json["streams"][0]:
        duration = float(json.loads(probe)["streams"][0]["duration"])
    elif "duration" in probe_json["format"]:
        duration = float(json.loads(probe)["format"]["duration"])
    else:
        raise ValueError(f"No duration found for video stream in file: {video_path}")

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
                f"Extracted {n} frames for video: {row['path']} (ID: {row['asset_id']})"
            )
        except Exception:
            failures += 1
            logger.error(
                f"Failed to extract frames for video: {row['path']} (ID: {row['asset_id']})",
                exc_info=True,
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
    ap.add_argument("--config", type=Path, default=Path("configs/config.json"))
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--frames-dir", type=Path, default=None)
    ap.add_argument("--log-file", type=Path, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--max-frame-rate", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    section = get_section(cfg, "extract_frames")

    index_path = resolve_path(resolve(args.index, section.get("index")))
    frames_dir = resolve_path(resolve(args.frames_dir, section.get("frames_dir")))
    log_file = resolve_path(resolve(args.log_file, section.get("log_file")))
    max_frames = resolve(args.max_frames, section.get("max_frames"))
    max_frame_rate = resolve(args.max_frame_rate, section.get("max_frame_rate"))

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logger.info("Starting frame extraction process")

    main(index_path, frames_dir, max_frames, max_frame_rate)

    logger.info("Frame extraction process completed")
