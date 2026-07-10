import hashlib
import json
import subprocess
from pathlib import Path

import cv2

from file_extensions import IMAGE_EXTS, VIDEO_EXTS


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def asset_id_for_file(path: Path) -> str:
    stat = path.stat()
    h = hashlib.sha1()
    h.update(str(path.resolve()).encode())
    h.update(str(stat.st_size).encode())
    h.update(str(int(stat.st_mtime)).encode())
    return h.hexdigest()


def sha256_for_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def ffprobe_metadata(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration,codec_name:format=duration",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.check_output(cmd)
    data = json.loads(out)
    stream = data["streams"][0]
    num, den = map(int, stream["r_frame_rate"].split("/"))
    fps = num / den if den != 0 else None
    duration = stream.get("duration")
    if duration is None:
        duration = data.get("format", {}).get("duration")
    return {
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": fps,
        "duration": float(duration) if duration is not None else None,
        "codec": stream.get("codec_name"),
    }


def cv2_image_metadata(path: Path) -> dict:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    height, width = img.shape[:2]
    return {
        "width": width,
        "height": height,
        "fps": None,
        "duration": None,
        "codec": None,
    }


def extract_frames_sparse(
    video_path: Path,
    out_dir: Path,
    max_frames: int,
    max_frame_rate: float,
    duration: float,
) -> list[Path]:
    """Uniform, count-bounded sampling across the whole video (cheap, corpus-wide)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("frame_*.jpg"))
    if existing:
        return existing

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
    return sorted(out_dir.glob("frame_*.jpg"))


def extract_frames_dense(
    video_path: Path,
    out_dir: Path,
    interval_sec: float,
) -> list[Path]:
    """Fixed-interval sampling, used on-demand for a single shortlisted video pair."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("frame_*.jpg"))
    if existing:
        return existing

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval_sec}",
        str(out_dir / "frame_%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("frame_*.jpg"))
