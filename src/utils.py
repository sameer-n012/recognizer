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


def ffprobe_metadata(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.check_output(cmd)
    data = json.loads(out)
    stream = data["streams"][0]
    num, den = map(int, stream["r_frame_rate"].split("/"))
    fps = num / den if den != 0 else None
    return {
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": fps,
        "duration": float(stream.get("duration", 0.0)),
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
    }
