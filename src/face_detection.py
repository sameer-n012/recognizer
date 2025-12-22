import argparse
import json
import logging
from pathlib import Path

import cv2
import pandas as pd
from insightface.app import FaceAnalysis
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_FACE_DET_SIZE = (640, 640)
DEFAULT_FACE_DET_CONF_THRESHOLD = 0.6
INSIGHTFACE_ROOT = "./models/insightface"


def crop(img, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return img[y1:y2, x1:x2]


def main(index_path, frames_dir, persons_dir, out_dir, det_size, conf_threshold):
    out_dir.mkdir(parents=True, exist_ok=True)

    app = FaceAnalysis(
        "buffalo_l", root=INSIGHTFACE_ROOT, providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=tuple(det_size))

    df = pd.read_parquet(index_path)

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Processing assets", unit="asset"
    ):
        asset_id = row["asset_id"]
        out_path = out_dir / f"{asset_id}.json"
        if out_path.exists():
            continue

        persons_path = persons_dir / f"{asset_id}.json"
        if not persons_path.exists():
            continue

        with open(persons_path) as f:
            persons = json.load(f)["detections"]

        faces_out = []

        if row["type"] == "image":
            img = cv2.imread(row["path"])
            sources = [(None, img)]
        else:
            frame_root = frames_dir / asset_id
            sources = []
            for frame in sorted(frame_root.glob("*.jpg")):
                sources.append((frame.name, cv2.imread(str(frame))))

        for src_name, img in sources:
            for idx, det in enumerate(persons):
                if det["source"] != src_name:
                    continue

                person_crop = crop(img, det["bbox"])
                if person_crop.size == 0:
                    continue

                faces = app.get(person_crop)
                for face in faces:
                    if face.det_score < conf_threshold:
                        continue

                    faces_out.append(
                        {
                            "source": src_name,
                            "person_det_index": idx,
                            "bbox": face.bbox.tolist(),
                            "confidence": float(face.det_score),
                        }
                    )

        with open(out_path, "w") as f:
            json.dump({"asset_id": asset_id, "faces": faces_out}, f)

        logger.info(f"Processed asset: {asset_id}, faces detected: {len(faces_out)}")

    print("Face detection complete")
    logger.info("Face detection complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("cache/file_index.parquet"))
    ap.add_argument("--frames-dir", type=Path, default=Path("data/frames"))
    ap.add_argument("--persons-dir", type=Path, default=Path("data/detections/persons"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/detections/faces"))
    ap.add_argument("--log-file", type=Path, default=Path("logs/face_detection.log"))
    ap.add_argument(
        "--det-size",
        type=int,
        nargs=2,
        default=[DEFAULT_FACE_DET_SIZE[0], DEFAULT_FACE_DET_SIZE[1]],
    )
    ap.add_argument(
        "--conf-threshold",
        type=float,
        default=DEFAULT_FACE_DET_CONF_THRESHOLD,
    )
    args = ap.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Starting face detection process")

    main(
        args.index,
        args.frames_dir,
        args.persons_dir,
        args.out_dir,
        tuple(args.det_size),
        args.conf_threshold,
    )

    logger.info("Face detection process completed")
