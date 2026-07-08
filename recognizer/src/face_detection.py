import argparse
import json
import logging
from pathlib import Path

import cv2
import pandas as pd
from insightface.app import FaceAnalysis
from tqdm import tqdm

from config import get_section, load_config, resolve, resolve_path

logger = logging.getLogger(__name__)


def clamp_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(x1), float(width)))
    y1 = max(0.0, min(float(y1), float(height)))
    x2 = max(0.0, min(float(x2), float(width)))
    y2 = max(0.0, min(float(y2), float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def offset_bbox(bbox, offset_xy):
    ox, oy = offset_xy
    x1, y1, x2, y2 = bbox
    return [x1 + ox, y1 + oy, x2 + ox, y2 + oy]


def crop(img, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return img[y1:y2, x1:x2]


def main(
    index_path,
    frames_dir,
    persons_dir,
    out_dir,
    det_size,
    conf_threshold,
    save_embeddings,
    no_cache,
    insightface_root,
    model_name,
    providers,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    app = FaceAnalysis(model_name, root=insightface_root, providers=providers)
    app.prepare(ctx_id=0, det_size=tuple(det_size))

    df = pd.read_parquet(index_path)

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Processing assets", unit="asset"
    ):
        asset_id = row["asset_id"]
        out_path = out_dir / f"{asset_id}.json"
        if out_path.exists() and not no_cache:
            continue

        persons_path = persons_dir / f"{asset_id}.json"
        if not persons_path.exists():
            continue

        with open(persons_path) as f:
            persons = json.load(f)["detections"]

        faces_out = []

        if row["type"] == "image":
            img = cv2.imread(row["path"])
            image_name = Path(row["path"]).name
            sources = [(image_name, img)]
        else:
            frame_root = frames_dir / asset_id
            sources = []
            for frame in sorted(frame_root.glob("*.jpg")):
                sources.append((frame.name, cv2.imread(str(frame))))

        for src_name, img in sources:
            if img is None:
                continue
            height, width = img.shape[:2]
            for idx, det in enumerate(persons):
                if det["source"] != src_name:
                    continue

                person_bbox = clamp_bbox(det["bbox"], width, height)
                if person_bbox is None:
                    continue
                person_crop = crop(img, person_bbox)
                if person_crop.size == 0:
                    continue

                faces = app.get(person_crop)
                for face in faces:
                    if face.det_score < conf_threshold:
                        continue

                    abs_bbox = offset_bbox(
                        face.bbox.tolist(), (person_bbox[0], person_bbox[1])
                    )
                    abs_bbox = clamp_bbox(abs_bbox, width, height)
                    if abs_bbox is None:
                        continue

                    payload = {
                        "source": src_name,
                        "person_det_index": idx,
                        "bbox": abs_bbox,
                        "confidence": float(face.det_score),
                    }
                    if save_embeddings and getattr(face, "embedding", None) is not None:
                        payload["embedding"] = face.embedding.tolist()
                    faces_out.append(payload)

        with open(out_path, "w") as f:
            json.dump({"asset_id": asset_id, "faces": faces_out}, f)

        logger.info(f"Processed asset: {asset_id}, faces detected: {len(faces_out)}")

    print("Face detection complete")
    logger.info("Face detection complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.json"))
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--frames-dir", type=Path, default=None)
    ap.add_argument("--persons-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--log-file", type=Path, default=None)
    ap.add_argument("--no-cache", action="store_true", default=None)
    ap.add_argument(
        "--no-embeddings",
        action="store_true",
        default=None,
        help="Do not store face embeddings in detections output",
    )
    ap.add_argument("--det-size", type=int, nargs=2, default=None)
    ap.add_argument("--conf-threshold", type=float, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--providers", type=str, nargs="+", default=None)
    ap.add_argument("--insightface-root", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    section = get_section(cfg, "face_detection")

    index_path = resolve_path(resolve(args.index, section.get("index")))
    frames_dir = resolve_path(resolve(args.frames_dir, section.get("frames_dir")))
    persons_dir = resolve_path(resolve(args.persons_dir, section.get("persons_dir")))
    out_dir = resolve_path(resolve(args.out_dir, section.get("out_dir")))
    log_file = resolve_path(resolve(args.log_file, section.get("log_file")))
    det_size = resolve(args.det_size, section.get("det_size"))
    conf_threshold = resolve(args.conf_threshold, section.get("conf_threshold"))
    no_cache = resolve(args.no_cache, section.get("no_cache"), False)
    no_embeddings = resolve(
        args.no_embeddings, not section.get("save_embeddings"), False
    )
    model_name = resolve(args.model, section.get("model"))
    providers = resolve(args.providers, section.get("providers"))
    insightface_root = resolve_path(
        resolve(args.insightface_root, section.get("insightface_root"))
    )

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Starting face detection process")

    main(
        index_path,
        frames_dir,
        persons_dir,
        out_dir,
        tuple(det_size),
        conf_threshold,
        not no_embeddings,
        bool(no_cache),
        str(insightface_root),
        model_name,
        providers,
    )

    logger.info("Face detection process completed")
