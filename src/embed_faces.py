import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_MIN_CONFIDENCE = 0.6
DEFAULT_FACE_PAD_RATIO = 0.05
DEFAULT_MATCH_IOU = 0.2


def l2norm(x):
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm


def clamp_bbox(bbox, width, height, pad_ratio=0.0):
    x1, y1, x2, y2 = bbox
    box_w = max(0.0, float(x2) - float(x1))
    box_h = max(0.0, float(y2) - float(y1))
    pad_x = box_w * pad_ratio
    pad_y = box_h * pad_ratio
    x1 = max(0.0, min(float(x1) - pad_x, float(width)))
    y1 = max(0.0, min(float(y1) - pad_y, float(height)))
    x2 = max(0.0, min(float(x2) + pad_x, float(width)))
    y2 = max(0.0, min(float(y2) + pad_y, float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [int(x1), int(y1), int(x2), int(y2)]


def bbox_iou(box_a, box_b):
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_width = max(0.0, xB - xA)
    inter_height = max(0.0, yB - yA)
    inter_area = inter_width * inter_height
    if inter_area == 0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = max(area_a + area_b - inter_area, 1e-6)
    return inter_area / union_area


def match_face_to_detection(face_bbox, detected_faces, min_iou):
    best_face = None
    best_iou = min_iou
    for face in detected_faces:
        face_bbox_candidate = face.bbox
        if hasattr(face_bbox_candidate, "tolist"):
            face_bbox_candidate = face_bbox_candidate.tolist()
        current_iou = bbox_iou(face_bbox, face_bbox_candidate)
        if current_iou > best_iou:
            best_iou = current_iou
            best_face = face
    return best_face


def main(entities_dir, frames_dir, out_dir, index_path, min_confidence):
    out_dir.mkdir(parents=True, exist_ok=True)

    app = FaceAnalysis("buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0)

    n = len(list(entities_dir.glob("*.json")))

    file_index_df = pd.read_parquet(index_path).set_index("asset_id")

    for ent_file in tqdm(
        entities_dir.glob("*.json"), desc="Embedding faces", unit="entity", total=n
    ):
        entity = json.loads(ent_file.read_text())
        if not entity["has_face"]:
            continue

        out_path = out_dir / f"{entity['entity_id']}.npy"
        if out_path.exists():
            continue

        embeds = []

        try:
            asset_path = file_index_df.loc[entity["asset_id"], "path"]
        except KeyError:
            logger.warning("Missing asset_id in index: %s", entity["asset_id"])
            continue

        for frame in entity["frames"]:
            face_records = frame["faces"]
            if not face_records:
                continue

            if entity["type"] == "image":
                img = cv2.imread(asset_path)
            else:
                img = cv2.imread(str(frames_dir / entity["asset_id"] / frame["source"]))
            if img is None:
                continue
            height, width = img.shape[:2]
            detected_faces = None

            if any("embedding" not in f for f in face_records):
                detected_faces = app.get(img)

            for f in face_records:
                if f.get("confidence", 0.0) < min_confidence:
                    continue
                if "embedding" in f:
                    embeds.append(l2norm(np.asarray(f["embedding"], dtype=np.float32)))
                    continue
                bbox = clamp_bbox(
                    f["bbox"], width, height, pad_ratio=DEFAULT_FACE_PAD_RATIO
                )
                if bbox is None:
                    continue
                if not detected_faces:
                    continue
                matched_face = match_face_to_detection(
                    bbox, detected_faces, DEFAULT_MATCH_IOU
                )
                if matched_face and matched_face.det_score >= min_confidence:
                    embeds.append(l2norm(matched_face.embedding))

        if embeds:
            emb = l2norm(np.mean(embeds, axis=0))
            np.save(out_path, emb)
            logger.info(f"Saved face embedding for entity {entity['entity_id']}")

    print("Face embeddings complete")
    logger.info("Face embeddings complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities-dir", type=Path, default=Path("data/entities"))
    ap.add_argument("--frames-dir", type=Path, default=Path("data/frames"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/embeddings/face"))
    ap.add_argument("--index-path", type=Path, default=Path("cache/file_index.parquet"))
    ap.add_argument("--log-file", type=Path, default=Path("logs/face_embeddings.log"))
    ap.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)

    args = ap.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Starting face embedding extraction")

    main(
        args.entities_dir,
        args.frames_dir,
        args.out_dir,
        args.index_path,
        args.min_confidence,
    )

    logger.info("Face embedding extraction complete")
