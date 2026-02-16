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


def l2norm(x):
    return x / np.linalg.norm(x)


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

        asset_path = file_index_df.loc[entity["asset_id"], "path"]

        for frame in entity["frames"]:
            if not frame["faces"]:
                continue

            if entity["type"] == "image":
                img = cv2.imread(asset_path)
            else:
                img = cv2.imread(str(frames_dir / entity["asset_id"] / frame["source"]))

            for f in frame["faces"]:
                x1, y1, x2, y2 = map(int, f["bbox"])
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                faces = app.get(crop)
                for face in faces:
                    if face.det_score >= min_confidence:
                        embeds.append(l2norm(face.embedding))

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
