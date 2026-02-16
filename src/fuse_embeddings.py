import argparse
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

FUSION_WEIGHTS_CONFIG = {
    "face_present": {
        "face": 0.6,
        "body": 0.25,
        "clip": 0.15,
    },
    "no_face": {
        "face": 0.0,
        "body": 0.65,
        "clip": 0.35,
    },
    "no_person": {
        "face": 0.0,
        "body": 0.0,
        "clip": 1.0,
    },
}


def l2norm(x):
    return x / np.linalg.norm(x)


def main(entities_dir, face_dir, body_dir, clip_dir, out_dir, weights):
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(list(entities_dir.glob("*.json")))

    for ent_file in tqdm(
        entities_dir.glob("*.json"), desc="Fusing embeddings", unit="entity", total=n
    ):
        entity = json.loads(ent_file.read_text())
        eid = entity["entity_id"]

        face_path = face_dir / f"{eid}.npy"
        body_path = body_dir / f"{eid}.npy"
        clip_path = clip_dir / f"{entity['asset_id']}.npy"

        face = np.load(face_path) if face_path.exists() else None
        body = np.load(body_path) if body_path.exists() else None
        clip = np.load(clip_path)

        if face is not None:
            w = weights["face_present"]
        elif body is not None:
            w = weights["no_face"]
        else:
            w = weights["no_person"]

        emb = 0
        if face is not None:
            emb += w["face"] * face
        if body is not None:
            emb += w["body"] * body
        emb += w["clip"] * clip

        emb = l2norm(emb)
        np.save(out_dir / f"{eid}.npy", emb)

        logger.info(f"Fused embedding for entity {eid}")

    print("Fusion complete")
    logger.info("Fusion complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities-dir", type=Path, default=Path("data/entities"))
    ap.add_argument("--face-dir", type=Path, default=Path("data/embeddings/face"))
    ap.add_argument("--body-dir", type=Path, default=Path("data/embeddings/body"))
    ap.add_argument("--clip-dir", type=Path, default=Path("data/embeddings/clip"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/embeddings/fused"))
    ap.add_argument("--log-file", type=Path, default=Path("logs/fuse_embeddings.log"))
    ap.add_argument("--face-weights", nargs=3, type=float, default=[0.6, 0.25, 0.15])
    ap.add_argument("--no-face-weights", nargs=3, type=float, default=[0.0, 0.65, 0.35])
    ap.add_argument("--no-person-weights", nargs=3, type=float, default=[0.0, 0.0, 1.0])

    args = ap.parse_args()

    weights = {
        "face_present": {
            "face": ap.parse_args().face_weights[0],
            "body": ap.parse_args().face_weights[1],
            "clip": ap.parse_args().face_weights[2],
        },
        "no_face": {
            "face": ap.parse_args().no_face_weights[0],
            "body": ap.parse_args().no_face_weights[1],
            "clip": ap.parse_args().no_face_weights[2],
        },
        "no_person": {
            "face": ap.parse_args().no_person_weights[0],
            "body": ap.parse_args().no_person_weights[1],
            "clip": ap.parse_args().no_person_weights[2],
        },
    }

    logging.basicConfig(
        filename=args.log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Starting embedding fusion process")

    main(
        args.entities_dir,
        args.face_dir,
        args.body_dir,
        args.clip_dir,
        args.out_dir,
        weights,
    )

    logger.info("Embedding fusion process complete")
