import argparse
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

FUSION_WEIGHTS_CONFIG = {
    "face_present": {
        "face": 0.9,
        "body": 0.05,
        "clip": 0.05,
    },
    "no_face": {
        "face": 0.0,
        "body": 0.6,
        "clip": 0.4,
    },
    "no_person": {
        "face": 0.0,
        "body": 0.0,
        "clip": 1.0,
    },
}


def l2norm(x):
    return x / np.linalg.norm(x)


def main(
    entities_dir: Path,
    face_dir: Path,
    body_dir: Path,
    clip_dir: Path,
    out_dir: Path,
    weights: dict[str, dict[str, float]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    entity_files = sorted(entities_dir.glob("*.json"))
    total_entities = len(entity_files)

    for ent_file in tqdm(
        entity_files, desc="Fusing embeddings", unit="entity", total=total_entities
    ):
        entity = json.loads(ent_file.read_text())
        eid = entity["entity_id"]

        face_path = face_dir / f"{eid}.npy"
        body_path = body_dir / f"{eid}.npy"
        clip_path = clip_dir / f"{eid}.npy"

        face = np.load(face_path) if face_path.exists() else None
        body = np.load(body_path) if body_path.exists() else None
        if not clip_path.exists():
            logger.warning("Skipping %s because clip embedding is missing", eid)
            continue
        clip = np.load(clip_path)

        if face is not None:
            weight_config = weights["face_present"]
        elif body is not None:
            weight_config = weights["no_face"]
        else:
            weight_config = weights["no_person"]

        emb = np.zeros_like(clip)
        if face is not None:
            emb += weight_config["face"] * face
        if body is not None:
            emb += weight_config["body"] * body
        emb += weight_config["clip"] * clip

        emb = l2norm(emb)
        np.save(out_dir / f"{eid}.npy", emb)

        logger.info("Fused embedding for entity %s", eid)

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
            "face": args.face_weights[0],
            "body": args.face_weights[1],
            "clip": args.face_weights[2],
        },
        "no_face": {
            "face": args.no_face_weights[0],
            "body": args.no_face_weights[1],
            "clip": args.no_face_weights[2],
        },
        "no_person": {
            "face": args.no_person_weights[0],
            "body": args.no_person_weights[1],
            "clip": args.no_person_weights[2],
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
