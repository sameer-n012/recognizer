import argparse
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import get_section, load_config, resolve, resolve_path

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
    ap.add_argument("--config", type=Path, default=Path("configs/config.json"))
    ap.add_argument("--entities-dir", type=Path, default=None)
    ap.add_argument("--face-dir", type=Path, default=None)
    ap.add_argument("--body-dir", type=Path, default=None)
    ap.add_argument("--clip-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--log-file", type=Path, default=None)
    ap.add_argument("--face-weights", nargs=3, type=float, default=None)
    ap.add_argument("--no-face-weights", nargs=3, type=float, default=None)
    ap.add_argument("--no-person-weights", nargs=3, type=float, default=None)

    args = ap.parse_args()

    cfg = load_config(args.config)
    section = get_section(cfg, "fuse_embeddings")

    entities_dir = resolve_path(resolve(args.entities_dir, section.get("entities_dir")))
    face_dir = resolve_path(resolve(args.face_dir, section.get("face_dir")))
    body_dir = resolve_path(resolve(args.body_dir, section.get("body_dir")))
    clip_dir = resolve_path(resolve(args.clip_dir, section.get("clip_dir")))
    out_dir = resolve_path(resolve(args.out_dir, section.get("out_dir")))
    log_file = resolve_path(resolve(args.log_file, section.get("log_file")))
    face_weights = resolve(args.face_weights, section.get("face_weights"))
    no_face_weights = resolve(args.no_face_weights, section.get("no_face_weights"))
    no_person_weights = resolve(
        args.no_person_weights, section.get("no_person_weights")
    )

    weights = {
        "face_present": {
            "face": face_weights[0],
            "body": face_weights[1],
            "clip": face_weights[2],
        },
        "no_face": {
            "face": no_face_weights[0],
            "body": no_face_weights[1],
            "clip": no_face_weights[2],
        },
        "no_person": {
            "face": no_person_weights[0],
            "body": no_person_weights[1],
            "clip": no_person_weights[2],
        },
    }

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Starting embedding fusion process")

    main(
        entities_dir,
        face_dir,
        body_dir,
        clip_dir,
        out_dir,
        weights,
    )

    logger.info("Embedding fusion process complete")
