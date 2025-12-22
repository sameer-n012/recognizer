import argparse
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_MIN_FRAMES_PER_ENTITY = 3


def make_entity_id(asset_id, key):
    h = hashlib.sha1()
    h.update(asset_id.encode())
    h.update(str(key).encode())
    return h.hexdigest()


def main(index_path, persons_dir, faces_dir, out_dir, min_frames_per_entity):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(index_path)

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Constructing entities", unit="asset"
    ):
        asset_id = row["asset_id"]

        persons_path = persons_dir / f"{asset_id}.json"
        if not persons_path.exists():
            continue

        with open(persons_path) as f:
            persons = json.load(f)["detections"]

        faces = []
        faces_path = faces_dir / f"{asset_id}.json"
        if faces_path.exists():
            with open(faces_path) as f:
                faces = json.load(f)["faces"]

        face_map = defaultdict(list)
        for fdet in faces:
            face_map[fdet["person_det_index"]].append(fdet)

        if row["type"] == "image":
            for i, det in enumerate(persons):
                eid = make_entity_id(asset_id, i)
                payload = {
                    "entity_id": eid,
                    "asset_id": asset_id,
                    "type": "image",
                    "frames": [
                        {
                            "source": det["source"],
                            "person_bbox": det["bbox"],
                            "faces": face_map.get(i, []),
                        }
                    ],
                    "has_face": i in face_map,
                }
                with open(out_dir / f"{eid}.json", "w") as f:
                    json.dump(payload, f)
            continue

        # VIDEO
        # Simple temporal grouping: each person detection index is an entity
        entity_frames = defaultdict(list)

        for idx, det in enumerate(persons):
            entity_frames[idx].append(
                {
                    "source": det["source"],
                    "person_bbox": det["bbox"],
                    "faces": face_map.get(idx, []),
                }
            )

        for idx, frames in entity_frames.items():
            if len(frames) < min_frames_per_entity:
                continue

            eid = make_entity_id(asset_id, idx)
            payload = {
                "entity_id": eid,
                "asset_id": asset_id,
                "type": "video",
                "frames": frames,
                "has_face": any(len(f["faces"]) > 0 for f in frames),
            }
            with open(out_dir / f"{eid}.json", "w") as f:
                json.dump(payload, f)

    print("Entity construction complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("cache/file_index.parquet"))
    ap.add_argument("--persons-dir", type=Path, default=Path("data/detections/persons"))
    ap.add_argument("--faces-dir", type=Path, default=Path("data/detections/faces"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/entities"))
    ap.add_argument("--log-file", type=Path, default=Path("logs/person_detection.log"))
    ap.add_argument(
        "--min-frames-per-entity",
        type=int,
        default=DEFAULT_MIN_FRAMES_PER_ENTITY,
    )
    args = ap.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logger.info("Starting entity construction process")

    main(
        args.index,
        args.persons_dir,
        args.faces_dir,
        args.out_dir,
        args.min_frames_per_entity,
    )

    logger.info("Entity construction process completed")
