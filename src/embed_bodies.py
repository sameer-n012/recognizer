import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torchreid
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_BODY_INPUT_SIZE = (256, 128)
BATCH_SIZE = 16


def l2norm(x):
    return x / np.linalg.norm(x)


def main(entities_dir, frames_dir, out_dir, input_size, batch_size):
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = torchreid.models.build_model(
        name="osnet_x1_0", num_classes=1, pretrained=True
    )
    model.eval().to(device)

    h, w = input_size

    for ent_file in tqdm(
        entities_dir.glob("*.json"), desc="Processing entities", unit="entity"
    ):
        entity = json.loads(ent_file.read_text())
        out_path = out_dir / f"{entity['entity_id']}.npy"
        if out_path.exists():
            continue

        embeds = []

        for frame in entity["frames"]:
            if entity["type"] == "image":
                img = cv2.imread(frame["source"])
            else:
                img = cv2.imread(str(frames_dir / entity["asset_id"] / frame["source"]))

            x1, y1, x2, y2 = map(int, frame["person_bbox"])
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop = cv2.resize(crop, (w, h))
            crop = crop[:, :, ::-1].transpose(2, 0, 1)
            crop = torch.tensor(crop, dtype=torch.float32).unsqueeze(0) / 255.0

            with torch.no_grad():
                feat = model(crop.to(device)).cpu().numpy()[0]
                embeds.append(l2norm(feat))

        if embeds:
            emb = l2norm(np.mean(embeds, axis=0))
            np.save(out_path, emb)

        logger.info(f"Saved body embedding for entity {entity['entity_id']}")

    print("Body embeddings complete")
    logger.info("Body embeddings complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities-dir", type=Path, default=Path("data/entities"))
    ap.add_argument("--frames-dir", type=Path, default=Path("data/frames"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/embeddings/body"))
    ap.add_argument("--log-file", type=Path, default=Path("logs/body_embeddings.log"))
    ap.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=DEFAULT_BODY_INPUT_SIZE,
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
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
        args.entities_dir,
        args.frames_dir,
        args.out_dir,
        args.input_size,
        args.batch_size,
    )

    logger.info("Body embeddings process complete")
