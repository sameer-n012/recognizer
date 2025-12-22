import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import open_clip
import pandas as pd
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

BATCH_SIZE = 16


def l2norm(x):
    return x / np.linalg.norm(x)


def main(index_path, frames_dir, out_dir, batch_size):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(index_path)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval().to(device)

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Processing assets", unit="asset"
    ):
        out_path = out_dir / f"{row['asset_id']}.npy"
        if out_path.exists():
            continue

        imgs = []

        if row["type"] == "image":
            imgs.append(cv2.imread(row["path"]))
        else:
            frame_dir = frames_dir / row["asset_id"]
            for f in sorted(frame_dir.glob("*.jpg")):
                imgs.append(cv2.imread(str(f)))

        feats = []
        for img in imgs:
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                f = model.encode_image(t).cpu().numpy()[0]
                feats.append(l2norm(f))

        if feats:
            emb = l2norm(np.mean(feats, axis=0))
            np.save(out_path, emb)

        logger.info(f"Saved CLIP embedding for asset {row['asset_id']}")

    print("CLIP embeddings complete")
    logger.info("CLIP embeddings complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("cache/file_index.parquet"))
    ap.add_argument("--frames-dir", type=Path, default=Path("data/frames"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/embeddings/clip"))
    ap.add_argument("--log-file", type=Path, default=Path("logs/clip_embeddings.log"))
    ap.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    args = ap.parse_args()
    main(args.index, args.frames_dir, args.out_dir, args.batch_size)
