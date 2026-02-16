import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)

BATCH_SIZE = 16

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def l2norm(x):
    return x / np.linalg.norm(x)


def preprocess_tensor(img: np.ndarray, device):
    t = torch.from_numpy(img).to(device)
    t = t.permute(2, 0, 1).float() / 255.0
    t = F.interpolate(
        t.unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    mean = CLIP_MEAN[:, None, None].to(device)
    std = CLIP_STD[:, None, None].to(device)
    t = (t - mean) / std
    return t


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
        batch = []
        for img in imgs:
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch.append(preprocess_tensor(img, device))

            if len(batch) == batch_size:
                batch_t = torch.stack(batch)
                with torch.no_grad():
                    f = model.encode_image(batch_t).cpu().numpy()
                f /= np.linalg.norm(f, axis=1, keepdims=True)
                feats.append(f)
                batch.clear()

        # flush remainder
        if batch:
            batch_t = torch.stack(batch)
            with torch.no_grad():
                f = model.encode_image(batch_t).cpu().numpy()
            f /= np.linalg.norm(f, axis=1, keepdims=True)
            feats.append(f)

        if feats:
            feats = np.vstack(feats)
            emb = feats.mean(axis=0)
            emb /= np.linalg.norm(emb)
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
