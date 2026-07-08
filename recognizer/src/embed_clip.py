import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import get_section, load_config, resolve, resolve_path

logger = logging.getLogger(__name__)

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def preprocess_tensor(img: np.ndarray, device: torch.device | str) -> torch.Tensor:
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


def clamp_bbox(
    bbox: Sequence[float], width: int, height: int
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def main(
    entities_dir: Path,
    frames_dir: Path,
    index_path: Path,
    out_dir: Path,
    batch_size: int,
    model_name: str,
    pretrained: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.eval().to(device)

    index_df = pd.read_parquet(index_path).set_index("asset_id")

    entity_files = sorted(entities_dir.glob("*.json"))
    for entity_path in tqdm(
        entity_files, desc="Embedding entities", unit="entity", total=len(entity_files)
    ):
        entity = json.loads(entity_path.read_text())
        out_path = out_dir / f"{entity['entity_id']}.npy"
        if out_path.exists():
            continue

        if entity["asset_id"] not in index_df.index:
            logger.warning("Missing index entry for asset %s", entity["asset_id"])
            continue

        asset_path = Path(index_df.loc[entity["asset_id"], "path"])
        crops: list[torch.Tensor] = []

        for frame in entity["frames"]:
            if entity["type"] == "image":
                img_path = asset_path
            else:
                frame_source = frame["source"]
                img_path = frames_dir / entity["asset_id"] / frame_source

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            bbox = clamp_bbox(frame["person_bbox"], img.shape[1], img.shape[0])
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(preprocess_tensor(crop, device))

        if not crops:
            logger.warning(
                "Entity %s has no valid crops, skipping", entity["entity_id"]
            )
            continue

        feats: list[np.ndarray] = []
        batch: list[torch.Tensor] = []
        for tensor in crops:
            batch.append(tensor)
            if len(batch) == batch_size:
                batch_tensor = torch.stack(batch)
                with torch.no_grad():
                    emb = model.encode_image(batch_tensor).cpu().numpy()
                emb /= np.linalg.norm(emb, axis=1, keepdims=True)
                feats.append(emb)
                batch.clear()

        if batch:
            batch_tensor = torch.stack(batch)
            with torch.no_grad():
                emb = model.encode_image(batch_tensor).cpu().numpy()
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
            feats.append(emb)

        if not feats:
            logger.warning("Entity %s produced no embeddings", entity["entity_id"])
            continue

        features = np.vstack(feats)
        emb = l2norm(features.mean(axis=0))
        np.save(out_path, emb)
        logger.info("Saved CLIP embedding for entity %s", entity["entity_id"])

    print("CLIP embeddings complete")
    logger.info("CLIP embeddings complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.json"))
    ap.add_argument("--entities-dir", type=Path, default=None)
    ap.add_argument("--frames-dir", type=Path, default=None)
    ap.add_argument("--index-path", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--log-file", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--pretrained", type=str, default=None)

    args = ap.parse_args()

    cfg = load_config(args.config)
    section = get_section(cfg, "embed_clip")

    entities_dir = resolve_path(resolve(args.entities_dir, section.get("entities_dir")))
    frames_dir = resolve_path(resolve(args.frames_dir, section.get("frames_dir")))
    index_path = resolve_path(resolve(args.index_path, section.get("index_path")))
    out_dir = resolve_path(resolve(args.out_dir, section.get("out_dir")))
    log_file = resolve_path(resolve(args.log_file, section.get("log_file")))
    batch_size = resolve(args.batch_size, section.get("batch_size"))
    model_name = resolve(args.model, section.get("model"))
    pretrained = resolve(args.pretrained, section.get("pretrained"))

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Starting CLIP embedding extraction")
    main(
        entities_dir,
        frames_dir,
        index_path,
        out_dir,
        batch_size,
        model_name,
        pretrained,
    )
    logger.info("CLIP embedding extraction complete")
