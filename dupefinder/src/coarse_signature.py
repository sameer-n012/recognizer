import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from config import get_section, load_config, resolve, resolve_path
from hashing import compute_hashes
from tqdm import tqdm
from utils import extract_frames_sparse

logger = logging.getLogger(__name__)

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def l2norm(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm


def preprocess_tensor(img_bgr: np.ndarray, device: torch.device | str) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).to(device)
    t = t.permute(2, 0, 1).float() / 255.0
    t = F.interpolate(
        t.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
    ).squeeze(0)
    mean = CLIP_MEAN[:, None, None].to(device)
    std = CLIP_STD[:, None, None].to(device)
    return (t - mean) / std


def encode_images(
    model, imgs: list[np.ndarray], device: torch.device | str, batch_size: int
) -> np.ndarray:
    feats = []
    for i in range(0, len(imgs), batch_size):
        batch = torch.stack(
            [preprocess_tensor(img, device) for img in imgs[i : i + batch_size]]
        )
        with torch.no_grad():
            emb = model.encode_image(batch).cpu().numpy()
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        feats.append(emb)
    return np.vstack(feats)


def main(
    index_path: Path,
    exact_groups_path: Path,
    frames_dir: Path,
    hashes_out: Path,
    clip_out_dir: Path,
    max_frames: int,
    max_frame_rate: float,
    hash_size: int,
    model_name: str,
    pretrained: str,
    batch_size: int,
) -> None:
    clip_out_dir.mkdir(parents=True, exist_ok=True)
    hashes_out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(index_path)

    resolved_ids: set[str] = set()
    if exact_groups_path.exists():
        exact_df = pd.read_parquet(exact_groups_path)
        resolved_ids = set(exact_df["asset_id"])

    existing_hashes = None
    if hashes_out.exists():
        existing_hashes = pd.read_parquet(hashes_out)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.eval().to(device)

    logger.info("Running coarse signature extraction on device: %s", device)

    new_hash_rows = []
    failures = 0

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Computing coarse signatures", unit="asset"
    ):
        asset_id = row["asset_id"]
        if asset_id in resolved_ids:
            continue

        clip_path = clip_out_dir / f"{asset_id}.npy"
        if clip_path.exists():
            continue

        try:
            if row["type"] == "image":
                img = cv2.imread(row["path"])
                if img is None:
                    failures += 1
                    continue
                hashes = compute_hashes(img, hash_size)
                new_hash_rows.append({"asset_id": asset_id, "frame_index": 0, **hashes})
                emb = encode_images(model, [img], device, batch_size)[0]
            else:
                if not row["duration"] or row["duration"] <= 0:
                    failures += 1
                    continue
                frame_dir = frames_dir / asset_id
                frame_paths = extract_frames_sparse(
                    Path(row["path"]),
                    frame_dir,
                    max_frames,
                    max_frame_rate,
                    row["duration"],
                )
                imgs = []
                for idx, fp in enumerate(frame_paths):
                    img = cv2.imread(str(fp))
                    if img is None:
                        continue
                    hashes = compute_hashes(img, hash_size)
                    new_hash_rows.append(
                        {"asset_id": asset_id, "frame_index": idx, **hashes}
                    )
                    imgs.append(img)
                if not imgs:
                    failures += 1
                    continue
                frame_embs = encode_images(model, imgs, device, batch_size)
                emb = l2norm(frame_embs.mean(axis=0))

            np.save(clip_path, emb)
            # logger.info("Saved coarse signature for %s", asset_id)
        except Exception:
            failures += 1
            # logger.error("Failed coarse signature for %s", asset_id, exc_info=True)

    if new_hash_rows:
        new_df = pd.DataFrame(new_hash_rows)
        combined = (
            pd.concat([existing_hashes, new_df], ignore_index=True)
            if existing_hashes is not None
            else new_df
        )
        combined.to_parquet(hashes_out, index=False)

    print(f"Coarse signature extraction complete ({failures} failures)")
    logger.info("Coarse signature extraction complete (%d failures)", failures)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.json"))
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--exact-groups", type=Path, default=None)
    ap.add_argument("--frames-dir", type=Path, default=None)
    ap.add_argument("--hashes-out", type=Path, default=None)
    ap.add_argument("--clip-out-dir", type=Path, default=None)
    ap.add_argument("--log-file", type=Path, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--max-frame-rate", type=float, default=None)
    ap.add_argument("--hash-size", type=int, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--pretrained", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    section = get_section(cfg, "coarse_signature")

    index_path = resolve_path(resolve(args.index, section.get("index")))
    exact_groups_path = resolve_path(
        resolve(args.exact_groups, section.get("exact_groups"))
    )
    frames_dir = resolve_path(resolve(args.frames_dir, section.get("frames_dir")))
    hashes_out = resolve_path(resolve(args.hashes_out, section.get("hashes_out")))
    clip_out_dir = resolve_path(resolve(args.clip_out_dir, section.get("clip_out_dir")))
    log_file = resolve_path(resolve(args.log_file, section.get("log_file")))
    max_frames = resolve(args.max_frames, section.get("max_frames"))
    max_frame_rate = resolve(args.max_frame_rate, section.get("max_frame_rate"))
    hash_size = resolve(args.hash_size, section.get("hash_size"))
    model_name = resolve(args.model, section.get("model"))
    pretrained = resolve(args.pretrained, section.get("pretrained"))
    batch_size = resolve(args.batch_size, section.get("batch_size"))

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Starting coarse signature extraction")
    main(
        index_path,
        exact_groups_path,
        frames_dir,
        hashes_out,
        clip_out_dir,
        max_frames,
        max_frame_rate,
        hash_size,
        model_name,
        pretrained,
        batch_size,
    )
    logger.info("Coarse signature extraction complete")
