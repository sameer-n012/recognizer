import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import open_clip
import pandas as pd
import torch
from tqdm import tqdm

from coarse_signature import encode_images
from config import get_section, load_config, resolve, resolve_path
from utils import extract_frames_dense

logger = logging.getLogger(__name__)


def get_dense_embedding(
    asset_id: str,
    path: Path,
    frames_dense_dir: Path,
    clip_dense_dir: Path,
    model,
    device: torch.device | str,
    dense_interval_sec: float,
) -> np.ndarray | None:
    """Dense frame extraction + CLIP encoding, cached per asset_id rather than per
    pair. With top_k > 1 in build_candidates, the same video can be escalated in
    several pairs (one per similar neighbor) — caching per asset means it's only
    ffmpeg-extracted and CLIP-encoded once total, then reused for every pair it
    appears in, instead of once per pair."""
    emb_path = clip_dense_dir / f"{asset_id}.npy"
    if emb_path.exists():
        return np.load(emb_path)

    frames = extract_frames_dense(path, frames_dense_dir / asset_id, dense_interval_sec)
    if not frames:
        return None

    imgs = [cv2.imread(str(p)) for p in frames]
    imgs = [img for img in imgs if img is not None]
    if not imgs:
        return None

    emb = encode_images(model, imgs, device, batch_size=16)
    clip_dense_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb)
    return emb


def best_diagonal_overlap(sim_matrix: np.ndarray, align_threshold: float) -> dict:
    """Search every constant-offset diagonal of the frame-similarity matrix for the
    longest run of above-threshold matches. Tolerates a time offset between the two
    videos (diagonal position) and, within a diagonal, doesn't require every frame to
    match (playback-rate / sampling jitter), but scores by count of matching frames.
    """
    n, m = sim_matrix.shape
    best = {"offset": None, "match_count": 0, "diagonal_length": 0}
    for offset in range(-(n - 1), m):
        i_start = max(0, -offset)
        i_end = min(n, m - offset)
        if i_end <= i_start:
            continue
        diag = sim_matrix[np.arange(i_start, i_end), np.arange(i_start, i_end) + offset]
        match_count = int(np.sum(diag >= align_threshold))
        if match_count > best["match_count"]:
            best = {
                "offset": offset,
                "match_count": match_count,
                "diagonal_length": i_end - i_start,
            }
    return best


def main(
    index_path: Path,
    scored_pairs_path: Path,
    frames_dense_dir: Path,
    clip_dense_dir: Path,
    out_path: Path,
    dense_interval_sec: float,
    model_name: str,
    pretrained: str,
    align_similarity_threshold: float,
    min_overlap_frames: int,
    min_overlap_ratio: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(index_path).set_index("asset_id")
    scored_pairs = pd.read_parquet(scored_pairs_path)
    escalated = scored_pairs[scored_pairs["status"] == "escalate"]

    if escalated.empty:
        pd.DataFrame(
            columns=["asset_id_a", "asset_id_b", "offset", "overlap_ratio", "match_count"]
        ).to_parquet(out_path, index=False)
        print("No video pairs escalated to tier 3")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval().to(device)

    rows = []
    for _, pair in tqdm(
        escalated.iterrows(), total=len(escalated), desc="Aligning video pairs", unit="pair"
    ):
        a_id, b_id = pair["asset_id_a"], pair["asset_id_b"]

        try:
            a_path = Path(df.loc[a_id, "path"])
            b_path = Path(df.loc[b_id, "path"])

            a_emb = get_dense_embedding(
                a_id, a_path, frames_dense_dir, clip_dense_dir, model, device, dense_interval_sec
            )
            b_emb = get_dense_embedding(
                b_id, b_path, frames_dense_dir, clip_dense_dir, model, device, dense_interval_sec
            )
            if a_emb is None or b_emb is None:
                continue

            sim_matrix = a_emb @ b_emb.T
            best = best_diagonal_overlap(sim_matrix, align_similarity_threshold)

            shorter_len = min(len(a_emb), len(b_emb))
            overlap_ratio = best["match_count"] / shorter_len if shorter_len else 0.0

            is_duplicate = (
                best["match_count"] >= min_overlap_frames and overlap_ratio >= min_overlap_ratio
            )

            rows.append(
                {
                    "asset_id_a": a_id,
                    "asset_id_b": b_id,
                    "offset": best["offset"],
                    "match_count": best["match_count"],
                    "overlap_ratio": overlap_ratio,
                    "is_duplicate": is_duplicate,
                }
            )
            logger.info(
                "Aligned %s vs %s: offset=%s match_count=%d overlap_ratio=%.3f duplicate=%s",
                a_id,
                b_id,
                best["offset"],
                best["match_count"],
                overlap_ratio,
                is_duplicate,
            )
        except Exception:
            logger.error("Failed to align pair %s / %s", a_id, b_id, exc_info=True)

    result = pd.DataFrame(
        rows,
        columns=[
            "asset_id_a",
            "asset_id_b",
            "offset",
            "match_count",
            "overlap_ratio",
            "is_duplicate",
        ],
    )
    result.to_parquet(out_path, index=False)

    n_dup = int(result["is_duplicate"].sum()) if not result.empty else 0
    print(f"Video pair alignment complete: {n_dup}/{len(result)} pairs confirmed -> {out_path}")
    logger.info("Video pair alignment complete: %d/%d pairs confirmed", n_dup, len(result))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.json"))
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--scored-pairs", type=Path, default=None)
    ap.add_argument("--frames-dense-dir", type=Path, default=None)
    ap.add_argument("--clip-dense-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--log-file", type=Path, default=None)
    ap.add_argument("--dense-interval-sec", type=float, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--pretrained", type=str, default=None)
    ap.add_argument("--align-similarity-threshold", type=float, default=None)
    ap.add_argument("--min-overlap-frames", type=int, default=None)
    ap.add_argument("--min-overlap-ratio", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    section = get_section(cfg, "match_video_pairs")

    index_path = resolve_path(resolve(args.index, section.get("index")))
    scored_pairs_path = resolve_path(resolve(args.scored_pairs, section.get("scored_pairs")))
    frames_dense_dir = resolve_path(
        resolve(args.frames_dense_dir, section.get("frames_dense_dir"))
    )
    clip_dense_dir = resolve_path(resolve(args.clip_dense_dir, section.get("clip_dense_dir")))
    out_path = resolve_path(resolve(args.out, section.get("out")))
    log_file = resolve_path(resolve(args.log_file, section.get("log_file")))
    dense_interval_sec = resolve(args.dense_interval_sec, section.get("dense_interval_sec"))
    model_name = resolve(args.model, section.get("model"))
    pretrained = resolve(args.pretrained, section.get("pretrained"))
    align_similarity_threshold = resolve(
        args.align_similarity_threshold, section.get("align_similarity_threshold")
    )
    min_overlap_frames = resolve(args.min_overlap_frames, section.get("min_overlap_frames"))
    min_overlap_ratio = resolve(args.min_overlap_ratio, section.get("min_overlap_ratio"))

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Starting video pair alignment")
    main(
        index_path,
        scored_pairs_path,
        frames_dense_dir,
        clip_dense_dir,
        out_path,
        dense_interval_sec,
        model_name,
        pretrained,
        align_similarity_threshold,
        min_overlap_frames,
        min_overlap_ratio,
    )
    logger.info("Video pair alignment complete")
