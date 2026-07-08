import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

from config import get_section, load_config, resolve, resolve_path

logger = logging.getLogger(__name__)


def main(
    index_path: Path,
    frames_root: Path,
    out_dir: Path,
    confidence_threshold: float,
    model_path: Path,
):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(model_path)).to(device)

    logger.info(f"Running YOLOv8s on device: {device}")

    df = pd.read_parquet(index_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Processing assets", unit="asset"
    ):
        asset_id = row["asset_id"]

        try:
            out_path = out_dir / f"{asset_id}.json"
            if out_path.exists():
                continue

            detections = []

            if row["type"] == "image":
                sources = [Path(row["path"])]
            else:
                frame_dir = frames_root / asset_id
                if not frame_dir.exists():
                    continue
                sources = sorted(frame_dir.glob("*.jpg"))

            for src in sources:
                results = model(
                    str(src),
                    conf=confidence_threshold,
                    classes=[0],  # person only
                    verbose=False,
                )[0]

                for box in results.boxes:
                    detections.append(
                        {
                            "source": src.name,
                            "bbox": box.xyxy[0].tolist(),
                            "confidence": float(box.conf[0]),
                        }
                    )

            payload = {
                "asset_id": asset_id,
                "type": row["type"],
                "detections": detections,
            }

            with open(out_path, "w") as f:
                json.dump(payload, f)

            logger.info(
                f"Processed asset: {asset_id}, detections: {len(detections)}, sources: {len(sources)}"
            )

        except Exception:
            failures += 1
            logger.error(f"Error processing asset {asset_id}", exc_info=False)

    print(f"Person detection complete ({failures} failures)")
    logging.info(f"Person detection complete ({failures} failures)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.json"))
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--frames-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--log-file", type=Path, default=None)
    ap.add_argument("--confidence-threshold", type=float, default=None)
    ap.add_argument("--model-path", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    section = get_section(cfg, "person_detection")

    index_path = resolve_path(resolve(args.index, section.get("index")))
    frames_dir = resolve_path(resolve(args.frames_dir, section.get("frames_dir")))
    out_dir = resolve_path(resolve(args.out_dir, section.get("out_dir")))
    log_file = resolve_path(resolve(args.log_file, section.get("log_file")))
    confidence_threshold = resolve(
        args.confidence_threshold, section.get("confidence_threshold")
    )
    model_path = resolve_path(resolve(args.model_path, section.get("model_path")))

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Starting person detection process")

    main(index_path, frames_dir, out_dir, confidence_threshold, model_path)

    logger.info("Person detection process completed")
