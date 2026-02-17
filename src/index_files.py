import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import get_section, load_config, resolve, resolve_path
from utils import (
    asset_id_for_file,
    cv2_image_metadata,
    ffprobe_metadata,
    is_image,
    is_video,
)

logger = logging.getLogger(__name__)


def main(input_dir: Path, out_path: Path):
    records = []

    failures = 0

    for path in tqdm(input_dir.rglob("*"), desc="Indexing files", unit="file"):
        if not path.is_file():
            continue

        if not (is_image(path) or is_video(path)):
            continue

        aid = asset_id_for_file(path)
        rec = {
            "asset_id": aid,
            "path": str(path.resolve()),
            "type": "video" if is_video(path) else "image",
            "mtime": int(path.stat().st_mtime),
            "size": path.stat().st_size,
            "num_frames": None,
        }

        if rec["type"] == "video":
            try:
                meta = ffprobe_metadata(path)
                rec.update(meta)
                logger.info(f"Indexed video: {path} (ID: {aid})")
            except Exception:
                failures += 1
                rec.update(
                    {"width": None, "height": None, "fps": None, "duration": None}
                )
                logger.error(
                    f"Failed to index video: {path} (ID: {aid})", exc_info=False
                )
        elif rec["type"] == "image":
            try:
                meta = cv2_image_metadata(path)
                rec.update(meta)
                logger.info(f"Indexed image: {path} (ID: {aid})")
            except Exception:
                failures += 1
                rec.update(
                    {"width": None, "height": None, "fps": None, "duration": None}
                )
                logger.error(
                    f"Failed to index image: {path} (ID: {aid})", exc_info=False
                )
        else:
            failures += 1
            rec.update({"width": None, "height": None, "fps": None, "duration": None})
            logger.warning(f"Indexed unknown file type: {path} (ID: {aid})")

        records.append(rec)

    df = pd.DataFrame(records).sort_values("asset_id")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"Indexed {len(df) - failures} assets ({failures} failures) -> {out_path}")
    logger.info(
        f"Indexed {len(df) - failures} assets ({failures} failures) -> {out_path}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.json"))
    ap.add_argument("--input-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--log-file", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    section = get_section(cfg, "index_files")

    input_dir = resolve_path(resolve(args.input_dir, section.get("input_dir")))
    if input_dir is None:
        raise ValueError("input_dir must be provided via CLI or config")
    out_path = resolve_path(resolve(args.out, section.get("out")))
    log_file = resolve_path(resolve(args.log_file, section.get("log_file")))

    logging.basicConfig(
        filename=str(log_file),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logger.info("Starting file indexing")

    main(input_dir, out_path)

    logger.info("File indexing completed")
