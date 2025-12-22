import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

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
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("cache/file_index.parquet"))
    ap.add_argument("--log-file", type=Path, default=Path("logs/index_files.log"))
    args = ap.parse_args()

    logging.basicConfig(
        filename=str(args.log_file),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logger.info("Starting file indexing")

    main(args.input_dir, args.out)

    logger.info("File indexing completed")
