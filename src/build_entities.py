import argparse
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from tqdm import tqdm
from yolox.tracker.byte_tracker import BYTETracker

logger = logging.getLogger(__name__)


def make_entity_id(asset_id, track_id):
    h = hashlib.sha1()
    h.update(asset_id.encode())
    h.update(str(track_id).encode())
    return h.hexdigest()


def load_detections_by_frame(persons):
    frames = defaultdict(list)
    for i, det in enumerate(persons):
        frames[det["source"]].append((i, det))
    return frames


def bbox_iou(box_a, box_b):
    # intersections first
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_width = max(0.0, xB - xA)
    inter_height = max(0.0, yB - yA)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union_area = max(area_a + area_b - inter_area, 1e-6)
    return inter_area / union_area


def main(index_path, persons_dir, faces_dir, out_dir, min_frames_per_entity):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(index_path)

    tracker_args = SimpleNamespace(
        track_thresh=0.3,
        match_thresh=0.8,
        frame_rate=30,
        track_buffer=30,
        mot20=False,
    )

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Building entities", unit="asset"
    ):
        tracker = BYTETracker(tracker_args, frame_rate=tracker_args.frame_rate)

        if not row["width"] or not row["height"]:
            continue

        asset_id = row["asset_id"]
        img_size = (row["height"], row["width"])

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

        faces_by_det = defaultdict(list)
        for f in faces:
            faces_by_det[f["person_det_index"]].append(f)

        detections_by_frame = load_detections_by_frame(persons)

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
                            "faces": faces_by_det.get(i, []),
                        }
                    ],
                    "has_face": i in faces_by_det,
                }
                with open(out_dir / f"{eid}.json", "w") as f:
                    json.dump(payload, f)
            continue

        tracks = defaultdict(list)
        frame_names = sorted(detections_by_frame.keys())

        # tracker.reset()

        for frame_idx, frame_name in enumerate(frame_names):
            dets = detections_by_frame[frame_name]

            if not dets:
                # tracker.update(np.empty((0, 5)))
                tracker.update(
                    np.empty((0, 5), dtype=np.float32),
                    img_info=img_size,
                    img_size=img_size,
                )
                continue

            boxes = []
            det_indices = []

            for det_idx, det in dets:
                x1, y1, x2, y2 = det["bbox"]
                conf = det["confidence"]
                boxes.append([x1, y1, x2 - x1, y2 - y1, conf])
                det_indices.append(det_idx)

            boxes = np.asarray(boxes, dtype=np.float32)
            online_targets = tracker.update(boxes, img_size, img_size)

            for t_idx, t in enumerate(online_targets):
                if t.tlwh is None:
                    continue

                track_id = t.track_id
                tlwh = t.tlwh
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h

                # find closest detection (IoU not needed; ByteTrack keeps order)
                det_idx = det_indices[t_idx]

                tracks[track_id].append(
                    {
                        "source": frame_name,
                        "person_bbox": [x1, y1, x2, y2],
                        "faces": faces_by_det.get(det_idx, []),
                    }
                )

        for track_id, frames in tracks.items():
            if len(frames) < min_frames_per_entity:
                continue

            eid = make_entity_id(asset_id, track_id)

            payload = {
                "entity_id": eid,
                "asset_id": asset_id,
                "type": "video",
                "frames": frames,
                "has_face": any(len(f["faces"]) > 0 for f in frames),
            }

            with open(out_dir / f"{eid}.json", "w") as f:
                json.dump(payload, f)

        logger.info(f"Processed asset: {asset_id}, entities found: {len(tracks)}")

    logger.info("Entity construction (ByteTrack) complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("cache/file_index.parquet"))
    ap.add_argument("--persons-dir", type=Path, default=Path("data/detections/persons"))
    ap.add_argument("--faces-dir", type=Path, default=Path("data/detections/faces"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/entities"))
    ap.add_argument("--min-frames-per-entity", type=int, default=3)
    ap.add_argument("--log-file", type=Path, default=Path("logs/build_entities.log"))
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

    logger.info("Entity construction process complete")
