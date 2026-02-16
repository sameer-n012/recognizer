import argparse
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm
from yolox.tracker.byte_tracker import BYTETracker

logger = logging.getLogger(__name__)

DEFAULT_MIN_FRAMES_PER_ENTITY = 3
IOU_MATCH_THRESHOLD = 0.2


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


def match_detection_to_track(
    track_tlbr: Sequence[float],
    detection_boxes: Sequence[Sequence[float]],
    used_indices: set[int],
    min_iou: float = IOU_MATCH_THRESHOLD,
) -> int | None:
    """
    Return the detection index whose bounding box has the highest IoU with the
    current track bounding box, skipping already assigned detections.
    """
    best_idx: int | None = None
    best_iou = min_iou
    for idx, det_box in enumerate(detection_boxes):
        if idx in used_indices:
            continue
        current_iou = bbox_iou(track_tlbr, det_box)
        if current_iou > best_iou:
            best_iou = current_iou
            best_idx = idx
    return best_idx


def main(
    index_path: Path,
    persons_dir: Path,
    faces_dir: Path,
    out_dir: Path,
    min_frames_per_entity: int,
) -> None:
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
        for face in faces:
            faces_by_det[face["person_det_index"]].append(face)

        detections_by_frame = load_detections_by_frame(persons)

        if row["type"] == "image":
            for det_index, det in enumerate(persons):
                eid = make_entity_id(asset_id, det_index)
                payload = {
                    "entity_id": eid,
                    "asset_id": asset_id,
                    "type": "image",
                    "frames": [
                        {
                            "source": det["source"],
                            "person_bbox": det["bbox"],
                            "faces": faces_by_det.get(det_index, []),
                        }
                    ],
                    "has_face": bool(faces_by_det.get(det_index)),
                }
                with open(out_dir / f"{eid}.json", "w") as f:
                    json.dump(payload, f)
            continue

        tracks = defaultdict(list)
        frame_names = sorted(detections_by_frame.keys())

        for frame_name in frame_names:
            detections = detections_by_frame[frame_name]

            if not detections:
                tracker.update(
                    np.empty((0, 5), dtype=np.float32),
                    img_info=img_size,
                    img_size=img_size,
                )
                continue

            tlwh_boxes = []
            detection_boxes: list[list[float]] = []

            for det_index, det in detections:
                x1, y1, x2, y2 = det["bbox"]
                confidence = float(det["confidence"])
                tlwh_boxes.append([x1, y1, x2 - x1, y2 - y1, confidence])
                detection_boxes.append([x1, y1, x2, y2])

            tlwh_boxes = np.asarray(tlwh_boxes, dtype=np.float32)
            online_targets = tracker.update(tlwh_boxes, img_size, img_size)
            used_indices: set[int] = set()

            for target in online_targets:
                if target.tlwh is None:
                    continue

                match_idx = match_detection_to_track(
                    target.tlbr, detection_boxes, used_indices
                )
                if match_idx is None:
                    continue

                det_index, detection = detections[match_idx]
                used_indices.add(match_idx)

                tracks[target.track_id].append(
                    {
                        "source": frame_name,
                        "person_bbox": detection_boxes[match_idx],
                        "faces": faces_by_det.get(det_index, []),
                    }
                )

        retained_tracks = {
            tid: frames
            for tid, frames in tracks.items()
            if len(frames) >= min_frames_per_entity
        }

        for track_id, frames in retained_tracks.items():
            eid = make_entity_id(asset_id, track_id)
            payload = {
                "entity_id": eid,
                "asset_id": asset_id,
                "type": "video",
                "frames": frames,
                "has_face": any(len(frame["faces"]) > 0 for frame in frames),
            }
            with open(out_dir / f"{eid}.json", "w") as f:
                json.dump(payload, f)

        logger.info(
            "Processed %s: %d tracks retained (%d dropped)",
            asset_id,
            len(retained_tracks),
            len(tracks) - len(retained_tracks),
        )

    logger.info("Entity construction (ByteTrack) complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("cache/file_index.parquet"))
    ap.add_argument("--persons-dir", type=Path, default=Path("data/detections/persons"))
    ap.add_argument("--faces-dir", type=Path, default=Path("data/detections/faces"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/entities"))
    ap.add_argument(
        "--min-frames-per-entity",
        type=int,
        default=DEFAULT_MIN_FRAMES_PER_ENTITY,
    )
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
