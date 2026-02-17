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

from config import get_section, load_config, resolve, resolve_path

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


def match_detection_to_track(
    track_tlbr: Sequence[float],
    detection_boxes: Sequence[Sequence[float]],
    used_indices: set[int],
    min_iou: float,
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
    iou_match_threshold: float,
    tracker_cfg: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(index_path)

    tracker_args = SimpleNamespace(**tracker_cfg)

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
                    target.tlbr,
                    detection_boxes,
                    used_indices,
                    min_iou=iou_match_threshold,
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
    ap.add_argument("--config", type=Path, default=Path("configs/config.json"))
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--persons-dir", type=Path, default=None)
    ap.add_argument("--faces-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--min-frames-per-entity", type=int, default=None)
    ap.add_argument("--log-file", type=Path, default=None)
    ap.add_argument("--iou-match-threshold", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    section = get_section(cfg, "build_entities")

    index_path = resolve_path(resolve(args.index, section.get("index")))
    persons_dir = resolve_path(resolve(args.persons_dir, section.get("persons_dir")))
    faces_dir = resolve_path(resolve(args.faces_dir, section.get("faces_dir")))
    out_dir = resolve_path(resolve(args.out_dir, section.get("out_dir")))
    min_frames_per_entity = resolve(
        args.min_frames_per_entity, section.get("min_frames_per_entity")
    )
    log_file = resolve_path(resolve(args.log_file, section.get("log_file")))
    iou_match_threshold = resolve(
        args.iou_match_threshold, section.get("iou_match_threshold")
    )
    tracker_cfg = section.get("tracker", {})

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logger.info("Starting entity construction process")

    main(
        index_path,
        persons_dir,
        faces_dir,
        out_dir,
        min_frames_per_entity,
        iou_match_threshold,
        tracker_cfg,
    )

    logger.info("Entity construction process complete")
