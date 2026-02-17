import argparse
import logging
from pathlib import Path

from build_entities import main as build_entities_main
from cluster_entites import main as cluster_main
from config import get_section, load_config, resolve, resolve_path
from embed_bodies import main as embed_bodies_main
from embed_clip import main as embed_clip_main
from embed_faces import main as embed_faces_main
from extract_frames import main as extract_frames_main
from face_detection import main as face_detection_main
from fuse_embeddings import main as fuse_embeddings_main
from index_files import main as index_files_main
from person_detection import main as person_detection_main
from refine_clusters import main as refine_clusters_main

logger = logging.getLogger(__name__)

STAGE_ORDER = [
    "index",
    "extract_frames",
    "person_detection",
    "face_detection",
    "build_entities",
    "embed_faces",
    "embed_bodies",
    "embed_clip",
    "fuse_embeddings",
    "cluster_entities",
    "refine_clusters",
]

STAGE_MAP = {
    0: "index",
    1: "extract_frames",
    2: "person_detection",
    3: "face_detection",
    4: "build_entities",
    5: "embed_faces",
    6: "embed_bodies",
    7: "embed_clip",
    8: "fuse_embeddings",
    9: "cluster_entities",
    10: "refine_clusters",
}

DATA_LAYOUT = {
    "cache": None,
    "logs": None,
    "data": {
        "clusters": {
            "refined": None,
        },
        "detections": {
            "faces": None,
            "persons": None,
        },
        "embeddings": {
            "body": None,
            "clip": None,
            "face": None,
            "fused": None,
        },
        "entities": None,
        "frames": None,
        "raw": {
            "images": None,
            "videos": None,
        },
    },
}


def ensure_data_layout(base_dir: Path) -> None:
    for key, sub_layout in DATA_LAYOUT.items():
        current_path = base_dir / key
        if not current_path.exists():
            current_path.mkdir(parents=True)
        if sub_layout is not None:
            ensure_data_layout(current_path)


def run_pipeline(
    stages: list[str],
    config_path: Path,
    input_dir: Path,
    data_dir: Path,
) -> None:
    cfg = load_config(config_path)

    stages = [STAGE_MAP[int(s)] if s.isdigit() else s for s in stages]

    ensure_data_layout(data_dir)

    for idx, stage in enumerate(STAGE_ORDER):
        if stage is None or stage not in stages:
            continue

        logger.info("Starting stage: %s", stage)
        if stage == "index":
            section = get_section(cfg, "index_files")
            out_path = resolve_path(section.get("out"), data_dir)
            index_files_main(input_dir, out_path)
        elif stage == "extract_frames":
            section = get_section(cfg, "extract_frames")
            extract_frames_main(
                resolve_path(section.get("index"), data_dir),
                resolve_path(section.get("frames_dir"), data_dir),
                section["max_frames"],
                section["max_frame_rate"],
            )
        elif stage == "person_detection":
            section = get_section(cfg, "person_detection")
            person_detection_main(
                resolve_path(section["index"], data_dir),
                resolve_path(section["frames_dir"], data_dir),
                resolve_path(section["out_dir"], data_dir),
                section["confidence_threshold"],
                Path(section["model_path"]),
            )
        elif stage == "face_detection":
            section = get_section(cfg, "face_detection")
            face_detection_main(
                resolve_path(section["index"], data_dir),
                resolve_path(section["frames_dir"], data_dir),
                resolve_path(section["persons_dir"], data_dir),
                resolve_path(section["out_dir"], data_dir),
                tuple(section["det_size"]),
                section["conf_threshold"],
                section["save_embeddings"],
                section["no_cache"],
                section["insightface_root"],
                section["model"],
                section["providers"],
            )
        elif stage == "build_entities":
            section = get_section(cfg, "build_entities")
            build_entities_main(
                resolve_path(section["index"], data_dir),
                resolve_path(section["persons_dir"], data_dir),
                resolve_path(section["faces_dir"], data_dir),
                resolve_path(section["out_dir"], data_dir),
                section["min_frames_per_entity"],
                section["iou_match_threshold"],
                section["tracker"],
            )
        elif stage == "embed_faces":
            section = get_section(cfg, "embed_faces")
            embed_faces_main(
                resolve_path(section["entities_dir"], data_dir),
                resolve_path(section["frames_dir"], data_dir),
                resolve_path(section["out_dir"], data_dir),
                resolve_path(section["index_path"], data_dir),
                section["min_confidence"],
                section["face_pad_ratio"],
                section["match_iou"],
                section["model"],
                section["providers"],
            )
        elif stage == "embed_bodies":
            section = get_section(cfg, "embed_bodies")
            embed_bodies_main(
                resolve_path(section["entities_dir"], data_dir),
                resolve_path(section["frames_dir"], data_dir),
                resolve_path(section["index"], data_dir),
                resolve_path(section["out_dir"], data_dir),
                section["input_size"],
                section["batch_size"],
                section["model"],
            )
        elif stage == "embed_clip":
            section = get_section(cfg, "embed_clip")
            embed_clip_main(
                resolve_path(section["entities_dir"], data_dir),
                resolve_path(section["frames_dir"], data_dir),
                resolve_path(section["index_path"], data_dir),
                resolve_path(section["out_dir"], data_dir),
                section["batch_size"],
                section["model"],
                section["pretrained"],
            )
        elif stage == "fuse_embeddings":
            section = get_section(cfg, "fuse_embeddings")
            fuse_embeddings_main(
                resolve_path(section["entities_dir"], data_dir),
                resolve_path(section["face_dir"], data_dir),
                resolve_path(section["body_dir"], data_dir),
                resolve_path(section["clip_dir"], data_dir),
                resolve_path(section["out_dir"], data_dir),
                {
                    "face_present": {
                        "face": section["face_weights"][0],
                        "body": section["face_weights"][1],
                        "clip": section["face_weights"][2],
                    },
                    "no_face": {
                        "face": section["no_face_weights"][0],
                        "body": section["no_face_weights"][1],
                        "clip": section["no_face_weights"][2],
                    },
                    "no_person": {
                        "face": section["no_person_weights"][0],
                        "body": section["no_person_weights"][1],
                        "clip": section["no_person_weights"][2],
                    },
                },
            )
        elif stage == "cluster_entities":
            section = get_section(cfg, "cluster_entities")
            cluster_main(
                resolve_path(section["fused_dir"], data_dir),
                resolve_path(section["out_dir"], data_dir),
                section["similarity"],
            )
        elif stage == "refine_clusters":
            section = get_section(cfg, "refine_clusters")
            refine_clusters_main(
                resolve_path(section["cluster_dir"], data_dir),
                resolve_path(section["out_dir"], data_dir),
                section,
                section["merge_threshold"],
                section["noise_threshold"],
            )
        logger.info("Completed stage: %s", stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.json"),
        help="Pipeline config file",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_ORDER,
        default=STAGE_ORDER,
        help="Stages to run, in order",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Media input directory for indexing",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory that contains data/, cache/, logs/",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    run_pipeline(args.stages, args.config, args.input_dir, args.data_dir)
