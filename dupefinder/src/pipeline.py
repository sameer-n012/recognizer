import argparse
import logging
from pathlib import Path

from build_candidates import main as build_candidates_main
from cluster_duplicates import main as cluster_duplicates_main
from coarse_signature import main as coarse_signature_main
from config import get_section, load_config, resolve_path
from exact_duplicates import main as exact_duplicates_main
from index_files import main as index_files_main
from match_pairs import main as match_pairs_main
from match_video_pairs import main as match_video_pairs_main
from near_duplicates import main as near_duplicates_main
from rank_group_members import main as rank_group_members_main

logger = logging.getLogger(__name__)

STAGE_ORDER = [
    "index",
    "exact_duplicates",
    "coarse_signature",
    "near_duplicates",
    "build_candidates",
    "match_pairs",
    "match_video_pairs",
    "cluster_duplicates",
    "rank_group_members",
]

STAGE_MAP = {
    0: "index",
    1: "exact_duplicates",
    2: "coarse_signature",
    3: "near_duplicates",
    4: "build_candidates",
    5: "match_pairs",
    6: "match_video_pairs",
    7: "cluster_duplicates",
    8: "rank_group_members",
}

# Maps a pipeline stage name to its configs/config.json section — identical except
# "index", whose section is named "index_files".
STAGE_CONFIG_SECTION = {
    "index": "index_files",
    "exact_duplicates": "exact_duplicates",
    "coarse_signature": "coarse_signature",
    "near_duplicates": "near_duplicates",
    "build_candidates": "build_candidates",
    "match_pairs": "match_pairs",
    "match_video_pairs": "match_video_pairs",
    "cluster_duplicates": "cluster_duplicates",
    "rank_group_members": "rank_group_members",
}

DATA_LAYOUT = {
    "cache": None,
    "logs": None,
    "data": {
        "frames_sparse": None,
        "frames_dense": None,
        "hashes": None,
        "embeddings": {
            "clip_coarse": None,
            "clip_dense": None,
        },
        "candidates": None,
        "duplicates": None,
    },
}


def ensure_data_layout(base_dir: Path) -> None:
    for key, sub_layout in DATA_LAYOUT.items():
        current_path = base_dir / key
        if not current_path.exists():
            current_path.mkdir(parents=True)
        if sub_layout is not None:
            ensure_data_layout_nested(current_path, sub_layout)


def ensure_data_layout_nested(base_dir: Path, layout: dict) -> None:
    for key, sub_layout in layout.items():
        current_path = base_dir / key
        if not current_path.exists():
            current_path.mkdir(parents=True)
        if sub_layout is not None:
            ensure_data_layout_nested(current_path, sub_layout)


def configure_stage_logging(log_path: Path) -> None:
    """Reconfigure the root logger to point at this stage's own log file, matching
    what each stage's standalone __main__ block already does. Every stage's logger
    propagates to the root config, so without this a single pipeline-wide log file (or
    worse, the console) would either interleave all stages together or garble tqdm's
    live progress bars with per-item logger.info/error calls. force=True drops the
    previous stage's file handler before attaching this one."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_path),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )


def run_pipeline(
    stages: list[str],
    config_path: Path,
    input_dir: Path,
    data_dir: Path,
) -> None:
    cfg = load_config(config_path)

    stages = [STAGE_MAP[int(s)] if s.isdigit() else s for s in stages]

    ensure_data_layout(data_dir)

    for stage in STAGE_ORDER:
        if stage not in stages:
            continue

        section = get_section(cfg, STAGE_CONFIG_SECTION[stage])
        configure_stage_logging(resolve_path(section["log_file"], data_dir))

        logger.info("Starting stage: %s", stage)
        if stage == "index":
            out_path = resolve_path(section.get("out"), data_dir)
            index_files_main(input_dir, out_path)
        elif stage == "exact_duplicates":
            exact_duplicates_main(
                resolve_path(section["index"], data_dir),
                resolve_path(section["out"], data_dir),
            )
        elif stage == "coarse_signature":
            coarse_signature_main(
                resolve_path(section["index"], data_dir),
                resolve_path(section["exact_groups"], data_dir),
                resolve_path(section["frames_dir"], data_dir),
                resolve_path(section["hashes_out"], data_dir),
                resolve_path(section["clip_out_dir"], data_dir),
                section["max_frames"],
                section["max_frame_rate"],
                section["hash_size"],
                section["model"],
                section["pretrained"],
                section["batch_size"],
            )
        elif stage == "near_duplicates":
            near_duplicates_main(
                resolve_path(section["exact_groups"], data_dir),
                resolve_path(section["hashes"], data_dir),
                resolve_path(section["out"], data_dir),
                section["ahash_threshold"],
                section["dhash_threshold"],
                section["phash_threshold"],
                section["min_frame_match_ratio"],
            )
        elif stage == "build_candidates":
            build_candidates_main(
                resolve_path(section["index"], data_dir),
                resolve_path(section["exact_groups"], data_dir),
                resolve_path(section["near_dup_groups"], data_dir),
                resolve_path(section["clip_dir"], data_dir),
                resolve_path(section["out"], data_dir),
                section["top_k"],
                section["similarity_floor"],
            )
        elif stage == "match_pairs":
            match_pairs_main(
                resolve_path(section["index"], data_dir),
                resolve_path(section["candidates"], data_dir),
                resolve_path(section["out"], data_dir),
                section["resolve_threshold"],
                section["escalate_threshold"],
            )
        elif stage == "match_video_pairs":
            match_video_pairs_main(
                resolve_path(section["index"], data_dir),
                resolve_path(section["scored_pairs"], data_dir),
                resolve_path(section["frames_dense_dir"], data_dir),
                resolve_path(section["clip_dense_dir"], data_dir),
                resolve_path(section["out"], data_dir),
                section["dense_interval_sec"],
                section["model"],
                section["pretrained"],
                section["align_similarity_threshold"],
                section["min_overlap_frames"],
                section["min_overlap_ratio"],
            )
        elif stage == "cluster_duplicates":
            cluster_duplicates_main(
                resolve_path(section["index"], data_dir),
                resolve_path(section["exact_groups"], data_dir),
                resolve_path(section["near_dup_groups"], data_dir),
                resolve_path(section["scored_pairs"], data_dir),
                resolve_path(section["video_alignment"], data_dir),
                resolve_path(section["out"], data_dir),
            )
        elif stage == "rank_group_members":
            rank_group_members_main(
                resolve_path(section["index"], data_dir),
                resolve_path(section["groups"], data_dir),
                resolve_path(section["out"], data_dir),
                section["duration_epsilon_seconds"],
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

    # Logging is configured per-stage inside run_pipeline (configure_stage_logging),
    # so each stage writes to its own logs/<stage>.log — same as running it standalone
    # — instead of one pipeline-wide file or, worse, the console garbling tqdm's live
    # progress bars with per-item logger.info/error calls.
    run_pipeline(args.stages, args.config, args.input_dir, args.data_dir)
