import json
import logging
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from readerwriterlock import rwlock

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "server" / "config.json"
TEMPLATE_DIR = Path("templates")
DEFAULT_PORT = 5034

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder="static")


def load_config(data_dir: Path | None = None) -> Dict[str, Any]:
    with CONFIG_PATH.open() as cfg_file:
        raw = json.load(cfg_file)

    base_data_dir = data_dir or PROJECT_ROOT
    return {
        "groups_path": base_data_dir / Path(raw["groups_path"]),
        "overrides_path": base_data_dir / Path(raw["overrides_path"]),
        "file_index": base_data_dir / Path(raw["file_index"]),
    }


def ensure_data_layout(base: Path) -> None:
    for sub in ("data/duplicates", "cache", "logs"):
        (base / sub).mkdir(parents=True, exist_ok=True)


context: Dict[str, Any] = {
    "config": load_config(),
    "file_index": None,
    "asset_lookup": {},
    "groups": {},
    "undo_stack": [],
    "redo_stack": [],
    "lock": rwlock.RWLockFairD(),
}


def load_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["asset_id", "group_id"])
    return pd.read_parquet(path)


# Keep in sync with src/rank_group_members.py's duration_epsilon_seconds default —
# both implement the same keeper heuristic independently (dashboard recomputes live
# rather than trusting ranked.parquet, see refresh_context).
DURATION_EPSILON_SECONDS = 1.5


def compute_duration_bucket(row: pd.Series) -> float:
    """Videos within DURATION_EPSILON_SECONDS of each other are treated as "the same
    length" (same bucket); anything longer wins outright, ahead of resolution — a
    higher-resolution video that's only a subsection of another should never outrank
    the full-length one."""
    duration = row.get("duration")
    if row["type"] != "video" or not duration or duration <= 0:
        return 0.0
    return round(float(duration) / DURATION_EPSILON_SECONDS) * DURATION_EPSILON_SECONDS


def compute_quality_key(row: pd.Series) -> float:
    if row["type"] == "video" and row.get("duration") and row["duration"] > 0:
        return float(row["size"]) / float(row["duration"])
    return float(row["size"])


def compute_effective_groups(
    file_index: pd.DataFrame, base_groups: pd.DataFrame, overrides: pd.DataFrame
) -> Dict[int, list[str]]:
    """All effective groups, including singletons (size 1) — a group shrinks to a
    singleton when a user removes members down to one, or removes an asset from a
    duplicate bucket via the dashboard. Callers filter by size as needed."""
    assignment: Dict[str, int] = {}
    for _, row in base_groups.iterrows():
        assignment[row["asset_id"]] = int(row["group_id"])
    for _, row in overrides.iterrows():
        assignment[row["asset_id"]] = int(row["group_id"])

    groups: Dict[int, list[str]] = defaultdict(list)
    for asset_id, group_id in assignment.items():
        if asset_id in file_index.index:
            groups[group_id].append(asset_id)

    return dict(groups)


def next_singleton_id(base_groups: pd.DataFrame, overrides: pd.DataFrame) -> int:
    ids = []
    if not base_groups.empty:
        ids.append(int(base_groups["group_id"].max()))
    if not overrides.empty:
        ids.append(int(overrides["group_id"].max()))
    return (max(ids) + 1) if ids else 1


def add_unmatched_singletons(
    file_index: pd.DataFrame, groups: Dict[int, list[str]]
) -> Dict[int, list[str]]:
    """Assets the pipeline never flagged as a duplicate of anything don't have a row
    in groups.parquet or overrides at all, so they're otherwise invisible even with
    "show singletons" on. Surface them as ephemeral negative-id singleton groups so
    they're browsable and can be manually merged into a real bucket if a reviewer
    spots something the pipeline missed. Negative ids are never written to overrides —
    merge_groups always resolves a persisted id before writing (see next_singleton_id
    usage there), so these are purely a view-layer convenience, recomputed fresh (and
    is thus safe to allocate arbitrarily) on every refresh."""
    assigned = {asset_id for members in groups.values() for asset_id in members}
    unmatched = sorted(set(file_index.index) - assigned)
    extended = dict(groups)
    for i, asset_id in enumerate(unmatched):
        extended[-(i + 1)] = [asset_id]
    return extended


def serialize_overrides(overrides: pd.DataFrame) -> list[dict[str, Any]]:
    if overrides.empty:
        return []
    return overrides[["asset_id", "group_id"]].to_dict(orient="records")


def write_overrides(overrides: pd.DataFrame, overrides_path: Path) -> pd.DataFrame:
    overrides = overrides.drop_duplicates(subset=["asset_id"], keep="last")
    overrides_path.parent.mkdir(parents=True, exist_ok=True)
    overrides.to_parquet(overrides_path, index=False)
    return overrides


def refresh_context() -> None:
    with context["lock"].gen_wlock():
        config = context["config"]
        file_index = pd.read_parquet(config["file_index"]).set_index("asset_id", drop=False)
        base_groups = load_pairs(config["groups_path"])
        overrides = load_pairs(config["overrides_path"])

        context["file_index"] = file_index
        context["asset_lookup"] = {
            row["asset_id"]: row["path"] for _, row in file_index.iterrows()
        }
        context["base_groups"] = base_groups
        context["overrides"] = overrides
        groups = compute_effective_groups(file_index, base_groups, overrides)
        context["groups"] = add_unmatched_singletons(file_index, groups)

        logger.info(
            "Context refreshed (%d duplicate groups, %d assets indexed)",
            len(context["groups"]),
            len(file_index),
        )


refresh_context()

# Bucket list page size, and how many members of each bucket are included inline in
# that page before the client has to page further into a specific bucket — both keep
# a single /api/groups response bounded regardless of how many duplicates or how large
# any one bucket is, since eagerly serializing every member of every bucket doesn't
# scale to a large library.
GROUP_PAGE_SIZE_DEFAULT = 24
MEMBER_PAGE_SIZE_DEFAULT = 12


def rank_members(file_index: pd.DataFrame, member_ids: list[str]) -> list[dict[str, Any]]:
    """Every member's metadata, sorted keeper-first then descending quality — the same
    order used by rank_group_members.py's heuristic. This is the single canonical
    ordering shared by the truncated preview in /api/groups and the paginated
    /api/group/<id>/members, so paging through a large bucket never skips or repeats
    an item."""
    scored = []
    for asset_id in member_ids:
        if asset_id not in file_index.index:
            continue
        row = file_index.loc[asset_id]
        duration_bucket = compute_duration_bucket(row)
        resolution = float(row["width"] or 0) * float(row["height"] or 0)
        quality_key = compute_quality_key(row)
        key = (duration_bucket, resolution, quality_key, row["mtime"])
        scored.append((key, asset_id, row))
    scored.sort(key=lambda entry: entry[0], reverse=True)

    members = []
    for asset_id, row in ((a, r) for _, a, r in scored):
        members.append(
            {
                "asset_id": asset_id,
                "path": row["path"],
                "basename": Path(row["path"]).name,
                "type": row["type"],
                "width": None if pd.isna(row["width"]) else int(row["width"]),
                "height": None if pd.isna(row["height"]) else int(row["height"]),
                "size": int(row["size"]),
            }
        )
    if members:
        members[0]["is_keeper"] = True
        for member in members[1:]:
            member["is_keeper"] = False
    return members


def build_group_payload(
    group_id: int,
    member_ids: list[str],
    member_offset: int = 0,
    member_limit: int | None = None,
) -> dict[str, Any]:
    ranked = rank_members(context["file_index"], member_ids)
    total = len(ranked)
    page = ranked[member_offset : member_offset + member_limit] if member_limit else ranked[member_offset:]

    return {
        "group_id": group_id,
        "size": total,
        "is_singleton": total == 1,
        "keeper_asset_id": ranked[0]["asset_id"] if ranked else None,
        "members": page,
        "member_offset": member_offset,
        "has_more_members": member_offset + len(page) < total,
    }


@app.route("/")
def index():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    return render_template("index.html")


@app.route("/api/refresh")
def refresh():
    refresh_context()
    return "Refreshed", 200


@app.route("/api/groups")
def get_groups():
    include_singletons = request.args.get("include_singletons", "").lower() in (
        "1",
        "true",
        "yes",
    )
    offset = max(0, request.args.get("offset", 0, type=int))
    limit = max(1, request.args.get("limit", GROUP_PAGE_SIZE_DEFAULT, type=int))
    member_limit = max(1, request.args.get("member_limit", MEMBER_PAGE_SIZE_DEFAULT, type=int))

    with context["lock"].gen_rlock():
        entries = list(context["groups"].items())

    # Bucket size is just len(members) — cheap enough to sort/filter/count/sum on
    # before paying for rank_members' per-bucket file_index lookups, which only run
    # for the page actually being returned.
    duplicate_count = sum(1 for _, members in entries if len(members) > 1)
    singleton_count = sum(1 for _, members in entries if len(members) == 1)
    duplicate_asset_count = sum(len(members) for _, members in entries if len(members) > 1)

    if not include_singletons:
        entries = [(gid, members) for gid, members in entries if len(members) > 1]

    entries.sort(key=lambda entry: len(entry[1]), reverse=True)
    total = len(entries)
    page = entries[offset : offset + limit]
    payloads = [
        build_group_payload(gid, members, member_limit=member_limit) for gid, members in page
    ]

    return jsonify(
        {
            "groups": payloads,
            "offset": offset,
            "total": total,
            "has_more": offset + len(page) < total,
            "duplicate_asset_count": duplicate_asset_count,
            "duplicate_bucket_count": duplicate_count,
            "singleton_bucket_count": singleton_count,
        }
    )


@app.route("/api/group/<group_id>/members")
def get_group_members(group_id: str):
    try:
        gid = int(group_id)
    except ValueError:
        return jsonify({"error": "Invalid group_id"}), 400

    offset = max(0, request.args.get("offset", 0, type=int))
    limit = max(1, request.args.get("limit", MEMBER_PAGE_SIZE_DEFAULT, type=int))

    with context["lock"].gen_rlock():
        member_ids = context["groups"].get(gid)

    if member_ids is None:
        return jsonify({"error": "Bucket not found"}), 404

    return jsonify(build_group_payload(gid, member_ids, member_offset=offset, member_limit=limit))


@app.route("/api/group/merge", methods=["POST"])
def merge_groups():
    payload = request.get_json(silent=True) or {}
    group_ids = payload.get("group_ids", [])
    if not group_ids or len(group_ids) < 2:
        return jsonify({"error": "Provide at least two group_ids to merge"}), 400

    with context["lock"].gen_wlock():
        config = context["config"]
        groups = context["groups"]
        overrides = context["overrides"]
        base_groups = context["base_groups"]

        # Ephemeral "unique" pseudo-groups (see add_unmatched_singletons) use negative
        # ids that are never persisted — merging must always resolve to a real,
        # persistable id: the lowest real id among the selection, or a freshly
        # allocated one if every selected group is a unique pseudo-group.
        real_ids = [int(gid) for gid in group_ids if int(gid) >= 0]
        target_id = min(real_ids) if real_ids else next_singleton_id(base_groups, overrides)
        context["undo_stack"].append(serialize_overrides(overrides))
        context["redo_stack"].clear()

        override_map = {row["asset_id"]: int(row["group_id"]) for _, row in overrides.iterrows()}
        for gid in group_ids:
            for asset_id in groups.get(int(gid), []):
                override_map[asset_id] = target_id

        new_overrides = pd.DataFrame(
            [{"asset_id": k, "group_id": v} for k, v in override_map.items()]
        )
        context["overrides"] = write_overrides(new_overrides, config["overrides_path"])

    refresh_context()
    return jsonify({"status": "ok", "target_group_id": target_id})


@app.route("/api/group/remove", methods=["POST"])
def remove_from_group():
    payload = request.get_json(silent=True) or {}
    asset_id = payload.get("asset_id")
    if not asset_id:
        return jsonify({"error": "Missing asset_id"}), 400

    with context["lock"].gen_wlock():
        config = context["config"]
        overrides = context["overrides"]
        base_groups = context["base_groups"]

        context["undo_stack"].append(serialize_overrides(overrides))
        context["redo_stack"].clear()

        override_map = {row["asset_id"]: int(row["group_id"]) for _, row in overrides.iterrows()}
        override_map[asset_id] = next_singleton_id(base_groups, overrides)

        new_overrides = pd.DataFrame(
            [{"asset_id": k, "group_id": v} for k, v in override_map.items()]
        )
        context["overrides"] = write_overrides(new_overrides, config["overrides_path"])

    refresh_context()
    return jsonify({"status": "ok"})


@app.route("/api/group/undo", methods=["POST"])
def undo_edit():
    with context["lock"].gen_wlock():
        config = context["config"]
        if not context["undo_stack"]:
            return jsonify({"error": "No undo steps available"}), 400
        context["redo_stack"].append(serialize_overrides(context["overrides"]))
        prev_state = context["undo_stack"].pop()
        prev_df = pd.DataFrame(prev_state, columns=["asset_id", "group_id"])
        context["overrides"] = write_overrides(prev_df, config["overrides_path"])

    refresh_context()
    return jsonify({"status": "ok"})


@app.route("/api/group/redo", methods=["POST"])
def redo_edit():
    with context["lock"].gen_wlock():
        config = context["config"]
        if not context["redo_stack"]:
            return jsonify({"error": "No redo steps available"}), 400
        context["undo_stack"].append(serialize_overrides(context["overrides"]))
        next_state = context["redo_stack"].pop()
        next_df = pd.DataFrame(next_state, columns=["asset_id", "group_id"])
        context["overrides"] = write_overrides(next_df, config["overrides_path"])

    refresh_context()
    return jsonify({"status": "ok"})


@app.route("/media/<asset_id>")
def serve_media(asset_id: str):
    with context["lock"].gen_rlock():
        path = context["asset_lookup"].get(asset_id)
    if not path:
        return "Not found", 404
    return send_file(path, conditional=True)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--log-file", type=Path, default=PROJECT_ROOT / "logs" / "server.log"
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT,
        help="Base directory for data/cache/logs",
    )
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    ensure_data_layout(args.data_dir)
    context["config"] = load_config(args.data_dir)
    refresh_context()

    logger.info("Starting dupefinder server")
    app.run(debug=True, host="127.0.0.1", port=args.port)
