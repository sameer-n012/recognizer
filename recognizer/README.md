# Recognizer

A local, offline pipeline that scans a directory of photos/videos, detects people, and
clusters them into identities ("entities" → "clusters") using a fusion of face, body
(re-ID), and CLIP embeddings. Ships with a Flask dashboard for browsing and manually
correcting the resulting clusters.

Nothing leaves the machine — all detection/embedding models run locally (CPU or Apple
Silicon MPS).

## Architecture

The system is an 11-stage batch pipeline (`src/pipeline.py`) that turns raw media into
clustered "person" identities, plus a Flask web app (`server/`) for reviewing/correcting
the output.

```
input_dir/ (your photos & videos)
    │
    ▼
 0. index_files        → cache/file_index.parquet   (asset_id, path, type, dims, fps, duration)
    │
    ▼
 1. extract_frames     → data/frames/<asset_id>/*.jpg   (videos only, ffmpeg, adaptive sampling)
    │
    ▼
 2. person_detection   → data/detections/persons/<asset_id>.json   (YOLOv8s, person class only)
    │
    ▼
 3. face_detection     → data/detections/faces/<asset_id>.json     (InsightFace, run inside person crops)
    │
    ▼
 4. build_entities     → data/entities/<entity_id>.json   (ByteTrack links person detections
    │                     across video frames into one "entity" per tracked person;
    │                     each image detection becomes its own entity)
    ▼
 5. embed_faces        → data/embeddings/face/<entity_id>.npy   (InsightFace face embedding, mean-pooled)
 6. embed_bodies       → data/embeddings/body/<entity_id>.npy   (torchreid osnet_x1_0 re-ID embedding)
 7. embed_clip         → data/embeddings/clip/<entity_id>.npy   (open_clip ViT-B-32 embedding)
    │
    ▼
 8. fuse_embeddings    → data/embeddings/fused/<entity_id>.npy
    │                     weighted sum of face/body/clip, L2-normalized. Weights depend on
    │                     what's available per entity (face_present / no_face / no_person).
    ▼
 9. cluster_entities   → data/clusters/{assignments.parquet, centroids.npy}
    │                     HDBSCAN over fused embeddings (cosine), with oversized clusters
    │                     re-split via agglomerative clustering.
    ▼
10. refine_clusters    → data/clusters/refined/{assignments_refined.parquet, merges.json}
                          heuristic cleanup pass: merge near-duplicate clusters, reassign
                          noise points, merge/force-merge tiny clusters, split oversized
                          clusters, reassign singletons.
```

Each stage is idempotent — it skips assets/entities whose output file already exists, so
the pipeline can be re-run incrementally after adding new media (see `--stages` below).

### Key components

- **`src/pipeline.py`** — orchestrates all stages in order; also defines `DATA_LAYOUT`,
  the expected `data/`, `cache/`, `logs/` directory tree, and creates it on first run.
- **`src/config.py`** — tiny helpers (`load_config`, `get_section`, `resolve`,
  `resolve_path`) shared by every stage script for reading `configs/config.json` and
  layering CLI overrides on top of it.
- **`src/build_entities.py`** — the identity-tracking step. Uses `ByteTrack`
  (vendored in `ByteTrack/`, imported as the `yolox` package) to associate YOLO person
  detections across video frames into stable tracks, then matches face detections back
  onto tracked person boxes by IoU.
- **`src/fuse_embeddings.py`** — combines the three embedding modalities with
  different weightings depending on whether a face was detected, only a body, or
  neither (CLIP-only fallback for entities with unusable face/body crops).
- **`src/cluster_entites.py`** / **`src/refine_clusters.py`** — two-pass clustering:
  an initial HDBSCAN pass, then a rule-based refinement pass that fixes common HDBSCAN
  failure modes (fragmented near-duplicate clusters, noise points, oversized clusters).
- **`ByteTrack/`** — vendored upstream [ByteTrack](https://github.com/ifzhang/ByteTrack)
  repo; only `yolox.tracker.byte_tracker.BYTETracker` is used by this project.
- **`server/`** — Flask dashboard (see below).
- **`tests/`** — unit tests for bbox utilities (IoU, clamping, padding) used in face
  detection/embedding.

Every stage script is also independently runnable from the CLI (`python src/<stage>.py
--config ... --index ... `), with config values as defaults and CLI flags as overrides
(see `config.resolve`).

### Data flow / on-disk layout

All pipeline output lives under a `--data-dir` (defaults documented in
`configs/config.json`, and typically the `recognizer/` root):

```
cache/file_index.parquet         # master asset index
data/
  raw/{images,videos}/           # (optional) place to keep originals; not populated by the pipeline
  frames/<asset_id>/*.jpg        # extracted video frames
  detections/{persons,faces}/    # per-asset YOLO / InsightFace detections (JSON)
  entities/<entity_id>.json      # tracked people (one JSON per entity)
  embeddings/{face,body,clip,fused}/<entity_id>.npy
  clusters/{assignments.parquet,centroids.npy}
  clusters/refined/{assignments_refined.parquet,merges.json,assignments_overrides.parquet}
logs/<stage>.log                 # one log file per stage
models/                          # local model weights (yolov8s.pt, insightface/buffalo_l/*)
```

### Web dashboard (`server/server.py`)

A Flask app that loads `data/clusters/refined/assignments_refined.parquet` +
`data/entities/*.json` + all embedding modalities into an in-memory `context` dict
(guarded by a `readerwriterlock`), and serves:

- `GET /dashboard` — cluster overview UI (`server/templates/index.html` +
  `server/static/app.js`).
- `GET /dashboard/graph` — 2D PCA projection of cluster centroids for visual exploration
  (`server/templates/graph.html` + `server/static/graph.js`).
- `GET /api/clusters`, `GET /api/cluster/<id>`, `GET /api/cluster/<id>/suggestions`,
  `GET /api/cluster_graph` — read APIs backing the UI.
- `POST /api/cluster/edit` — manual corrections (`move_entities`, `merge_selected`,
  `merge_clusters`, `split_cluster`); writes to an `assignments_overrides.parquet`
  overlay (not the original assignments) and supports `undo`/`redo` via in-memory stacks.
- `GET /media/<asset_id>` — serves the original image/video file for preview.

Per-modality similarity scores shown in the UI are computed against each cluster's
centroid using `configs/config.json`'s `similarity` block (`metric` + `transform`, e.g.
cosine distance → reciprocal similarity).

## Configuration

All stage parameters live in `configs/config.json`, one top-level key per stage (plus a
shared `paths` block). Every value can be overridden per-invocation via CLI flags on the
individual stage script; `pipeline.py` itself only reads `--config`, `--stages`,
`--input-dir`, `--data-dir`.

Notable knobs:

- `person_detection.confidence_threshold`, `model_path` — YOLOv8s person detector.
- `face_detection` / `embed_faces` — InsightFace model name (`buffalo_l`), ONNX
  execution providers, detection size, confidence thresholds.
- `build_entities.tracker` — ByteTrack params (`track_thresh`, `match_thresh`,
  `track_buffer`, `frame_rate`, `mot20`).
- `fuse_embeddings.{face_weights,no_face_weights,no_person_weights}` — modality
  weighting per entity condition.
- `cluster_entities.similarity.hdbscan` / `.cluster_split` — HDBSCAN params and the
  agglomerative re-split threshold for oversized clusters.
- `refine_clusters.*` — merge/noise/singleton thresholds for the refinement pass.

`server/config.json` is separate and only configures the dashboard (paths to
assignments/entities/embeddings/overrides + the `similarity` display config).

## Setup

Dependencies are managed via a Conda environment (`recognizer-env`) — `requirements.txt`
in this directory is actually a `conda env export`, not a pip requirements file:

```bash
conda env create -f requirements.txt -n recognizer-env
conda activate recognizer-env
```

Requires `ffmpeg`/`ffprobe` on `PATH` for video indexing and frame extraction. Model
weights are expected under `models/` (`yolov8s.pt`, `insightface/models/buffalo_l/`) —
InsightFace will auto-download `buffalo_l` into `insightface_root` if missing.

## Running the pipeline

Run all stages against an input directory of media:

```bash
python src/pipeline.py \
  --config configs/config.json \
  --input-dir /path/to/photos_and_videos \
  --data-dir .
```

Run a subset of stages (by name or numeric index, see `STAGE_MAP` in `pipeline.py`),
useful when re-running after a config change downstream of an earlier stage:

```bash
python src/pipeline.py --input-dir /path/to/media --data-dir . \
  --stages cluster_entities refine_clusters
# or equivalently:
python src/pipeline.py --input-dir /path/to/media --data-dir . --stages 9 10
```

Stage order: `index → extract_frames → person_detection → face_detection →
build_entities → embed_faces → embed_bodies → embed_clip → fuse_embeddings →
cluster_entities → refine_clusters`.

Each stage can also be invoked standalone, e.g.:

```bash
python src/index_files.py --config configs/config.json --input-dir /path/to/media
python src/cluster_entites.py --config configs/config.json
```

Logs for each stage are written to `logs/<stage>.log`.

## Running the dashboard

```bash
python server/server.py --data-dir . --port 5033
```

Then open `http://127.0.0.1:5033/dashboard`. `--data-dir` should point at the same
directory used for `--data-dir` during pipeline runs (defaults to the `recognizer/`
project root).

## Resetting data

`clean.sh` wipes all generated data (detections, embeddings, clusters, entities, frames,
and raw copies) after a confirmation prompt. It does **not** touch `cache/`,
`configs/`, or `models/`:

```bash
./clean.sh
```

## Tests

```bash
python -m unittest discover tests
```

Currently covers bbox utilities (IoU, clamping with padding, face/track IoU matching)
used in `face_detection.py` and `embed_faces.py`.
