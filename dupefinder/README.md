# Dupefinder

**Status: implemented, not yet run against real data.** All 9 pipeline stages, the
`pipeline.py` orchestrator, unit/integration tests (against the fixtures in
`tests/videos/`), and the Flask review dashboard (`server/`) are in place. Design
questions from the review pass are resolved (see "Design decisions" at the bottom) — a
few numeric thresholds are starting estimates, to refine once the pipeline runs against
real data.

A local, offline pipeline that scans a directory of photos/videos and finds duplicate or
near-duplicate assets — including matches across resolution, aspect ratio/crop, file
format/codec, and (for video) trims/cuts of a longer source clip. Mirrors
`recognizer/`'s structure: a config-driven, idempotent, multi-stage batch pipeline
(`src/`) plus a Flask review dashboard (`server/`). Unlike `recognizer`, it never
modifies, moves, or deletes the source media it scans — see "Non-destructive by design"
below.

## Why duplicates are hard here

A naive "same file" check (byte hash) only catches exact copies. This tool needs to
survive:

- **Resolution changes** — same photo/video re-exported at a different size.
- **Aspect-ratio changes** — cropped, letterboxed, or padded versions.
- **Format differences** — JPEG vs PNG vs WebP, MP4 vs MOV vs MKV with different codecs.
- **Video cuts/trims** — a video that is a subclip (or superset) of another, possibly at
  a different frame rate or with a time offset.

No single signal handles all of these, so the pipeline computes multiple
resolution/format-invariant signals and fuses them — but unlike `recognizer` (which
computes every embedding modality for every entity unconditionally), this pipeline is
structured as a **cascade**: cheap, high-precision checks run first and resolve as many
duplicates as they confidently can; only assets/pairs left *unresolved* escalate to the
next, more expensive tier. A cheap signal with a low false-positive rate should obviate
the need for an expensive one, not just supplement it. This also bounds the one stage
that's inherently expensive (video cut/trim alignment) to a small shortlist instead of
the whole corpus.

## Proposed architecture (cascading tiers)

```
input_dir/ (your photos & videos)
    │
    ▼
 0. index_files            → cache/file_index.parquet
    │                         asset_id, path, type, size, mtime, dims, fps, duration,
    │                         format/codec, sha256 (full-file content hash)
    ▼
── Tier 0: exact (free) ──────────────────────────────────────────────────
 1. exact_duplicates       → data/duplicates/exact_groups.parquet
    │                         group by sha256: byte-identical files, 100% confidence.
    │                         Fully resolved — excluded from every later stage.
    ▼
── Tier 1: near-exact (cheap, low false-positive rate) ───────────────────
 2. coarse_signature       → data/hashes/hashes.parquet
    │                         data/embeddings/clip_coarse/<asset_id>.npy
    │                         Multiple perceptual hashes per image / per sparse video
    │                         frame sample (aHash, dHash, pHash — hand-rolled with
    │                         opencv+numpy, see "Perceptual hashing" below), plus a
    │                         coarse CLIP embedding (image, or mean-pooled over the same
    │                         sparse video frame sample, recognizer-style density —
    │                         cheap, corpus-wide). This tier only needs "plausibly the
    │                         same content," not precise temporal alignment, so it does
    │                         NOT need dense video sampling.
    ▼
 3. near_duplicates        → data/duplicates/near_dup_groups.parquet
    │                         duplicate if ANY of the hash variants is within its own
    │                         (small, strict) Hamming-distance threshold — an OR across
    │                         hash types, each individually low-false-positive. Resolved
    │                         without touching CLIP. Assets fully resolved here are
    │                         excluded from stage 4+.
    ▼
── Tier 2: semantic near-duplicate (moderate cost, corpus-wide but shrinking) ──
 4. build_candidates       → data/candidates/pairs.parquet
    │                         Brute-force cosine similarity (numpy, already a dependency
    │                         — no new ANN library for now) over remaining *unresolved*
    │                         assets' coarse CLIP signatures → candidate pairs
    │                         (image↔image, image↔video, video↔video).
    ▼
 5. match_pairs            → data/candidates/scored_pairs.parquet
    │                         coarse CLIP cosine similarity on remaining candidates.
    │                         Resolves crop/aspect-ratio/heavy-format-change duplicates
    │                         for images, image↔video "is this photo a frame from that
    │                         video" matches, and video↔video pairs whose coarse
    │                         (whole-video-averaged) signatures already agree closely
    │                         (likely full-duplicate videos). Only video↔video pairs
    │                         that are "plausibly related but not conclusively resolved"
    │                         escalate further.
    ▼
── Tier 3: video cut/trim alignment (expensive, but scoped to a shortlist) ─────
 6. match_video_pairs      → data/candidates/video_alignment.parquet
    │                         Only for the small number of video↔video candidate pairs
    │                         still unresolved after tier 2. For just those two videos,
    │                         densely re-sample frames (fixed time interval) and embed
    │                         them, then search the frame-to-frame similarity matrix for
    │                         a monotonic/diagonal high-similarity band (tolerating time
    │                         offset + frame-rate mismatch) → overlap_ratio + match_score.
    │                         Dense sampling only happens here, on-demand, cached per
    │                         asset (not per pair — a video appearing in several
    │                         escalated pairs is only ever densely resampled and
    │                         encoded once) — never corpus-wide.
    ▼
 7. cluster_duplicates     → data/duplicates/groups.parquet
    │                         union-find / connected components over every tier's
    │                         resolved groups/pairs (0, 1, 2, 3) → final duplicate groups
    ▼
 8. rank_group_members     → data/duplicates/ranked.parquet
                              within each group, rank members by a "keep" heuristic —
                              video duration bucket first (so a higher-resolution
                              subsection never outranks the full-length video), then
                              resolution, then bitrate/size, then mtime — to *suggest*
                              which copy to keep — informational only, see
                              "Non-destructive by design"
```

Every stage is idempotent (skips assets/pairs whose output already exists) and
independently runnable, exactly like `recognizer`'s stages — `src/pipeline.py` will
orchestrate them via a `--stages` flag and a shared `configs/config.json`. The cascade
means each tier's *input set* shrinks as earlier tiers resolve assets/pairs, so the
expensive tiers (CLIP, and especially dense video alignment) only ever run on what
cheaper tiers couldn't already confidently decide.

### Perceptual hashing (no new dependency)

Rather than pull in `imagehash`, hash variants are hand-rolled from `opencv` + `numpy`
(both already dependencies): average hash (aHash), difference hash (dHash), and a DCT
perceptual hash (pHash). Each is a 64-bit hash compared by Hamming distance. Tier 1
treats two assets as duplicates if **any single hash variant** falls within its own
strict threshold (≤ 2 bits out of 64 per variant — see "Threshold tuning" below) —
using multiple hash families lowers the false-negative rate (different hash types are
fooled by different kinds of edits) while keeping each individual threshold tight
enough to keep the false-positive rate low.

### Key components

- **`src/config.py`**, **`src/index_files.py`**, hashing/embedding helpers — written
  standalone in `dupefinder/src/`, *not* imported from `recognizer/src/` (decided: no
  code sharing between the two projects for now). Some logic will look similar
  (`recognizer/src/config.py`'s `load_config`/`get_section`/`resolve`/`resolve_path`
  pattern, `recognizer/src/index_files.py`'s file-walk shape) but is duplicated
  on purpose rather than factored into a shared package.
- **`src/coarse_signature.py`** — computes both cheap tier-1 signals in one pass per
  asset: the three hand-rolled hashes and a coarse CLIP embedding (image, or
  mean-pooled over a sparse video frame sample at recognizer-style density).
- **`src/near_duplicates.py`** — resolves tier-1 duplicates from hash agreement alone;
  assets resolved here never reach the CLIP/candidate stages. Candidate *generation*
  (which pairs to even Hamming-compare) uses LSH-style banding
  (`build_candidate_pairs`/`band_keys`) rather than all-pairs comparison — see
  "Performance" below.
- **`src/build_candidates.py`** — brute-force cosine-similarity candidate generation
  over coarse CLIP embeddings, run only over assets tier 1 left unresolved.
- **`src/match_pairs.py`** — coarse CLIP cosine similarity scoring for image/image,
  image/video, and whole-video/whole-video candidate pairs.
- **`src/match_video_pairs.py`** — the video-specific sequence-alignment matcher: for
  assets in a shortlisted pair, densely re-samples frames on demand and embeds them
  (cached per `asset_id` in `get_dense_embedding` — a video escalated in multiple
  pairs, e.g. via `build_candidates.top_k > 1`, is only extracted/encoded once), then
  searches the frame-similarity matrix for a best-aligned overlapping segment. The
  most novel piece — validate against `tests/videos/` (see "Tests") before building
  the rest of the pipeline around it.
- **`src/cluster_duplicates.py`** — union-find over every tier's resolved
  groups/pairs.
- **`src/rank_group_members.py`** — keeper-suggestion heuristic (suggestion only, never
  acted on automatically).
- **`src/pipeline.py`** — stage orchestration + `data/`, `cache/`, `logs/` layout, same
  pattern as `recognizer/src/pipeline.py`. `configure_stage_logging` reconfigures the
  root logger (every stage's logger propagates to it) before each stage runs, pointing
  it at that stage's own `log_file` from `configs/config.json` — same
  `logs/<stage>.log` file each stage writes to when run standalone, rather than one
  pipeline-wide file or the console. Writing to the console would garble tqdm's live
  progress bars with per-item `logger.info`/`logger.error` calls; tqdm writes straight
  to stderr and each stage's one-line completion `print()` goes to stdout, so both stay
  clean regardless of where the loggers point.
- **`server/`** — Flask dashboard for reviewing duplicate groups (see below).
- **`models/`** — symlinked to `recognizer/models` (`../recognizer/models`) so CLIP
  weights aren't downloaded twice.
- **`tests/`**, **`tests/videos/`** — unit tests for hash/similarity utilities, plus
  sample video fixtures (`sample1-5s-360p.mp4` / `sample1-5s-720p.mp4` — same source at
  two resolutions; `sample2-10s-vp9.mp4`; `sample3-20s-360p.mp4`;
  `sample4-30s-360p.mp4`) used as raw material for `ffmpeg`-generated trims/crops/
  transcodes to validate hash invariance and `match_video_pairs`' alignment search.

### Data flow / on-disk layout

```
cache/file_index.parquet
data/
  frames_sparse/<asset_id>/*.jpg                 # tier-1 coarse video frame sample
  frames_dense/<asset_id>/*.jpg                   # tier-3 on-demand dense re-sample,
                                                   # only for assets in an escalated pair
  hashes/hashes.parquet                           # asset_id[/frame] -> {ahash,dhash,phash}
  embeddings/clip_coarse/<asset_id>.npy           # image / mean-pooled sparse-video CLIP
  embeddings/clip_dense/<asset_id>.npy            # tier-3 dense per-frame CLIP embeddings
  candidates/pairs.parquet                        # candidate pairs (tier 2)
  candidates/scored_pairs.parquet                 # tier-2 fused pairwise duplicate scores
  candidates/video_alignment.parquet              # tier-3 alignment results
  duplicates/exact_groups.parquet                 # tier-0 sha256 exact-match groups
  duplicates/near_dup_groups.parquet              # tier-1 hash-resolved groups
  duplicates/groups.parquet                       # final duplicate groups (all tiers)
  duplicates/ranked.parquet                       # per-group keeper ranking (suggestion only)
  duplicates/group_overrides.parquet              # dashboard edits overlay (merge/remove)
logs/<stage>.log                                  # one file per stage, standalone or via pipeline.py
models/ -> ../recognizer/models                   # symlink, shared CLIP weights
```

`dupefinder` never reads from or writes to the original media files themselves beyond
indexing them (see below) — everything above is derived data living under
`dupefinder/{cache,data,logs}`, same isolation as `recognizer`.

### Non-destructive by design

The source media that `dupefinder` scans typically lives *outside* this repo entirely.
Because of that, this tool **never modifies, moves, or deletes original files** — not
even to a "staging" directory. Its entire output is an index: `duplicates/groups.parquet`
marks which assets belong to which duplicate group, and `duplicates/ranked.parquet`
records a suggested keeper per group. What to actually do with that information (delete,
archive, keep) is left to the person reviewing the dashboard, using their own tools —
this mirrors `recognizer/server/server.py`, which likewise only ever writes an overrides
overlay and never touches `path`-referenced originals.

### Web dashboard (`server/server.py`)

Deliberately simpler than `recognizer/server/server.py`: one view — a grid of
"buckets" (duplicate groups) — instead of a sidebar/detail/graph split, since the only
things a reviewer needs to do here are combine buckets and pull an item out of one.

Data model: `data/duplicates/groups.parquet` (the pipeline's output) is the base
assignment; `data/duplicates/group_overrides.parquet` is a same-shape overlay the
dashboard writes to and layers on top (override wins per `asset_id`) — the pipeline's
output on disk is never mutated directly, only shadowed, same pattern as
`recognizer/server/server.py`'s `assignments_overrides.parquet`. "Effective" group
membership (base + overrides, groups with fewer than 2 members dropped) is recomputed
on every load and edit; the suggested keeper per bucket is recomputed live from
`cache/file_index.parquet` (video duration bucket, then resolution, then bitrate/size,
then mtime — same heuristic as `rank_group_members.py`, kept in sync manually since
`dupefinder` doesn't share code between the pipeline and dashboard) rather than read
from `ranked.parquet`, so it stays correct after merges/removals instead of going
stale.

- `GET /api/groups?offset=&limit=&member_limit=&include_singletons=` — a *page* of
  buckets (default 24), sorted by size descending, each with only its first
  `member_limit` members (default 12) inline — not the whole library at once, which
  wouldn't scale to a large corpus. Response includes `total`/`has_more` for the
  bucket list and, per bucket, `has_more_members`; the dashboard uses two independent
  `IntersectionObserver`s — one sentinel at the bottom of the bucket grid that loads
  the next page of buckets, and one per bucket (only when `has_more_members`) that
  loads more of *that* bucket's members — so scrolling down loads more buckets, and
  scrolling within a large bucket loads more of its items, both automatically. Buckets
  that have shrunk to a single member (`is_singleton`) are hidden by default; pass
  `include_singletons=1` to include them — the dashboard's "Show singletons" toggle
  does this, so a removed item isn't just gone with no way back short of Undo. This
  also surfaces assets the pipeline never matched to anything at all — those don't
  have a row in `groups.parquet`/overrides, so the server synthesizes an ephemeral
  singleton entry (a negative, never-persisted `group_id`) for each on every refresh
  purely so they're browsable; merging one into a real bucket resolves it to a real,
  persisted id (see `merge_groups` in `server.py`). Without this, a genuinely unique
  file (nothing else in the library resembles it) would simply never appear anywhere
  in the dashboard, with no way to tell "the pipeline checked this and it's unique"
  apart from "this hasn't been processed yet."
- `GET /api/group/<id>/members?offset=&limit=` — the next page of a specific bucket's
  members, for the per-bucket lazy-load above. Members are ordered keeper-first, then
  descending quality (`rank_members` — the same canonical order `/api/groups`' preview
  uses), so paging through a large bucket never skips or repeats an item regardless of
  where the initial preview cut off.
- `POST /api/group/merge` — combine two or more selected buckets into the
  lowest-numbered one (writes overrides only).
- `POST /api/group/remove` — pull one asset out of its bucket by assigning it a fresh
  singleton group id (so it drops to a singleton bucket) — this is how "delete this
  from the bucket" works without ever touching the file itself.
- `POST /api/group/undo` / `redo` — step through the overrides history.
- `GET /media/<asset_id>` — read-only file serving; also what each thumbnail links to
  (opens in a new tab on click) for previews (images via `<img>`, videos via `<video>`).

There is intentionally no delete/move action anywhere in the dashboard — "remove from
bucket" only changes which bucket an asset is grouped into, never the file on disk.

## Configuration

`configs/config.json`, one section per stage, same override pattern as `recognizer`
(`config.resolve`: CLI flag > config value > default):

- `index_files` — input dir(s), content-hash on/off.
- `coarse_signature` — sparse video sample `max_frames`/`max_frame_rate` (recognizer-
  style, cheap), hash size, CLIP model/pretrained weights (reuse `ViT-B-32` / `openai`
  from `recognizer`), batch size.
- `near_duplicates` — per-hash-variant Hamming-distance thresholds (≤2/64 bits each,
  OR-combined across variants), `min_frame_match_ratio` for video↔video hash matching.
- `build_candidates` — `top_k` neighbors, `similarity_floor` for brute-force cosine
  search (the minimum coarse similarity for a pair to even become a candidate).
- `match_pairs` — coarse-CLIP similarity thresholds for image/image, image/video, and
  whole-video/whole-video resolves: `resolve_threshold` (≥ → resolved as duplicate at
  tier 2 outright) and `escalate_threshold` (≥ but below `resolve_threshold` → for
  video↔video pairs only, escalate to `match_video_pairs`; below it, drop).
- `match_video_pairs` — `dense_interval_sec` re-sample interval,
  `align_similarity_threshold` (per-frame CLIP similarity to count as a matching frame
  in the alignment search), `min_overlap_frames` / `min_overlap_ratio` (how much of the
  shorter video's frames must match to call the pair a duplicate).
- `rank_group_members` — keeper heuristic: `duration_epsilon_seconds` (videos within
  this many seconds of each other are treated as "the same length" for ranking
  purposes; a longer video always outranks a shorter one regardless of resolution,
  since a higher-resolution video that's only a subsection of another shouldn't be
  suggested as the keeper), then resolution, then bitrate/size, then mtime.

### Threshold tuning

The defaults in `configs/config.json` are tuned to be **selective** (favor missing a
real duplicate over flagging two different-but-similar assets as duplicates), based on
a concrete false positive found while testing: `tests/videos/sample3-20s-360p.mp4` and
`sample4-30s-360p.mp4` are two *different* views of the same park (same location,
different camera angle) — not duplicates — but were merged by an earlier, looser
threshold set. Tracing why: their whole-video coarse CLIP similarity was 0.905 (high,
because whole-frame CLIP embeddings mostly encode "what kind of scene is this," which
is forgiving of viewpoint changes), which escalated them to tier 3 under the old
`escalate_threshold` of 0.75; tier 3's old `align_similarity_threshold` of 0.85 was
still loose enough that 31 individual frame pairs across the two clips both looked like
"a park with cars and trees" and cleared it, producing a 0.775 overlap ratio — well
past the old `min_overlap_ratio` of 0.3.

The fix tightens every tier, not just the one that produced the visible symptom,
because a similarity score sitting just under one stage's threshold can still slip
through if the next stage's threshold is loose:

| Parameter | Old | New |
|---|---|---|
| `near_duplicates.{ahash,dhash,phash}_threshold` | 4 | 2 |
| `near_duplicates.min_frame_match_ratio` | 0.6 | 0.8 |
| `build_candidates.similarity_floor` | 0.75 | 0.85 |
| `match_pairs.resolve_threshold` | 0.95 | 0.97 |
| `match_pairs.escalate_threshold` | 0.75 | 0.85 |
| `match_video_pairs.align_similarity_threshold` | 0.85 | 0.95 |
| `match_video_pairs.min_overlap_frames` | 3 | 8 |
| `match_video_pairs.min_overlap_ratio` | 0.3 | 0.6 |

`match_pairs.escalate_threshold` was deliberately raised only moderately (not all the
way up to `resolve_threshold`) rather than tightened enough to exclude the
sample3/sample4 pair outright at tier 2 — tier 3 has the actual frame-level temporal
evidence needed to tell "same footage, cut differently" apart from "similar-looking but
different content," so it's the better place to enforce strictness for video pairs;
making tier 2 the sole gatekeeper would risk rejecting genuine trims whose *whole-video
averaged* similarity dips (e.g. a trim that only overlaps for a third of its length)
before they ever get a chance at tier 3's fine-grained check. These are still estimates
to refine after running against a larger, real photo/video library — not a formal
precision/recall calibration.

**Second round, from a real ~27k-asset run:** tier 2 escalated 25,550 of 59,736
candidate pairs (43%) to tier 3, with an estimated 6.6-hour runtime — not the "small
shortlist" tier 3 is supposed to get. Not a bug this time: a large personal video
library naturally has many videos whose whole-video coarse CLIP similarity sits above
0.9 (same rooms/people/settings across different, non-duplicate videos — the same
"CLIP encodes scene-type, not exact content" effect as the sample3/sample4 case, just
showing up at volume instead of as one outlier). Tightened further:

| Parameter | Was | Now |
|---|---|---|
| `build_candidates.top_k` | 10 | 5 |
| `match_pairs.escalate_threshold` | 0.85 | 0.93 |

Separately (a real bug, not a threshold issue): `match_video_pairs.py` was extracting
and CLIP-encoding dense frames **per pair** (`frames_dense/<pair_id>/{a,b}`) rather than
per asset — a video escalated in several pairs (common once `top_k > 1`) was redone
from scratch for every pair it appeared in. Fixed to cache by `asset_id`
(`get_dense_embedding`, `frames_dense/<asset_id>/`, `embeddings/clip_dense/<asset_id>.npy`)
— pure performance fix, doesn't change which pairs get checked or the result.

## Setup

Uses the same Conda environment file as `recognizer`
(`cp ../recognizer/requirements.txt .`, `conda env create -f requirements.txt -n
dupefinder-env`) for consistency, but the actual code deliberately uses a much smaller
subset of it than `recognizer` does — no `ultralytics` (YOLO), `insightface`,
`torchreid`, `onnx`/`onnxruntime`, `yolox`, or the augmentation/tracking stack. In
practice this pipeline only needs: `torch` + `open_clip` (CLIP embeddings), `opencv`
(frame extraction, hand-rolled hashing), `numpy`/`pandas`/`pyarrow` (candidate search,
data storage), `flask` + `readerwriterlock` (dashboard), `tqdm`. No ANN library
(`faiss`) or perceptual-hash library (`imagehash`) is planned for now — see "Design
decisions."

## Running

```bash
python src/pipeline.py --config configs/config.json --input-dir /path/to/media --data-dir .
python src/pipeline.py --input-dir /path/to/media --data-dir . --stages coarse_signature near_duplicates
python server/server.py --data-dir . --port 5034
```

## Resetting data

`clean.sh` wipes the file index, all logs, and everything under `data/` (detections,
hashes, embeddings, candidates, duplicate groups — including any manual dashboard edits
in `group_overrides.parquet`) after a confirmation prompt. It does **not** touch
`configs/`, `tests/`, or `models/` (same behavior as `recognizer/clean.sh`):

```bash
bash clean.sh
```

## Performance, relative to `recognizer`

Cheaper overall, mainly because this pipeline never runs `recognizer`'s detection models
(YOLO person detection, InsightFace face detection) or its extra embedding models (face,
body re-ID) — it only needs one embedding model (CLIP), and only for content that cheaper
tiers couldn't already resolve. Concretely:

- Tier 0 (`sha256`) and tier 1 (hashes) are non-ML and near-free, and are expected to
  resolve a large fraction of real-world duplicates (re-saves, re-encodes, resizes)
  outright — those assets never reach CLIP or candidate search at all. Being non-ML
  doesn't mean free of scaling problems, though: an initial all-pairs comparison loop
  in tier 1 took ~2 hours over 241M pairs on a real ~22k-asset run. `near_duplicates.py`
  now generates candidates with LSH-style banding (`build_candidate_pairs`) instead —
  each hash is split into `threshold + 1` disjoint bands, and only assets that share at
  least one band get Hamming-compared. By pigeonhole this has zero recall loss versus
  the all-pairs scan (proven by a property test in
  `tests/test_near_duplicates_logic.py`) while pruning away the vast majority of
  unrelated pairs before ever computing a Hamming distance.
- Tier 2 (`CLIP`) only runs on what's left, and only computes one embedding per
  image/coarse-video-sample — versus `recognizer` computing three embedding models
  (face/body/CLIP) per tracked entity per frame, on top of two detection models per
  frame.
- Tier 3 (dense video alignment) is the one stage that's inherently expensive
  (O(frames_A × frames_B) per pair, with dense on-demand resampling), but the cascade
  keeps its input to a small shortlist of still-ambiguous video pairs rather than the
  whole corpus.

## Tests

```bash
python -m unittest discover tests
```

- Hash/embedding invariance under resize/crop/format-change — using `ffmpeg` to derive
  resized/cropped/transcoded variants of `tests/videos/*.mp4` and asserting the hash/
  CLIP-similarity signals still call them duplicates.
- The video sequence-alignment search — `tests/videos/sample1-5s-360p.mp4` vs.
  `sample1-5s-720p.mp4` (known same-source, different resolution) as a baseline, plus
  `ffmpeg`-trimmed sub-clips of `sample3-20s-360p.mp4` / `sample4-30s-360p.mp4` with
  known offsets/durations to validate `match_video_pairs` recovers the correct overlap
  window.

---

## Design decisions

Resolved from the first review pass:

1. **Code sharing with `recognizer`.** Decided: no shared package for now.
   `config.py`, indexing, and CLIP embedding logic are written standalone in
   `dupefinder/src/`, duplicated rather than imported.
2. **Video cut/trim matching algorithm.** Decided: prototype/validate using the fixture
   videos in `tests/videos/` (ffmpeg-derived trims/crops/transcodes with known ground
   truth) before building the rest of the pipeline around it.
3. **Tier-1 hash threshold strictness.** Decided: start with a reasonable estimate
   (≤4/64 bits per hash variant), refine after running the pipeline on real data.
4. **ANN library.** Decided: either is fine; defaulting to brute-force numpy/sklearn
   cosine similarity for now since it's already a dependency and keeps the library
   footprint smaller (per the minimal-dependency goal below) — can swap to `faiss-cpu`
   later if corpus size makes candidate search a bottleneck.
5. **Perceptual hash library/algorithm.** Decided: use multiple hash variants, each
   with its own small/strict threshold, and mark a pair as a tier-1 duplicate if *any*
   variant matches. Implemented by hand with `opencv`/`numpy` rather than adding
   `imagehash` as a dependency.
6. **Tier-2→3 escalation threshold for video pairs.** Decided: start with a reasonable
   estimate (cosine ≥0.95 resolves at tier 2, 0.75–0.95 escalates to tier 3, <0.75
   dropped), refine after running on real data.
7. **Model weight sharing.** Decided: `dupefinder/models` is a symlink to
   `../recognizer/models` (created; already covered by the existing `models/` entry in
   `.gitignore`).
8. **Destructive actions in the dashboard.** Decided: none at all, not even a
   non-destructive staging move — source media lives outside this repo and is treated
   as read-only. `dupefinder` only ever maintains an index/marking of duplicates
   (`duplicates/groups.parquet`, `duplicates/ranked.parquet`); the dashboard is
   view/annotate-only. See "Non-destructive by design" above.
