# Drone detector

This folder holds the detector: code that turns rendered video into training tiles, trains a small detector on them, and tests it on real video filmed from one drone of another. The detector is trained only on rendered video and never sees a real pixel in training.

## How it works

1. **Input.** Each video frame is cut into 640 px tiles that overlap by 128 px (32 tiles for a 4K frame). A tile has five channels: the current frame in colour, plus the gray frames from 2 and 4 frames earlier.
2. **Alignment.** The earlier frames are moved so that their background lines up with the current frame: first one shift for the whole frame, then one per tile. A moving drone then stands out where the frames differ.
3. **Detection.** A YOLO26n network (Ultralytics) proposes drone boxes on every tile. It is trained from random weights.
4. **Tracking.** The most confident box above a threshold goes to one Kalman tracker. A track starts after 4 hits in a row (`--hits 4`) and coasts up to 0.35 s without one.
5. **Scoring.** Each tracked box is compared with the labels of the test video.

[How we test](docs/how-we-test.md) explains the test sets and what is counted. [Experiments](docs/experiments.md) lists every change we tried and what it did. [Data](docs/data.md) says where the training and test data come from. Terms are in the [glossary](docs/glossary.md).

## Results so far

The current model is called `am_nobirds_horizon` (since 2026-11-07). It trains on seven rendered scenes: the four of E1, two of our own scenes with grey drones (with the scene's bird flock removed) and one horizon view, with pasted moving cars, downscaled copies for small drones and the stand-out filter. Below are medians over 10 training seeds at confidence 0.25 with 4 hits, on test videos used during development. A tracked frame has the box within 20 px of the drone, a false box is more than 50 px away. On our own flights the labels are boxes that Segment Anything 2.1 drew by following the drone; on the 17% of frames without an accepted box the label is projected from the flight logs and read at 50 and 75 px.

| Test set | Drone frames | Tracked | False boxes | Frames with no drone that still show a box |
| --- | --- | --- | --- | --- |
| ARD100, 26 windows | 3,799 | 2,256 | 505 | 229 of 361 |
| NPS, 50 windows | 7,444 | 2,668 | 442 | none labelled |
| UAV-Anti-UAV, 274 windows | 31,586 | 3,494 | 1,762 | 366 of 8,490 |
| Our 4K flights, 42 windows | 5,906 | 5,356 | 106 | 458 of 640 |

The standard single-frame YOLO26n trained on the colour channels of the same tiles tracks almost nothing (our flights about 15 frames at 0.25, ARD100 157 at equal false boxes), so the motion channels carry the transfer from rendered to real video. The model still misses most hovering drones and tracks moving cars and people. The [experiments page](docs/experiments.md) has every round. A new training set, rendered with real-size drones along planned flight scenarios, is being tested to replace these scenes for the final model.

## Setup

Python 3.12 and an NVIDIA GPU with CUDA 12.8. From this folder:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests
```

## Running

All commands run from this folder.

**1. Build training tiles** from the rendered scenes (the team's dataset pipeline produces them; each scene folder holds the video and one `drone*_flight.csv` per drone):

```bash
python -m drone_model.build_tiles --renders /path/to/renders --out tiles/e1
```

The defaults build E1's training set: scenes `centras1`, `ktuxi2`, `eiguliai1` and `amaliai1`, 3,075 tiles and 4.1 GB, built in about 5 minutes. The current model's set (10,591 tiles) is built with

```bash
python -m drone_model.build_tiles --renders /path/to/renders --out tiles/am_nobirds_horizon --cars /path/to/car_patches --scale-copies \
    --scenes centras1 ktuxi2 eiguliai1 amaliai1 r3_move_grey r3_far_grey r7_horizon_grey \
    --standout-scenes eiguliai1 amaliai1 r3_move_grey r3_far_grey r7_horizon_grey
```

where `r3_move_grey` and `r3_far_grey` are the renders without the bird flock. Options add tiles without changing the others: `--cars FOLDER` pastes moving car patches cut from the renders, `--cars-control` adds the same extra tiles without the cars, `--scale-copies` adds tiles cut from frames downscaled by 2 or 3, for small drones (`--scale-factors` and `--copies-per-frame` change them), `--standout-scenes` keeps a drone tile only when the drone stands out in the motion channels, `--keep-every SCENE=K` keeps every K-th sampled frame of a scene, `--past-gap N` uses frame f-N as the oldest past frame and `--text-overlays` adds screen-fixed text as distractors (`python -m drone_model.build_tiles --help`). `python -m drone_model.colour_tiles TILES OUT` makes the colour-only copy of a tile set for the standard YOLO26n.

**2. Train** one model per seed, about 40 minutes each on an RTX 4070:

```bash
python -m drone_model.train --data tiles/e1/dataset.yaml --out runs --name e1_seed23 --seed 23
```

The network is YOLO26n. `--model yolo26s.yaml` trains the larger YOLO26s instead.

**3. Make the test plans.** `windows/` lists where each of our test windows lies and how it was chosen, but holds no labels, because they belong to each dataset's publisher. Download the dataset, then rebuild the plan with its labels:

```bash
python -m drone_model.make_plans --set ard100 --videos /path/to/ARD100/test_videos --labels /path/to/annotations --out plans/ard100.json
```

The sets are `ard100`, `nps`, `uav_anti_uav` and `local4k` (our own flights, not public). Each file in `windows/` says which folders `--videos` and `--labels` must point to. The script stops if your label files differ from the ones our results used, and the evaluation checks the videos the same way.

**4. Evaluate** on a plan:

```bash
python -m drone_model.evaluate --weights runs/e1_seed23/weights/last.pt --windows plans/ard100.json --out results/ard100_e1 --hits 4
```

The default is 3 hits, which E1's numbers used; every result since 2026-11-05 uses 4. `--channels 3` scores a colour-only model, `--alignment homography` uses a per-tile homography instead of the per-tile shift (on the CPU, about 3 frames per second at 4K), `--past-gap N` matches a model trained with it and `--upscale F` enlarges every frame before tiling. It writes `summary.csv` with one row per model and threshold, per-window counts in `metrics/`, and every proposal in `proposals/`, so other thresholds can be scored later without the GPU.

**Stopping and slowing down.** Training and evaluation can be stopped at any time and continued by running the same command with `--resume`. Training loses at most the epoch in progress. `--gpu-share-file FILE` takes a file holding a number from 1 to 100, the percent of normal pace. The file is reread every 5 s, so editing it slows a running job down or speeds it up.

## What has been checked

- The 50 unit tests pass.
- This code reproduces the research code it was cleaned from: identical training tiles, identical model inputs on 4K, 1920 × 1080 and 856 × 480 frames, identical proposals from the same weights, identical training weights after 2 short epochs (all 708 tensors of the saved checkpoint), and identical tracker and scorer results on all 7,056 saved E1 window replays. The `--cars` and `--scale-copies` tile sets are byte-identical to the research builder's.
- The current model's tile set (`am_nobirds_horizon`, 10,591 tiles) and its colour-only copy rebuild byte for byte from the same renders with the command above and `colour_tiles` (21,182 files each, checked 2026-11-19).
- `make_plans` rebuilds all four test plans (392 windows, 58,751 frames) with exactly the labels our results were scored on.
- Not measured: speed on a drone's onboard computer. The numbers above are from a desktop GPU.

## Layout

| Path | What it is |
| --- | --- |
| `drone_model/build_tiles.py` | Rendered scenes to 5-channel training tiles |
| `drone_model/train.py` | Training from random weights |
| `drone_model/evaluate.py` | Detection, tracking and scoring on labelled video windows |
| `drone_model/make_plans.py` | Test plans with labels, rebuilt from each dataset's own files |
| `drone_model/colour_tiles.py` | Colour-only copy of a tile set, for the standard YOLO26n |
| `drone_model/inputs.py` | Tiles, model inputs and tiled detection |
| `drone_model/alignment.py` | Whole-frame and per-tile alignment, on CPU or GPU |
| `drone_model/tracking.py` | The one-target tracker |
| `drone_model/scoring.py` | Comparing tracked boxes with labels |
| `drone_model/gpu_share.py` | Slowing a running job down |
| `configs/train.yaml` | Training settings (unchanged since E1) |
| `windows/` | Where each test window lies and how it was chosen, without labels |
| `tests/` | Unit tests |
| `docs/` | How we test, data, experiments, glossary |

No videos, tiles, labels or trained weights are stored here.

## Licence

GNU Affero General Public License, version 3 or (at your option) any later version (AGPL-3.0-or-later). See [LICENSE](../LICENSE). Ultralytics, which the detector is built on, uses the same licence, and Ultralytics states that models trained with its code fall under it too.
