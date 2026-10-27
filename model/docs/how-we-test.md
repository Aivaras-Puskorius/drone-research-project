# How we test

Every model is trained only on rendered video and tested on real air-to-air video it never saw. A test replays whole video windows frame by frame, the way the detector would run on a drone, and counts how often the tracked box sits on the real drone and how often it sits on something else.

## Test data

None of these videos is used for training. The numbers are what the current development tests contain.

| Test set | What it is | Windows | Drone frames | Frames with no drone | Labels |
| --- | --- | --- | --- | --- | --- |
| Our 4K flights | Two of our own drones, one filming the other from above, 3840 × 2160 | 42 | 5,906 | 640 | Projected from both drones' flight logs, median error about 30 px |
| ARD100 | Public drone-to-drone video, drones mostly under 20 px | 26 | 3,799 | 361 | Drawn by the dataset's annotators |
| UAV-Anti-UAV | Public video filmed from one drone of another, mostly 856 × 480 | 274 | 31,586 | 8,490 | Drawn by the dataset's annotators, with "absent" flags |
| NPS | Public video of fixed-wing drones, mostly seen from above | 50 | 7,444 | 0 | Drawn by the dataset's annotators |

16 ARD100 videos and 224 UAV-Anti-UAV videos are kept untouched for one final test, run once the method is frozen. Every number in this repository so far is on development data, which has shaped our choices. How each set's windows were chosen is written in its file in [`windows/`](../windows).

## What one test run does

1. **Replay.** A window of consecutive frames is played in order. At each frame the detector sees only the current frame and two earlier ones (2 and 4 frames back), never the future.
2. **Align.** Each earlier frame is shifted so its background lines up with the current frame: first one shift for the whole frame, then one per 640 px tile. A moving drone then stands out in the difference between the frames.
3. **Detect.** The frame is cut into 640 px tiles with 128 px overlap (32 tiles in a 4K frame). The detector gets five channels per tile: the current frame in colour plus the two aligned earlier frames in gray. It proposes boxes with a confidence score.
4. **Track.** The highest-scoring proposal above a confidence threshold goes to one tracker, the same for every model. It starts a track after 3 hits in a row and keeps it for up to 0.35 s without a hit.
5. **Score.** Each frame's tracked box is compared with the label.

## What we count

| Measure | Meaning |
| --- | --- |
| Tracked frames | Frames where the tracked box is within 20 px of the labelled drone (50 px on our flights, because their labels come from flight logs) |
| False boxes | Tracked boxes more than 50 px from the drone (75 px on our flights) |
| Boxes with no drone in view | Frames labelled empty or absent that still show a box |
| Correct first starts | Windows where the first track started on the drone |

A lower confidence threshold finds more drones and makes more false boxes, so every model is scored at 0.5, 0.25 and 0.1 and compared both at the same threshold and at equal numbers of false boxes. Each model is trained with 3 random seeds; we report the median and the range.

## Accuracy by drone speed and size

From the current base model, E1, at confidence 0.25. The bins are measured from the video and the labels only.

| Drone speed against the background, px per frame | under 0.5 | 0.5–2 | 2–5 | 5 and up |
| --- | --- | --- | --- | --- |
| ARD100, frames tracked | 11% | 25% | 64% | 60% |
| UAV-Anti-UAV, frames tracked | 9% | 10% | 18% | 15% |

| Drone size, px | under 10 | 10–20 | 20–40 | 40 and up |
| --- | --- | --- | --- | --- |
| ARD100, frames tracked | 6% | 40% | 70% | 42% |
| UAV-Anti-UAV, frames tracked | 2% | 8% | 21% | 13% |

Camera motion matters less: on ARD100 E1 tracks 38–44% of frames at low and medium camera speeds and 67% at the fastest, and on UAV-Anti-UAV 10–15% throughout. Hovering and tiny drones are the clearest weaknesses, because the detector relies on motion.

## Not tested yet

- **Controlled scenarios.** In real footage speed, size, distance, camera motion and background change together, so each bin above mixes causes. Rendered scenes know every drone's exact speed, height, distance and viewing angle, so test scenes where one factor changes at a time can be built from them. That includes the drone's own rotation and the viewing angle, which no test measures yet.
- **Speed on the drone.** On a desktop RTX 4070 the detector runs at about 12 frames per second on 4K video. No board for the drone has been measured.
- **The final test** on the untouched videos.
