# Glossary

| Term | Meaning |
| --- | --- |
| Tile | A 640 × 640 px crop of a frame. A 4K frame is covered by 32 tiles that overlap by 128 px. |
| Motion image | The difference between the current frame and an earlier frame after alignment. A moving drone shows up as a bright spot in it. |
| Alignment | Moving an earlier frame so its background lines up with the current one. |
| Per-tile shift | Today's alignment: one shift up/down and left/right per tile, measured by phase correlation. |
| Whole-frame shift | One shift for the whole frame, applied before the per-tile shift at test time, so fast camera motion still aligns. |
| Homography | A warp with 8 numbers that also undoes rotation, zoom and perspective of a flat surface. Fitted per tile from tracked corner points. |
| Flow warp | A homography followed by moving every pixel along a dense motion estimate, so the background, including trees and buildings, cancels. |
| Stand out | A drone stands out when its brightest motion pixel is brighter than 99.9% of the tile's background motion. |
| E1 | The current base model: four rendered scenes, only standing-out drone tiles from `eiguliai1` and `amaliai1`, `amaliai1`'s drone-free tiles left out, per-tile shift in training, whole-frame plus per-tile shift at test time. |
| Round | One planned experiment with several models trained and scored together. Its question and pass rule are written down before any result. |
| Arm | One of the models in a round. |
| Seed | The random start of a training run. Each arm is trained with 3 seeds; results are medians with their range. |
| Confidence threshold | The detector score a box needs to be passed to the tracker: 0.5, 0.25 or 0.1. |
| Tracked frame | A frame whose tracked box is within 20 px of the labelled drone (50 px on our own flights). |
| False box | A tracked box more than 50 px from the drone (75 px on our own flights). |
| Correct first start | A test window whose first track started on the drone. |
| Equal false boxes | Comparing two models at the threshold where they make the same number of false boxes, read between the scored thresholds. |
| Development data | Test videos already used while developing the method. Results on them can be optimistic. |
| Reserve | Test videos kept untouched for one final test. |
| Hovering drone | A drone that moves less than 0.5 px per frame against the background. |
