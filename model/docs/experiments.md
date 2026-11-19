# Experiment log

Every experiment of the drone detector, newest first, in plain words. Numbers are medians over training seeds on development data. "Tracked" counts frames whose tracked box is within 20 px of the labelled drone (50 px on our own flights' projected labels), "false" counts tracked boxes more than 50 px away (75 px there). Our flights' numbers before 2026-11-19 use the projected labels. "At equal false boxes" compares tracked frames interpolated to the same number of false boxes; "the rule" is the gain rule in force at the time (since 2026-11-01: clearly better, p < 0.05 over 10 seeds, on our flights or on two public sets, and clearly worse on no set except NPS). See [how we test](how-we-test.md) for the full protocol and [the glossary](glossary.md) for terms.

| Dates | Experiment | Outcome |
| --- | --- | --- |
| 2026-11-16 to 2026-11-19 | [Labels for our own flights from Segment Anything 2.1](#2026-11-19-sam-labels) | Finding |
| 2026-11-10 to 2026-11-11 | [The standard single-frame YOLO26n (round AQ)](#2026-11-11-round-aq) | Reference |
| 2026-11-08 to 2026-11-11 | [Birds back, clouds, and a second horizon view (rounds AN, AO, AP)](#2026-11-08-rounds-an-ap) | Dropped |
| 2026-11-05 to 2026-11-07 | [A horizon scene and bird-free scenes: the current model (rounds AK, AL, AM)](#2026-11-05-rounds-ak-am) | Kept |
| 2026-11-03 to 2026-11-05 | [Distractors, mosaic, a P2 head, scaling and rotation (rounds AE to AI)](#2026-11-03-rounds-ae-ai) | Dropped |
| 2026-11-05 | [Four hits to start a track (round AJ)](#2026-11-05-round-aj) | Kept |
| 2026-11-01 to 2026-11-03 | [Test-time changes: an older past frame, homography, upscaling, an overlay filter (rounds V, W, X, AB, AC, AD)](#2026-11-01-test-time) | Dropped |
| 2026-10-31 to 2026-11-03 | [Our own render scenes: drone colour, hovering, distance, a second terrain (rounds N, N2, S, T, T2, U, Y, Z, AA)](#2026-10-31-own-renders) | Kept |
| 2026-10-31 | [Rendered moving cars (round Q)](#2026-10-31-round-q) | Dropped |
| 2026-10-31 | [A tracker gated to its own track (round P)](#2026-10-31-round-p) | Dropped |
| 2026-10-30 to 2026-10-31 | [Newer teammate scenes, more downscaled copies, longer training, a bigger network (rounds K, L, M, O)](#2026-10-30-rounds-k-o) | Dropped |
| 2026-10-29 to 2026-10-31 | [Pasted cars and downscaled copies (round J)](#2026-10-29-round-j) | Kept |
| 2026-10-27 | [Two new scenes, motion dropout, and the homography at test time only (round I)](#2026-10-27-round-i) | Lead |
| 2026-10-26 | [Training on homography- and flow-aligned tiles (round H)](#2026-10-26-round-h) | Dropped |
| 2026-10-26 | [Measuring better alignments and a hover-holding tracker (round G)](#2026-10-26-round-g) | Finding |
| 2026-10-25 to 2026-10-26 | [Pasted moving cars (round F)](#2026-10-25-round-f) | Option |
| 2026-10-23 to 2026-10-25 | [Degraded renders, and the test on our own 4K flights (round D)](#2026-10-23-round-d) | Dropped |
| 2026-10-24 | [The E1 recipe, and tiles built after the whole-frame shift (round E)](#2026-10-24-round-e) | Kept |
| 2026-10-23 to 2026-10-24 | [Why four scenes did worse than three (round C)](#2026-10-23-round-c) | Kept |
| 2026-10-22 to 2026-10-23 | [Four scenes with jitter, downscaled copies or a longer window (round B)](#2026-10-22-round-b) | Dropped |
| 2026-10-22 | [Whole-frame shift before the per-tile shift (step A)](#2026-10-22-whole-frame-shift) | Kept |
| 2026-10-22 | [Where the models fail (step 0)](#2026-10-22-failure-bins) | Finding |
| 2026-10-21 to 2026-10-22 | [A fair real reference, a fourth scene, and a stand-out filter](#2026-10-21-synthetic-week) | Kept |
| 2026-10-12 to 2026-10-14 | [Training only on rendered video](#2026-10-12-synthetic-to-real) | Finding |
| 2026-10-11 to 2026-10-12 | [Old pipeline, YOLOMG and UAV-Anti-UAV](#2026-10-11-external-baselines) | Reference |
| 2026-10-09 to 2026-10-11 | [Hard negative tiles](#2026-10-09-hard-negatives) | Kept |
| 2026-10-09 | [Per-tile alignment on the GPU](#2026-10-09-gpu-alignment) | Kept |
| 2026-10-07 to 2026-10-08 | [One network that sees colour and motion](#2026-10-07-one-network) | Kept |
| 2026-09-29 to 2026-10-01 | [Several drones at once, and an onboard compute budget](#2026-09-29-multi-drone-onboard) | Finding |
| 2026-09-23 to 2026-09-28 | [Finding the drone automatically with two YOLO detectors and verifiers](#2026-09-23-two-detector-acquisition) | Reference |
| 2026-09-13 to 2026-09-21 | [Interactive tracker that starts from a clicked drone](#2026-09-13-seeded-tracker) | Finding |

<a id="2026-11-19-sam-labels"></a>
## Labels for our own flights from Segment Anything 2.1

**Dates:** 2026-11-16 to 2026-11-19. **Phase:** 3. **Outcome:** Finding.

**Question.** How much of the misses on our flights comes from the labels, which were projected from the flight logs?

**What changed.** Segment Anything 2.1 followed the drone through every window, on a 1024 px full-resolution window around the projected position, started from one box per window and run forwards and backwards. Every tenth frame was reviewed on contact sheets by the AI assistant used for the analysis (not yet by a person), and stretches where it held a static object were rejected. 4,880 of the 5,906 drone frames of our hovering-camera flights got a box; the rest keep the projection.

**Result.** The projection is 20.5 px off the Segment Anything boxes (median; 85% within 50 px, up to 205 px, worst when the drones are closest). Scored against the new boxes, the current model tracks 90% of those frames within 20 px and 93% of all drone frames within 50 px, against 80% with the projected labels: most of the earlier misses on our flights were label error. The standard single-frame YOLO26n stays at 0-1%.

<a id="2026-11-11-round-aq"></a>
## The standard single-frame YOLO26n (round AQ)

**Dates:** 2026-11-10 to 2026-11-11. **Phase:** 3. **Outcome:** Reference.

**Question.** How much does the unmodified YOLO26n, which sees colour only, find when trained on the same rendered tiles?

**What changed.** The current model's tiles with only their B, G, R channels (`drone_model.colour_tiles`), 10 seeds from random weights and 10 from COCO-pretrained weights (the only arm that saw real photographs, through COCO), scored with `--channels 3`.

**Result.** Almost nothing transfers: ARD100 157 and 86 tracked frames against 2,279 and 2,238 for the current model at equal false boxes, NPS 596 and 522 against 1,100, UAV-Anti-UAV 344 and 561 against 2,004 and 2,702, our flights 14 and 58 against 4,703 and 4,781 at 0.25 (projected labels). Rendered validation mAP50 about 0.2 against 0.45. The motion channels carry the transfer, and COCO pretraining does not rescue colour alone.

<a id="2026-11-08-rounds-an-ap"></a>
## Birds back, clouds, and a second horizon view (rounds AN, AO, AP)

**Dates:** 2026-11-08 to 2026-11-11. **Phase:** 3. **Outcome:** Dropped.

**What changed.** On the current model: the scene's birds put back into one grey scene (AN); a cloud layer in the horizon scene's sky (AO); a second horizon render looking back along the camera's path (AP).

**Result.** Birds back lost small drones (ARD100 1,614 against 2,238 at equal false boxes, NPS 1,921 against 2,503, ARD100 drones of 10-20 px 36% against 51%) without a clear cut in false boxes. Clouds changed nothing clearly. The second horizon view lost ARD100 (1,616 and 1,687 against 2,238 and 2,310), NPS and our flights, with 40% more boxes on UAV-Anti-UAV's drone-free frames.

<a id="2026-11-05-rounds-ak-am"></a>
## A horizon scene and bird-free scenes: the current model (rounds AK, AL, AM)

**Dates:** 2026-11-05 to 2026-11-07. **Phase:** 3. **Outcome:** Kept.

**What changed.** AK: a rendered horizon view (camera near level, grey drones against sky, haze and treeline) added to the base. AL: the base's two grey scenes re-rendered without the 80 birds the shared Blender scene held, which had shown as unlabelled white shapes of drone size. AM: both together.

**Result.** The horizon scene lifted NPS sharply (2,181 and 2,233 against 321 and 210 at equal false boxes) and cost our flights a little. Removing the birds found many more small drones (ARD100 10-20 px about 51% against 39%, UAV-Anti-UAV under 10 px 3.7% against 0.4%) with more boxes on drone-free frames. Together (AM): ARD100 2,235 and 2,310 against 1,886 and 1,363, NPS 2,650 and 2,696 against 366 and 266, our flights level, UAV-Anti-UAV lower on one GPU type and level on the other; confirmed on 10 fresh seeds (ARD100 2,193 against 1,403, NPS 2,589 against 360). AM became the base, `am_nobirds_horizon`, on 2026-11-07.

<a id="2026-11-03-rounds-ae-ai"></a>
## Distractors, mosaic, a P2 head, scaling and rotation (rounds AE to AI)

**Dates:** 2026-11-03 to 2026-11-05. **Phase:** 3. **Outcome:** Dropped.

**What changed.** AE: screen-fixed text, or flapping birds and model planes, as unlabelled distractors. AF: Ultralytics' mosaic augmentation. AG: a P2 head (extra stride-4 output), or training for 1.5x magnification. AH: random resizing off or wider. AI: rotation and upside-down flips off.

**Result.** Every change lost clearly on at least one set, and none passed the rule. Distractors and the scale changes cost small ARD100 drones (10-20 px 13-22% against 41-43%); mosaic traded NPS for ARD100; without rotation or flips our flights lost. The settings that had never been tested on purpose (mosaic off, scaling 0.5-1.5, rotation 180°, upside-down flips) stay.

<a id="2026-11-05-round-aj"></a>
## Four hits to start a track (round AJ)

**Dates:** 2026-11-05. **Phase:** 3. **Outcome:** Kept.

**What changed.** The saved detections of the base replayed with 2, 3, 4 or 5 hits needed to start a track.

**Result.** 4 hits tracked more drones at equal false boxes on ARD100 (1,886 against 1,576) and UAV-Anti-UAV (2,491 against 2,179), with fewer drone-free boxes everywhere, and slightly fewer frames on our flights (4,822 against 4,892), since each extra hit drops one frame per track start. 4 hits are used from then on.

<a id="2026-11-01-test-time"></a>
## Test-time changes: an older past frame, homography, upscaling, an overlay filter (rounds V, W, X, AB, AC, AD)

**Dates:** 2026-11-01 to 2026-11-03. **Phase:** 3. **Outcome:** Dropped.

**What changed.** V and X: f-16 instead of f-4 as the oldest past frame (`--past-gap 16`). W: the per-tile homography at test time on the new base. AB and AC: frames upscaled by 1.5 or 2 before tiling, alone or combined with the 1x detections. AD: dropping detections on screen-fixed overlays.

**Result.** f-16 helped the older base on our flights but added nothing to the newer one. The homography and upscaling both trade ARD100 or UAV-Anti-UAV for NPS (upscaling triples UAV-Anti-UAV's false boxes), at two to three times the time per frame. The overlay filter dropped 0.1-0.7% of detections and changed nothing.

<a id="2026-10-31-own-renders"></a>
## Our own render scenes: drone colour, hovering, distance, a second terrain (rounds N, N2, S, T, T2, U, Y, Z, AA)

**Dates:** 2026-10-31 to 2026-11-03. **Phase:** 3. **Outcome:** Kept.

**What changed.** Scenes we rendered with the team's add-on: moving, hovering, distant and still drones; the same with grey instead of purple and orange drones; the stand-out filter on or off; the moving scene at a quarter dose; both grey scenes over a second terrain.

**Result.** Unfiltered, the new scenes wrecked transfer (ARD100 163-707 against 1,871). With the filter and grey drones, the moving and distant scenes together (`t_move_far`) were level on ARD100, clearly better on UAV-Anti-UAV (2,705 against 1,820) and our flights (4,892 against 4,024), and lower on NPS on one GPU type; they became the base on 2026-11-01. Grey drones fixed most of the loss of the purple ones (ARD100 1,531 against 419). Hovering and still drones never helped hovering drones: the filter drops the still ones, and kept unfiltered they make the network fire on static background. A second terrain lost ARD100 and our flights.

<a id="2026-10-31-round-q"></a>
## Rendered moving cars (round Q)

**Dates:** 2026-10-31. **Phase:** 3. **Outcome:** Dropped.

**What changed.** 39 cars modelled in Blender driving on a scene's roads; 464 car-centred empty tiles from that render, against the same tiles from the render with the cars hidden.

**Result.** The car pixels helped NPS (590 against 323) and our flights (4,277 against 3,362), but both arms lost about 40% of ARD100 against the base, so the pasted car patches stay.

<a id="2026-10-31-round-p"></a>
## A tracker gated to its own track (round P)

**Dates:** 2026-10-31. **Phase:** 3. **Outcome:** Dropped.

**What changed.** An established track takes the most confident detection inside its own gate instead of the frame's most confident one (`association='gated'`).

**Result.** Tracks lived longer but also held on to false targets: false boxes rose (ARD100 784 to 882) and no model tracked more at equal false boxes (ARD100 1,705 against 1,873). The frame's most confident detection stays.

<a id="2026-10-30-rounds-k-o"></a>
## Newer teammate scenes, more downscaled copies, longer training, a bigger network (rounds K, L, M, O)

**Dates:** 2026-10-30 to 2026-10-31. **Phase:** 3. **Outcome:** Dropped.

**What changed.** K and O: the teammates' scenes `centras2` and `ktuxi3`, with and without their cluttered empty tiles. L: a second downscaled copy per frame, or a 3.375 factor. M: 200 epochs instead of 100, or YOLO26s instead of YOLO26n.

**Result.** The newer scenes cut ARD100 about in half at equal false boxes (665-1,063 against 1,873) while helping UAV-Anti-UAV through fewer false boxes; their purple, orange or dark drones are the likely cause, which round S later supported. More copies, longer training and the bigger network gained nothing clear, so YOLO26n at 100 epochs stays.

<a id="2026-10-29-round-j"></a>
## Pasted cars and downscaled copies (round J)

**Dates:** 2026-10-29 to 2026-10-31. **Phase:** 3. **Outcome:** Kept.

**What changed.** E1 plus pasted moving car patches (`--cars`) and tiles from frames downscaled by 2 or 3 (`--scale-copies`), each alone and together, 10 seeds per arm from here on.

**Result.** Together (`cars_scale`) they beat E1 at equal false boxes on ARD100 (1,873 against 1,318), NPS (1,576 against 536) and UAV-Anti-UAV (3,403 against 2,924), level on our flights. Downscaled copies raise the share of ARD100 drones under 10 px tracked from 7% to 30-32%; the cars cut UAV-Anti-UAV drone-free boxes from 1,086 to 470-616, and a control with the same extra tiles but no car pixels shows the car pixels do it.

<a id="2026-10-27-round-i"></a>
## Two new scenes, motion dropout, and the homography at test time only (round I)

**Dates:** 2026-10-27. **Phase:** 3. Trained only on rendered video. **Outcome:** Lead.

**Question.** Do coloured drones, varied heights and hovering help? Does blanking motion in training teach the look of a drone? Does the homography help without retraining?

**What changed.** E1 plus scenes centras2 and ktuxi3; the same with motion blanked on 20% of tiles; E1's own network with the homography at test time.

**Result.** The new scenes made the model cautious (ARD100 1,690 to 970). Dropout taught nothing usable (0 of 118 real drones with motion blanked). The homography at test time helped NPS (882 to 2,057) and UAV-Anti-UAV (+11-42% at equal false boxes), kept our flights level and cost 5-9% on ARD100.

Results at confidence 0.25: tracked share (frames), false boxes.

| Model | Our 4K flights | ARD100 | UAV-Anti-UAV | NPS |
| --- | --- | --- | --- | --- |
| Six scenes | 66% (3,874), 1,202 false | 26% (970), 1,348 false | 8% (2,558), 1,257 false | 5% (340), 2,053 false |
| Six scenes, motion dropout | 65% (3,860), 994 false | 8% (308), 1,391 false | 5% (1,509), 1,153 false | 1% (86), 698 false |
| E1 with the homography at test time | 74% (4,345), 900 false | 42% (1,606), 1,232 false | 21% (6,530), 6,341 false | 28% (2,057), 3,741 false |
| Homography-trained, today's alignment | 55% (3,244), 1,857 false | 32% (1,232), 889 false | 10% (3,097), 2,509 false | 14% (1,061), 2,640 false |

<a id="2026-10-26-round-h"></a>
## Training on homography- and flow-aligned tiles (round H)

**Dates:** 2026-10-26. **Phase:** 3. Trained only on rendered video. **Outcome:** Dropped.

**Question.** Does a network trained on better-aligned tiles find more real drones?

**What changed.** E1's tiles rebuilt with the per-tile homography or the flow warp.

**Result.** Homography training helped NPS (2.3 times) and UAV-Anti-UAV (+40%) but hurt ARD100 (-18%) and our flights (-23%); a follow-up put the loss in the training, not the alignment. The flow warp had the fewest false boxes but lost too many drones.

Results at confidence 0.25: tracked share (frames), false boxes.

| Model | Our 4K flights | ARD100 | UAV-Anti-UAV | NPS |
| --- | --- | --- | --- | --- |
| Homography-aligned training | 57% (3,338), 1,802 false | 37% (1,387), 1,226 false | 18% (5,703), 3,980 false | 29% (2,139), 3,206 false |
| Flow-warped training | 50% (2,957), 2,320 false | 25% (949), 1,451 false | 11% (3,392), 3,173 false | 22% (1,636), 606 false |

<a id="2026-10-26-round-g"></a>
## Measuring better alignments and a hover-holding tracker (round G)

**Dates:** 2026-10-26. **Phase:** 3. Trained only on rendered video. **Outcome:** Finding.

**Question.** Before training, would a richer warp make hidden drones stand out, and would a tracker that holds slow tracks keep hovering drones?

**What changed.** Per-tile homography, a dense-flow warp and a parallax check, measured from video and labels; a tracker that coasts slow tracks up to 1 or 3 s.

**Result.** The homography brought out 25-66% of hidden drones and the flow warp 56-67%. The parallax check failed. The tracker hold added only 7 of 390 hovering ARD100 frames.

<a id="2026-10-25-round-f"></a>
## Pasted moving cars (round F)

**Dates:** 2026-10-25 to 2026-10-26. **Phase:** 3. Trained only on rendered video. **Outcome:** Option.

**Question.** Do moving cars pasted into the renders stop false tracks on real cars?

**What changed.** Rendered car patches, or cars plus drone-sized blobs, pasted as unlabelled moving objects.

**Result.** Cars cut false boxes (drone-free frames of our flights 548 to 489, UAV-Anti-UAV boxes while absent 1,188 to 447, NPS far boxes halved) but lost tiny ARD100 drones (1,690 to 1,301). Blobs taught the network to ignore small drones.

Results at confidence 0.25: tracked share (frames), false boxes.

| Model | Our 4K flights | ARD100 | UAV-Anti-UAV | NPS |
| --- | --- | --- | --- | --- |
| Pasted cars | 72% (4,279), 950 false | 34% (1,301), 874 false | 10% (3,051), 1,974 false | 18% (1,308), 2,051 false |
| Pasted cars and blobs | 73% (4,290), 717 false | 8% (308), 1,012 false | 6% (1,893), 1,779 false | 2% (120), 3,036 false |

<a id="2026-10-23-round-d"></a>
## Degraded renders, and the test on our own 4K flights (round D)

**Dates:** 2026-10-23 to 2026-10-25. **Phase:** 3. Trained only on rendered video. **Outcome:** Dropped.

**Question.** Do blurred and compressed renders help on real video? How do the rendered models do on our flights?

**What changed.** Half the rendered frames blurred, downscaled and JPEG-compressed. A test of 42 windows from our flights.

**Result.** Degrading made the model cautious with no gain. On our flights the rendered models track 62-73% of drone frames but keep a box on 80-87% of drone-free frames, on moving cars and a pedestrian.

Results at confidence 0.25: tracked share (frames), false boxes.

| Model | Our 4K flights | ARD100 | UAV-Anti-UAV | NPS |
| --- | --- | --- | --- | --- |
| Degraded renders | 62% (3,638), 1,493 false | 45% (1,710), 951 false | 7% (2,272), 2,069 false | 5% (344), 4,578 false |

<a id="2026-10-24-round-e"></a>
## The E1 recipe, and tiles built after the whole-frame shift (round E)

**Dates:** 2026-10-24. **Phase:** 3. Trained only on rendered video. **Outcome:** Kept.

**Question.** Which way of building the four-scene tiles works best?

**What changed.** Four scenes without amaliai1's empty tiles, tiles aligned per tile only (E1) or after the whole-frame shift (E2).

**Result.** E1 matches the best three-scene model and became the base. Building tiles after the whole-frame shift hurt (ARD100 1,385 against 1,690 at 0.25).

Results at confidence 0.25: tracked share (frames), false boxes.

| Model | Our 4K flights | ARD100 | UAV-Anti-UAV | NPS |
| --- | --- | --- | --- | --- |
| E1 (current base) | 73% (4,325), 968 false | 44% (1,690), 1,083 false | 13% (4,081), 4,095 false | 12% (882), 4,495 false |
| E2, tiles built after the shift | 67% (3,976), 1,243 false | 36% (1,385), 1,099 false | 10% (3,250), 4,858 false | 6% (438), 4,919 false |

<a id="2026-10-23-round-c"></a>
## Why four scenes did worse than three (round C)

**Dates:** 2026-10-23 to 2026-10-24. **Phase:** 3. Trained only on rendered video. **Outcome:** Kept.

**Question.** Is it the fourth scene amaliai1, or its drone-free tiles?

**What changed.** Four scenes without amaliai1, and four scenes without amaliai1's 493 drone-free tiles.

**Result.** The drone-free tiles were the cause (NPS 772 against 530 frames at 0.25 without them).

Results at confidence 0.25: tracked share (frames), false boxes.

| Model | ARD100 | UAV-Anti-UAV | NPS |
| --- | --- | --- | --- |
| Without amaliai1 | 43% (1,644), 1,086 false | 11% (3,358), 4,887 false | 10% (766), 4,630 false |
| Without amaliai1's empty tiles | 45% (1,711), 1,032 false | 12% (3,912), 5,370 false | 10% (772), 4,605 false |

<a id="2026-10-22-round-b"></a>
## Four scenes with jitter, downscaled copies or a longer window (round B)

**Dates:** 2026-10-22 to 2026-10-23. **Phase:** 3. Trained only on rendered video. **Outcome:** Dropped.

**Question.** Do training-time tricks make the model robust to imperfect alignment and tiny drones?

**What changed.** Jittered past frames; downscaled tile copies; an extra past frame 8 frames back.

**Result.** Jitter hurt everywhere. Downscaled copies quadrupled the tracked share of ARD100 drones under 10 px (0.06 to 0.24) but lost on UAV-Anti-UAV. The longer window changed nothing.

Results at confidence 0.25: tracked share (frames), false boxes.

| Model | ARD100 | UAV-Anti-UAV | NPS |
| --- | --- | --- | --- |
| 4 scenes, base | 46% (1,744), 904 false | 10% (3,192), 4,237 false | 7% (530), 4,749 false |
| Jittered past frames | 27% (1,020), 1,069 false | 8% (2,532), 2,190 false | 4% (272), 2,805 false |
| Downscaled copies | 49% (1,866), 600 false | 8% (2,477), 3,418 false | 8% (622), 4,702 false |
| Longer window (f-8) | 46% (1,766), 907 false | 9% (2,968), 3,506 false | 6% (435), 4,845 false |

<a id="2026-10-22-whole-frame-shift"></a>
## Whole-frame shift before the per-tile shift (step A)

**Dates:** 2026-10-22. **Phase:** 3. Trained only on rendered video. **Outcome:** Kept.

**Question.** Does shifting the whole frame first help fast camera motion?

**What changed.** One whole-frame shift of up to 256 px before the per-tile alignment, at test time.

**Result.** More frames aligned and fewer far boxes on ARD100 and NPS, little change on UAV-Anti-UAV.

Results at confidence 0.25: tracked share (frames), false boxes.

| Model | ARD100 | UAV-Anti-UAV | NPS |
| --- | --- | --- | --- |
| 3 scenes (b out), with the whole-frame shift | 45% (1,704), 1,003 false | 11% (3,507), 3,140 false | 13% (958), 4,431 false |
| Real tiles, with the whole-frame shift | 47% (1,794), 709 false | 9% (2,890), 4,294 false | 29% (2,152), 878 false |

<a id="2026-10-22-failure-bins"></a>
## Where the models fail (step 0)

**Dates:** 2026-10-22. **Phase:** 3. Trained only on rendered video. **Outcome:** Finding.

**Question.** Which frames do the models miss, by camera speed, drone speed, size and stand-out?

**What changed.** Saved results split by measures taken from the video and labels only.

**Result.** The strongest factor is whether the drone stands out in the motion image (tracked 2-15 times less when not, 17% of ARD100 and 40% of UAV-Anti-UAV frames). Hovering drones are tracked 2-6 times less on ARD100. 63% of ARD100 drone frames are under 20 px, smaller than any rendered drone.

<a id="2026-10-21-synthetic-week"></a>
## A fair real reference, a fourth scene, and a stand-out filter

**Dates:** 2026-10-21 to 2026-10-22. **Phase:** 3. Trained only on rendered video. **Outcome:** Kept.

**Question.** How good is a model trained on real tiles exactly like the rendered one? Does scene amaliai1 help? Do tiles where background clutter hides the drone hurt?

**What changed.** Real tiles with the rendered recipe; amaliai1 added; 373 eiguliai1 tiles where every drone stands out against 373 random ones.

**Result.** The fair real reference is stronger (ARD100 1,847 frames at 0.25). amaliai1 as delivered did not help. Keeping only standing-out tiles helped (NPS 793 against 203 at 0.5).

Results at confidence 0.25: tracked share (frames), false boxes.

| Model | ARD100 | UAV-Anti-UAV | NPS |
| --- | --- | --- | --- |
| Real tiles, same recipe | 49% (1,847), 817 false | 9% (2,947), 4,316 false | 27% (1,978), 1,269 false |
| 3 scenes, stand-out tiles (b out) | 45% (1,692), 1,026 false | 11% (3,507), 3,096 false | 10% (746), 4,381 false |
| 3 scenes, random tiles (b rnd) | 36% (1,371), 1,234 false | 10% (3,256), 4,755 false | 3% (257), 4,492 false |
| 2 scenes + amaliai1 | 25% (959), 1,556 false | 8% (2,396), 5,175 false | 2% (161), 5,349 false |

<a id="2026-10-12-synthetic-to-real"></a>
## Training only on rendered video

**Dates:** 2026-10-12 to 2026-10-14. **Phase:** 3. Trained only on rendered video. **Outcome:** Finding.

**Question.** Does a detector trained only on the teammate's renders find real drones?

**What changed.** Two rendered scenes, colour-only against colour plus motion, a COCO start against random weights; then a third scene.

**Result.** Colour plus motion from random weights transfers in part (ARD100 852 frames at 0.5). Colour alone transfers almost nothing (0-9 frames). The third scene made the model cautious. With motion blanked every rendered model finds 0 drones.

Results at confidence 0.25: tracked share (frames), false boxes.

| Model | ARD100 | UAV-Anti-UAV | NPS |
| --- | --- | --- | --- |
| 2 scenes, colour + motion | 26% (983), 1,582 false | 14% (4,311), 11,237 false | 11% (846), 3,597 false |
| 2 scenes, colour only | 0% (0), 2,311 false | 1% (223), 2,310 false | 0% (0), 203 false |
| 3 scenes, colour + motion | 18% (701), 1,270 false | 8% (2,552), 2,412 false | 3% (250), 4,433 false |

<a id="2026-10-11-external-baselines"></a>
## Old pipeline, YOLOMG and UAV-Anti-UAV

**Dates:** 2026-10-11 to 2026-10-12. **Phase:** 2. One network, trained on real tiles. **Outcome:** Reference.

**Question.** How do the one-network models compare with our old pipeline and the published YOLOMG, and how do they do on UAV-Anti-UAV?

**What changed.** Every model scored by the same tracker and scorers on the same windows.

**Result.** The one-network models beat the old pipeline. YOLOMG, trained on all 65 ARD100 videos, led on ARD100 (1,991-2,791 frames at 0.5). On UAV-Anti-UAV no model tracked more than 2% of drone frames at 0.5.

Results at confidence 0.25: tracked share (frames), false boxes.

| Model | ARD100 |
| --- | --- |
| YOLOMG 1280 (published) | 81% (3,075), 89 false |
| YOLOMG 640 (published) | 74% (2,795), 106 false |

<a id="2026-10-09-hard-negatives"></a>
## Hard negative tiles

**Dates:** 2026-10-09 to 2026-10-11. **Phase:** 2. One network, trained on real tiles. **Outcome:** Kept.

**Question.** Do drone-free tiles where the model fired by mistake cut false tracks?

**What changed.** 40, 80 or 160 mined negative tiles, or 160 random ones, 10 seeds each.

**Result.** 160 mined negatives cut ARD100 far boxes from 159 to 48 (medians) while tracking 690 instead of 838 frames, and beat random tiles on false boxes.

Results at confidence 0.5: tracked share (frames), false boxes.

| Model | ARD100 |
| --- | --- |
| No negatives | 22% (838), 159 false |
| 160 mined negatives | 18% (690), 48 false |
| 40 mined negatives | 22% (842), 66 false |
| 160 random negatives | 20% (760), 99 false |

<a id="2026-10-09-gpu-alignment"></a>
## Per-tile alignment on the GPU

**Dates:** 2026-10-09. **Phase:** 2. One network, trained on real tiles. **Outcome:** Kept.

**Question.** Can the per-tile alignment run fast enough?

**What changed.** The per-tile phase correlation and warp batched on the GPU.

**Result.** Identical scores, 460 ms down to 82 ms per 4K frame on an RTX 4070.

<a id="2026-10-07-one-network"></a>
## One network that sees colour and motion

**Dates:** 2026-10-07 to 2026-10-08. **Phase:** 2. One network, trained on real tiles. **Outcome:** Kept.

**Question.** Can one detector replace the two, using the current frame in colour plus aligned past frames?

**What changed.** A 5-channel YOLO26n (B, G, R plus two aligned past grays) against a 3-channel gray-motion reference.

**Result.** Colour plus per-tile alignment matched the reference on ARD100 at confidence 0.5, and whole-frame alignment cost accuracy. Correct first starts were 9-14 of 26 windows for every model.

Results at confidence 0.5: tracked share (frames), false boxes.

| Model | ARD100 |
| --- | --- |
| Colour + motion, per-tile alignment | 24% (919), 169 false |
| Gray motion reference | 23% (878), 377 false |
| Colour + motion, whole-frame alignment | 14% (543), 205 false |

<a id="2026-09-29-multi-drone-onboard"></a>
## Several drones at once, and an onboard compute budget

**Dates:** 2026-09-29 to 2026-10-01. **Phase:** 1. Tracking from a human click. **Outcome:** Finding.

**Question.** Can the pipeline keep searching while tracking, and what would a Jetson allow?

**What changed.** A multi-drone runner and a desktop proxy with CPU, memory and scan-cadence limits.

**Result.** One drone leaving and returning got four separate IDs. Scanning every 15 frames kept boxes up but few were on the drone. No board measurement.

<a id="2026-09-23-two-detector-acquisition"></a>
## Finding the drone automatically with two YOLO detectors and verifiers

**Dates:** 2026-09-23 to 2026-09-28. **Phase:** 1. Tracking from a human click. **Outcome:** Reference.

**Question.** Can the drone be found without a click, on whole 4K frames?

**What changed.** A gray YOLO and a temporal YOLO on 640 px tiles, two CNN verifiers, then a tracker.

**Result.** Candidates reached the drone in many frames, but the full pipeline started correctly in only 0 of 4 and 1 of 4 unseen windows on two reserved ARD100 video pairs.

<a id="2026-09-13-seeded-tracker"></a>
## Interactive tracker that starts from a clicked drone

**Dates:** 2026-09-13 to 2026-09-21. **Phase:** 1. Tracking from a human click. **Outcome:** Finding.

**Question.** Can a tracker follow a tiny drone in our 4K flights once a person marks it?

**What changed.** Motion proposals, appearance memory, prediction and short recovery after loss, in a local review tool.

**Result.** It follows the drone through long stretches and recovers after short losses, but it needs a human to start it and gives no metric speed without an assumed height.
