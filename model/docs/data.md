# Data

No videos, image tiles or trained weights are stored in this repository. This page says what the model uses and where each part comes from.

## Training data: rendered video

The detector is trained only on video rendered from photogrammetry models of real places. The team's dataset pipeline produces it; see that part of the repository. Each scene comes with exact per-frame boxes and 3D positions of every drone and of the camera.

| Scene | Frames | Drones | What it adds |
| --- | --- | --- | --- |
| `centras1` | 400 | 7 | City centre, slowly moving camera |
| `ktuxi2` | 400 | 7 | Still camera looking almost straight down |
| `eiguliai1` | 1,120 (3 clips) | 7 | Smaller drones over strong background parallax |
| `amaliai1` | 1,120 (3 clips) | 7 | Suburb, 340 frames with no drone in view, same drone paths as `eiguliai1` |
| `centras2` | 1,000 | 4 | Brightly coloured drones, flying heights 40–50 m, hovering |
| `ktuxi3` | 1,000 | 4 | Dark drones, the same height mix, hovering |
| `spirale` | 5,500 | 1 | A planned two-drone mission at our flight site, observer looking straight down; not used for training |

The current base model (E1) trains on the first four scenes. From `eiguliai1` and `amaliai1` it keeps only the drone tiles where every drone stands out in the motion image, and it leaves out `amaliai1`'s drone-free tiles.

## Test data

| Set | Source | Use terms as we understand them | In this repository |
| --- | --- | --- | --- |
| Our 4K flights | 20 flights by the team, two drones, 3840 × 2160 | Team-owned | Not included |
| ARD100 | Public drone-to-drone dataset, copy on Zenodo | Zenodo lists CC-BY-4.0; the copy was uploaded by a third party, and the authors state rights for the annotations only | Not included |
| UAV-Anti-UAV | Public benchmark | Academic, non-commercial and non-military use only | Not included |
| NPS-Drones | Public dataset of fixed-wing drones | BSD 3-Clause | Not included |

We have not had these terms checked by the university. The files in [`windows/`](../windows) say where each test window lies and how it was chosen, without labels. `python -m drone_model.make_plans` rebuilds the labels from your own copy of each dataset and checks them against ours by SHA-256.

## Software

The detector is YOLO26n from Ultralytics, trained from random weights. Ultralytics is released under AGPL-3.0, which matters for how this code can be licensed and shared. The team has not chosen a licence for this repository yet.
