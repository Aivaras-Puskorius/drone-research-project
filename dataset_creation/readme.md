# Realistic Drone Flight Generator (Blender add-on)

A Blender add-on for building synthetic drone-detection datasets. It animates a drone (or any other object) and exports per-frame telemetry, including the drone's 2D bounding box in the camera image, to CSV.

**Features**

- **Random flight**: generates a realistic, reproducible (seeded) drone flight with wind, tilt and small stabilisation wobble.
- **KMZ mission**: flies a real DJI waypoint mission (`.kmz`/`.wpml`) or a KML path exactly as planned.
- **CSV export**: writes position, rotation, height above ground, distance to the camera and the image-space bounding box for every frame.
- **CSV import**: recreates a flight from a previously exported CSV, so any dataset sequence can be reproduced.
- **Range-safe editing**: generating or loading a flight only replaces keyframes inside the chosen frame range. Animation outside the range is left untouched.

---

## Requirements

- Blender **4.0 or newer** (tested with Blender 4.3)
- No extra Python packages are needed.

## Installation

1. Download [`realistic_drone_flight_generator_v1_5.py`](realistic_drone_flight_generator_v1_5.py).
2. In Blender, open **Edit → Preferences → Add-ons**.
3. Install the file:
   - **Blender 4.2+**: click the **⌄** menu in the top-right corner → **Install from Disk…**
   - **Blender 4.0 / 4.1**: click **Install…**
4. Select the `.py` file and make sure **Animation: Realistic Drone Flight Generator** is ticked.
5. In the 3D Viewport, press **N** to open the sidebar and select the **Drone Flight** tab.

To update the add-on, disable or remove the old version in Preferences, then install the new file the same way.

---

## Scene setup

### Scene assets (not in this repository)

The photogrammetry terrain (~2.8 GB) and the example scene (`kmz_test_spiral.blend`, ~300 MB) are too large for GitHub. Download them from OneDrive:

**➡ [Download the scene assets](ONEDRIVE_LINK_HERE)**

Place them inside `dataset_creation/` with this structure:

```
dataset_creation/
├── kmz_test_spiral.blend
├── photogrametry/
│   └── ktuxi/
│       ├── odm_textured_model_geo.obj
│       ├── odm_textured_model_geo.mtl
│       └── odm_textured_model_geo_material*_map_Kd.png
└── drone_models/        (included in the repository)
```

These paths are ignored by Git (see `.gitignore`), so they won't be committed by accident.

### Quick start

Open `kmz_test_spiral.blend`. Its camera, terrain and drone are already set up. If the terrain appears without textures, delete it and import it again as described below.

### From scratch

1. **Terrain (photogrammetry)**
   - Import the model, e.g. `photogrametry/ktuxi/odm_textured_model_geo.obj`, with **File → Import → Wavefront (.obj)**.
   - Rotate it **-90° around X**: select it, then press `R` `X` and type `-90`.
   - Right-click it and choose **Set Origin → Geometry to Origin**.
   - Optional: put the terrain into its own collection and select that collection as **Ground Collection** in the add-on. This makes the height calculation faster and more reliable.

2. **Drone with a bounding box**

   The CSV bounding box is computed from an **Empty with Cube display**, so the drone model must be parented to one:
   - Add the cube with **Shift+A → Empty → Cube**.
   - Import the drone model, e.g. `drone_models/mavic/scene.gltf`.
   - Scale the model to its real size and fit the Empty's **Size** to the drone.
   - Parent the model to the Empty. Select the model first, then the Empty, and press **Ctrl+P → Object**.
   - Turn the model so its nose points along the Empty's **+X** axis. The add-on turns the object to face its direction of travel along +X.
   - Always select the **Empty** when using the add-on.

3. **Camera**

   Add a camera and make it the scene camera with **Ctrl+Numpad 0** or **Scene Properties → Camera**. CSV export needs an active camera.

All distances are in metres (Blender units).

---

## Usage

The add-on always works on the **active (selected) object**.

### Common behaviour

All three flight sources (random flight, KMZ mission and CSV import) share these rules:

- **Range-safe**: only keyframes inside the flight's frame range are replaced. Keyframes before and after it, and the motion between them, stay exactly as they were.
- **Scene range**: the scene's start and end frames are set to the new flight, so **Save CSV** exports exactly that flight.
- **Current frame**: when generation finishes, Blender is left on the flight's first frame.

### 1. Generate a random flight

1. Select the drone Empty and place it where the flight should start.
2. Set **Start Frame** and **Frames**. For example, Start Frame `250` with Frames `600` animates frames **250 to 849** inclusive. The panel shows the resulting range and length in seconds.
3. Adjust the parameters (see the [parameter reference](#parameter-reference)).
4. Click **Generate Flight**.

Notes:

- The flight starts from the object's position at the start frame. Altitude is measured from that starting height.
- The same **Seed** with the same settings always produces the same flight.
- You can build long flights from several segments, e.g. generate 1–249 and then 250–849.

### 2. Fly a KMZ mission

1. Place the **3D cursor** at the takeoff point (`Shift+Right-click`). Mission heights are measured from this point.
2. Select the drone Empty and set **Start Frame**.
3. Click **Fly KMZ Mission** and choose the file.

How the mission is mapped into the scene:

| Mission data | In Blender |
|---|---|
| Latitude / longitude | Local metres around the first waypoint: **+X = east, +Y = north** |
| Height | Z above the 3D cursor |
| Speed (global or per waypoint) | Timing of the flight |
| Hover actions | The drone stops for the given time |

Notes:

- **Length**: by default the flight takes as long as the mission does at the speed in the file. With **Fit To Frames** ticked, the mission is stretched or compressed to exactly **Frames** frames.
- **Exact path**: the mission is flown as planned. **Wind and Drone Motion settings are not applied**: the drone only turns to face its direction of travel, with no roll or pitch.
- **Samples / Second** sets how often keyframes are placed along the path. Use `24` (or your frame rate) to follow the path as closely as possible.
- **Supported files**: DJI KMZ (`wpmz/waylines.wpml` or `wpmz/template.kml`), plain `.wpml`, and `.kml` files with a `LineString` path (e.g. from Google Earth).

### 3. Export telemetry to CSV

1. Make sure the scene has an active camera and the drone Empty is selected.
2. Click **Save CSV** and choose the file name.

Every frame of the scene's frame range is exported:

| Column | Description |
|---|---|
| `frame`, `time_seconds` | Frame number and time (`frame / fps`) |
| `x_m`, `y_m`, `z_m` | World position in metres |
| `rotation_x_deg`, `rotation_y_deg`, `rotation_z_deg` | World rotation (XYZ Euler) in degrees |
| `height_above_ground_m` | Height above the terrain found by a downward raycast; `-1` if no ground was hit |
| `distance_to_camera_m` | Distance from the drone to the scene camera |
| `bbox_xmin_px`, `bbox_ymin_px`, `bbox_xmax_px`, `bbox_ymax_px` | Bounding box of the Empty's cube in image pixels (origin top-left) |
| `bbox_width_px`, `bbox_height_px` | Bounding box size in pixels |
| `bbox_visible` | `1` if the box is in front of the camera and at least partly inside the image, otherwise `0` |

Bounding box values are only filled in when the selected object is an **Empty with Cube display**. For any other object they are `0`.

### 4. Load a flight from CSV

1. Select the object that should fly. It can be the drone Empty or any other object, such as the camera.
2. Click **Load CSV** and choose a CSV exported by this add-on.

The flight is recreated with a keyframe on every frame in the file.

- **Frames**: by default the flight is placed on the frames stored in the CSV. Tick **Load At Start Frame** to shift it so it begins at **Start Frame** instead.
- **World coordinates**: the CSV stores world position and rotation, so parented objects are handled automatically.
- **Excel-edited files**: CSVs re-saved by Excel with `;` separators and decimal commas also load.
- **Frame-rate check**: if the CSV was saved at a different frame rate than the scene, you get a warning, because the flight would play at a different speed.

**Reproducing the dataset:** open the scene, then for each recorded object (drone, camera, …) select it and load its CSV.

### 5. Clear Flight

**Clear Flight** removes **all** animation from the selected object, not just the current range.

---

## Parameter reference

| Group | Parameter | Default | Meaning |
|---|---|---|---|
| Flight | Start Frame | 1 | First frame of the flight |
| | Frames | 720 | Number of frames, start frame included |
| | Average Speed | 8 m/s | Mean horizontal speed (varies ±25 %) |
| | Altitude Min / Max | 5 / 20 m | Height range above the starting height |
| | Direction Change | 3.5 s | How often a new heading is chosen |
| | Direction Randomness | 0.45 | How sharply the heading can change |
| Environment | Wind | 0.8 m/s | Horizontal wind drift |
| | Vertical Wind | 0.2 m/s | Vertical wind drift |
| | Wind Frequency | 0.12 | How fast the wind changes |
| | Stabilization | 0.7 | Small position wobble of the drone's stabilisation |
| Drone Motion | Tilt Strength | 0.035 | How much the drone tilts when it accelerates or turns |
| | Max Roll / Max Pitch | 20° / 14° | Tilt limits |
| | Micro Motion | 0.008 | Small rotation jitter |
| Generation | Seed | 42 | Random seed; same seed and settings give the same flight |
| | Samples / Second | 4 | Keyframes per second (random and KMZ flights) |
| KMZ Mission | KMZ File | – | Last used mission file |
| | Fit To Frames | off | Fit the mission into **Frames** instead of using its speed |
| Ground / Terrain | Ground Collection | – | Meshes used for the height raycast; all visible meshes if empty |
| | Raycast Distance | 1000 m | Maximum distance to search for ground below the drone |
| | Ground Offset | 0 m | Value added to the measured height |
| CSV Export | CSV File | `//drone_flight.csv` | Default file for Save / Load CSV |
| | Load At Start Frame | off | Shift a loaded flight to **Start Frame** |

The Environment and Drone Motion groups only affect **Generate Flight**.

---

## Known limitations

- **KMZ heights**: heights are treated as relative to the takeoff point (`relativeToStartPoint`, the DJI default). For missions with absolute heights (WGS84 / EGM96), move the 3D cursor down by the ground's elevation above sea level.
- **KMZ heading**: DJI heading modes are ignored. The drone always faces its direction of travel.
- **Rotation mode**: generated flights use the XYZ rotation mode and key `location` and `rotation_euler`.
- **Tilt axes**: in random flights, forward acceleration is applied as rotation around the object's X axis and sideways acceleration around its Y axis.

---

## Credits

This work is based on "DJI Mavic 3" (https://sketchfab.com/3d-models/dji-mavic-3-c5a5abae1dea468ab73b1bdc7d616fa6) by llirikslon (https://sketchfab.com/llirikslon) licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)
