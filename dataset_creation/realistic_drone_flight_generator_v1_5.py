bl_info = {
    "name": "Realistic Drone Flight Generator",
    "author": "OpenAI",
    "version": (1, 5, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Drone Flight",
    "description": "Generates realistic randomized or KMZ-based drone flight and exports per-frame telemetry to CSV.",
    "category": "Animation",
}

import bpy
import bisect
import csv
import math
import os
import random
import zipfile
import xml.etree.ElementTree as ET
from mathutils import Vector, Euler, Matrix
from bpy.props import (
    FloatProperty, IntProperty, BoolProperty,
    StringProperty, PointerProperty
)
from bpy.types import Operator, Panel, PropertyGroup
from bpy_extras.object_utils import world_to_camera_view


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


MAX_FRAME = 1048574
ANIMATED_PATHS = ("location", "rotation_euler")


def object_fcurves(obj, data_paths=ANIMATED_PATHS):
    """F-Curves of the object's action (legacy and slotted actions)."""
    ad = obj.animation_data
    if ad is None or ad.action is None:
        return []

    fcurves = None
    slot = getattr(ad, "action_slot", None)
    if slot is not None:
        try:
            from bpy_extras.anim_utils import action_get_channelbag_for_slot
        except ImportError:
            pass
        else:
            bag = action_get_channelbag_for_slot(ad.action, slot)
            fcurves = bag.fcurves if bag else []
    if fcurves is None:
        fcurves = ad.action.fcurves

    return [fc for fc in fcurves if fc.data_path in data_paths]


def evaluate_without_modifiers(fc, frame):
    states = [m.mute for m in fc.modifiers]
    for m in fc.modifiers:
        m.mute = True
    try:
        return fc.evaluate(frame)
    finally:
        for m, state in zip(fc.modifiers, states):
            m.mute = state


def _bezier_points(p0, p1, p2, p3, t):
    a, b, c = p0.lerp(p1, t), p1.lerp(p2, t), p2.lerp(p3, t)
    d, e = a.lerp(b, t), b.lerp(c, t)
    return a, c, d, e, d.lerp(e, t)


def split_bezier_segment(p0, p1, p2, p3, x):
    """
    Split an F-Curve Bezier segment at time x without changing its shape.
    Returns (left key right handle, new key left handle, new key,
             new key right handle, right key left handle).
    """
    p1, p2 = p1.copy(), p2.copy()

    # Same handle correction Blender applies before evaluating a segment.
    span = p3.x - p0.x
    len1, len2 = abs(p0.x - p1.x), abs(p3.x - p2.x)
    if len1 + len2 > span:
        fac = span / (len1 + len2)
        p1 = p0.lerp(p1, fac)
        p2 = p3.lerp(p2, fac)

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) * 0.5
        if _bezier_points(p0, p1, p2, p3, mid)[-1].x < x:
            lo = mid
        else:
            hi = mid

    a, c, d, e, point = _bezier_points(p0, p1, p2, p3, (lo + hi) * 0.5)
    return a, d, point, e, c


def _new_key(x, y, interpolation, handle_left=None, handle_right=None):
    if handle_left is None:
        handle_left = Vector((x - 1.0 / 3.0, y))
    if handle_right is None:
        handle_right = Vector((x + 1.0 / 3.0, y))
    return float(x), y, interpolation, handle_left, handle_right


def _set_key(kp, interpolation, handle_left, handle_right):
    kp.interpolation = interpolation
    kp.handle_left_type = 'FREE'
    kp.handle_right_type = 'FREE'
    kp.handle_left = handle_left
    kp.handle_right = handle_right


def clear_range_keep_outside(fc, start, end):
    """
    Remove the keys of an F-Curve inside [start, end] while keeping the curve
    identical on every frame outside that range.

    The nearest outside keys get FREE handles (so Blender does not recompute
    them for their new neighbours), and if the old curve crossed the range
    border, an extra key is placed at start - 1 / end + 1 holding the old value.
    """
    keys = sorted(fc.keyframe_points, key=lambda k: k.co.x)
    before = [k for k in keys if k.co.x < start]
    after = [k for k in keys if k.co.x > end]
    from_start = [k for k in keys if k.co.x >= start]
    to_end = [k for k in keys if k.co.x <= end]

    new_keys = []
    neighbours = []

    if before:
        left = before[-1]
        ipo = left.interpolation
        hl, hr = left.handle_left.copy(), left.handle_right.copy()
        f = start - 1
        if left.co.x < f:
            nxt = from_start[0] if from_start else None
            if nxt is None:
                # Frame f lies in the straight extrapolation after the last key.
                if ipo != 'CONSTANT':
                    ipo = 'LINEAR'
                new_keys.append(_new_key(f, evaluate_without_modifiers(fc, f), 'BEZIER'))
            elif ipo == 'BEZIER':
                hr, key_hl, point, _, _ = split_bezier_segment(
                    left.co.copy(), hr, nxt.handle_left.copy(), nxt.co.copy(), f)
                new_keys.append(_new_key(f, point.y, 'BEZIER', handle_left=key_hl))
            elif ipo in {'CONSTANT', 'LINEAR'}:
                new_keys.append(_new_key(f, evaluate_without_modifiers(fc, f), 'BEZIER'))
            else:
                # Easing curves cannot be split exactly: bake whole frames.
                for fr in range(math.floor(left.co.x) + 1, f + 1):
                    new_keys.append(_new_key(
                        fr, evaluate_without_modifiers(fc, fr),
                        'LINEAR' if fr < f else 'BEZIER'))
                ipo = 'LINEAR'
        neighbours.append((left.co.x, ipo, hl, hr))

    if after:
        right = after[0]
        hl, hr = right.handle_left.copy(), right.handle_right.copy()
        f = end + 1
        if right.co.x > f:
            prev = to_end[-1] if to_end else None
            # No key before: frame f lies in the straight extrapolation before the first key.
            ipo = prev.interpolation if prev else 'LINEAR'
            if ipo == 'BEZIER':
                _, _, point, key_hr, hl = split_bezier_segment(
                    prev.co.copy(), prev.handle_right.copy(), hl, right.co.copy(), f)
                new_keys.append(_new_key(f, point.y, 'BEZIER', handle_right=key_hr))
            elif ipo in {'CONSTANT', 'LINEAR'}:
                new_keys.append(_new_key(f, evaluate_without_modifiers(fc, f), ipo))
            else:
                for fr in range(f, math.ceil(right.co.x)):
                    new_keys.append(_new_key(fr, evaluate_without_modifiers(fc, fr), 'LINEAR'))
        neighbours.append((right.co.x, right.interpolation, hl, hr))

    kps = fc.keyframe_points
    for i in reversed(range(len(kps))):
        if start <= kps[i].co.x <= end:
            kps.remove(kps[i], fast=True)

    for x, y, ipo, hl, hr in new_keys:
        kp = kps.insert(x, y, options={'FAST'})
        _set_key(kp, ipo, hl, hr)

    for x, ipo, hl, hr in neighbours:
        for kp in kps:
            if abs(kp.co.x - x) < 1e-4:
                _set_key(kp, ipo, hl, hr)
                break


def generate_flight(obj, p):
    scene = bpy.context.scene
    fps = scene.render.fps / scene.render.fps_base
    start = p.start_frame
    frame_count = max(2, p.frame_count)
    end = start + frame_count - 1
    duration = frame_count / fps
    rng = random.Random(p.seed)

    sample_frames = sample_frame_list(start, end, fps, p.samples_per_second)

    controls =max(4, int(duration / max(p.direction_change_seconds, 0.25)) + 2)
    dirs, heights, speeds = [], [], []
    direction = Vector((1, 0, 0))

    for _ in range(controls):
        angle = rng.uniform(-p.direction_randomness, p.direction_randomness)
        direction.rotate(Euler((0, 0, angle), 'XYZ'))
        fresh_angle = rng.uniform(-math.pi, math.pi)
        fresh = Vector((math.cos(fresh_angle), math.sin(fresh_angle), 0))
        direction = (direction * 0.72 + fresh * 0.28).normalized()
        dirs.append(direction.copy())
        heights.append(rng.uniform(p.altitude_min, p.altitude_max))
        speeds.append(max(0.1, p.average_speed * rng.uniform(0.75, 1.25)))

    # The flight starts where the drone is at the start frame.
    scene.frame_set(start)
    positions, velocities = [], []
    pos = obj.location.copy()
    base_z = pos.z
    prev_pos = pos.copy()

    for idx, frame in enumerate(sample_frames):
        u = idx / max(1, len(sample_frames) - 1)
        x = u * (controls - 1)
        k = min(controls - 2, int(math.floor(x)))
        t = x - k
        s = smoothstep(t)

        d = (dirs[k] * (1 - s) + dirs[k + 1] * s).normalized()
        speed = speeds[k] * (1 - s) + speeds[k + 1] * s
        z_offset = heights[k] * (1 - s) + heights[k + 1] * s

        wind = Vector((
            math.sin(u * math.tau * p.wind_frequency + 1.7) * p.wind_strength,
            math.sin(u * math.tau * p.wind_frequency * 0.73 + 3.1) * p.wind_strength,
            math.sin(u * math.tau * p.wind_frequency * 0.51 + 0.4) * p.wind_vertical
        ))

        dt = (sample_frames[idx] - sample_frames[idx - 1]) / fps if idx else 0.0
        if idx:
            pos += (d * speed + wind) * dt
        pos.z = base_z + z_offset

        if p.stabilization > 0:
            stab = p.stabilization * 0.03
            pos.x += math.sin(u * 17.0 + 0.7) * stab
            pos.y += math.sin(u * 13.0 + 2.1) * stab
            pos.z += math.sin(u * 11.0 + 4.0) * stab * 0.6

        vel = (pos - prev_pos) / max(dt, 1.0 / fps)
        positions.append(pos.copy())
        velocities.append(vel.copy())
        prev_pos = pos.copy()

    rotations = flight_rotations(sample_frames, velocities, p, fps)
    return apply_flight(obj, sample_frames, positions, rotations, start, end)


def sample_frame_list(start, end, fps, samples_per_second):
    step = max(1, int(fps / max(samples_per_second, 1)))
    frames = list(range(start, end + 1, step))
    if frames[-1] != end:
        frames.append(end)
    return frames


def flight_heading(velocities):
    """Yaw per sample following the flight direction, kept continuous."""
    # Heading per sample; while hovering keep the last known heading.
    yaws = [
        math.atan2(v.y, v.x) if Vector((v.x, v.y)).length >= 0.001 else None
        for v in velocities
    ]
    first = next((y for y in yaws if y is not None), 0.0)
    prev = first
    for i, yaw in enumerate(yaws):
        if yaw is None:
            yaw = prev
        # Keep yaw continuous so the drone doesn't spin 360 deg crossing +-180 deg.
        while yaw - prev > math.pi:
            yaw -= math.tau
        while yaw - prev < -math.pi:
            yaw += math.tau
        yaws[i] = prev = yaw
    return yaws


def flight_rotations(sample_frames, velocities, p, fps):
    """Yaw follows the flight direction, roll/pitch come from acceleration."""
    yaws = flight_heading(velocities)

    rotations = []
    for i, (frame, vel) in enumerate(zip(sample_frames, velocities)):
        yaw = yaws[i]

        accel = velocities[i] - velocities[i - 1] if i > 0 else Vector((0, 0, 0))
        local_right = Vector((-math.sin(yaw), math.cos(yaw), 0))
        forward = Vector((math.cos(yaw), math.sin(yaw), 0))
        lateral_accel = accel.dot(local_right)
        forward_accel = accel.dot(forward)

        roll = max(-p.max_roll, min(p.max_roll, -lateral_accel * p.tilt_strength))
        pitch = max(-p.max_pitch, min(p.max_pitch, forward_accel * p.tilt_strength))

        if p.micro_motion > 0:
            tt = frame / fps
            roll += math.sin(tt * 7.1 + 0.4) * p.micro_motion
            pitch += math.sin(tt * 5.7 + 2.0) * p.micro_motion
            yaw += math.sin(tt * 2.3 + 1.2) * p.micro_motion * 0.35

        rotations.append(Euler((pitch, roll, yaw), 'XYZ'))

    return rotations


def apply_flight(obj, sample_frames, positions, rotations, start, end):
    scene = bpy.context.scene

    # Replace only keys inside [start, end]; animation outside stays as it was.
    for fc in object_fcurves(obj):
        clear_range_keep_outside(fc, start, end)

    obj.rotation_mode = 'XYZ'
    for frame, pos, rot in zip(sample_frames, positions, rotations):
        obj.location = pos
        obj.rotation_euler = rot
        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    for fc in object_fcurves(obj):
        for kp in fc.keyframe_points:
            if start <= kp.co.x <= end:
                kp.interpolation = 'BEZIER'
                kp.handle_left_type = 'AUTO_CLAMPED'
                kp.handle_right_type = 'AUTO_CLAMPED'
        fc.update()

    scene.frame_start = start
    scene.frame_end = end
    scene.frame_set(start)
    return start, end


# ---------------------------------------------------------------------------
# KMZ / KML missions (DJI WPML waypoints or plain KML paths)
# ---------------------------------------------------------------------------

def _tag(el):
    return el.tag.rsplit('}', 1)[-1]


def _child(el, name):
    for c in el:
        if _tag(c) == name:
            return c
    return None


def _text(el, name, default=None):
    c = _child(el, name)
    if c is None or c.text is None or not c.text.strip():
        return default
    return c.text.strip()


def _float(el, name, default=None):
    value = _text(el, name)
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def _parse_coordinates(text):
    """'lon,lat[,alt] lon,lat[,alt] ...' -> [(lon, lat, alt or None), ...]"""
    points = []
    for item in text.split():
        parts = item.split(",")
        if len(parts) >= 2:
            alt = float(parts[2]) if len(parts) > 2 and parts[2] else None
            points.append((float(parts[0]), float(parts[1]), alt))
    return points


def read_mission_file(filepath):
    """
    Read waypoints from a .kmz / .kml / .wpml file.
    Returns (waypoints, info); waypoint = dict(lon, lat, height, speed, hover).
    """
    if zipfile.is_zipfile(filepath):
        with zipfile.ZipFile(filepath) as z:
            names = {n.lower(): n for n in z.namelist()}
            # DJI: waylines.wpml is the executed route, template.kml the planned one.
            name = names.get("wpmz/waylines.wpml") or names.get("wpmz/template.kml")
            if name is None:
                name = next((n for low, n in names.items()
                             if low.endswith((".kml", ".wpml"))), None)
            if name is None:
                raise RuntimeError("No KML/WPML file found inside the KMZ.")
            data = z.read(name)
    else:
        with open(filepath, "rb") as f:
            data = f.read()

    root = ET.fromstring(data)
    waypoints = []
    info = {"height_mode": "relativeToStartPoint"}

    for el in root.iter():
        if _tag(el) in ("heightMode", "executeHeightMode"):
            info["height_mode"] = el.text.strip()
            break

    # DJI WPML waypoints: Placemarks with a Point inside a Folder.
    for folder in (el for el in root.iter() if _tag(el) == "Folder"):
        global_speed = _float(folder, "autoFlightSpeed", 5.0)
        global_height = _float(folder, "globalHeight", 0.0)

        for pm in folder:
            if _tag(pm) != "Placemark":
                continue
            point = _child(pm, "Point")
            coords = _parse_coordinates(_text(point, "coordinates", "")) if point is not None else []
            if not coords:
                continue
            lon, lat, alt = coords[0]

            if _text(pm, "useGlobalHeight") == "1":
                height = global_height
            else:
                height = _float(pm, "executeHeight", _float(pm, "height", alt))
                if height is None:
                    height = global_height

            speed = global_speed
            waypoint_speed = _float(pm, "waypointSpeed")
            if waypoint_speed and _text(pm, "useGlobalSpeed") != "1":
                speed = waypoint_speed

            hover = 0.0
            for action in (a for a in pm.iter() if _tag(a) == "action"):
                if _text(action, "actionActuatorFunc") == "hover":
                    params = _child(action, "actionActuatorFuncParam")
                    if params is not None:
                        hover += _float(params, "hoverTime", 0.0)

            waypoints.append({"lon": lon, "lat": lat, "height": height,
                              "speed": max(0.1, speed), "hover": hover})

    # Plain KML (e.g. Google Earth path): LineString coordinates.
    if not waypoints:
        for line in (el for el in root.iter() if _tag(el) == "LineString"):
            for lon, lat, alt in _parse_coordinates(_text(line, "coordinates", "")):
                waypoints.append({"lon": lon, "lat": lat, "height": alt or 0.0,
                                  "speed": 5.0, "hover": 0.0})

    if len(waypoints) < 2:
        raise RuntimeError("The file must contain at least 2 waypoints.")
    return waypoints, info


def waypoints_to_local(waypoints):
    """WGS84 lon/lat -> local metres around the first waypoint (X = east, Y = north)."""
    a, e2 = 6378137.0, 6.69437999014e-3
    lat0 = math.radians(waypoints[0]["lat"])
    lon0 = waypoints[0]["lon"]
    s2 = math.sin(lat0) ** 2
    n = a / math.sqrt(1.0 - e2 * s2)                 # prime vertical radius
    m = a * (1.0 - e2) / (1.0 - e2 * s2) ** 1.5      # meridian radius

    return [
        Vector((
            math.radians(w["lon"] - lon0) * n * math.cos(lat0),
            math.radians(w["lat"] - waypoints[0]["lat"]) * m,
            w["height"],
        ))
        for w in waypoints
    ]


def mission_timeline(waypoints, points):
    """Times (s) and positions of the mission, hovers included."""
    times, positions = [0.0], [points[0]]
    t = 0.0
    for i in range(len(points)):
        if waypoints[i]["hover"] > 0:
            t += waypoints[i]["hover"]
            times.append(t)
            positions.append(points[i])
        if i + 1 < len(points):
            t += (points[i + 1] - points[i]).length / waypoints[i]["speed"]
            times.append(t)
            positions.append(points[i + 1])
    return times, positions


def generate_flight_from_mission(obj, p, filepath):
    scene = bpy.context.scene
    fps = scene.render.fps / scene.render.fps_base

    waypoints, info = read_mission_file(filepath)
    points = waypoints_to_local(waypoints)
    times, track = mission_timeline(waypoints, points)
    total_time = times[-1]
    if total_time <= 0:
        raise RuntimeError("Mission has zero length.")

    start = p.start_frame
    if p.kmz_fit_frames:
        end = start + max(2, p.frame_count) - 1
    else:
        end = start + max(1, round(total_time * fps))
    if end > MAX_FRAME:
        raise RuntimeError(f"Mission needs frames up to {end}, limit is {MAX_FRAME}.")
    time_scale = total_time / ((end - start) / fps)

    origin = scene.cursor.location.copy()   # takeoff point
    sample_frames = sample_frame_list(start, end, fps, p.samples_per_second)

    positions, velocities = [], []
    for idx, frame in enumerate(sample_frames):
        t = min(total_time, (frame - start) / fps * time_scale)
        k = max(1, min(len(times) - 1, bisect.bisect_left(times, t)))
        t0, t1 = times[k - 1], times[k]
        u = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
        pos = origin + track[k - 1].lerp(track[k], u)

        if idx:
            dt = (frame - sample_frames[idx - 1]) / fps
            vel = (pos - positions[-1]) / dt
        else:
            vel = Vector((0, 0, 0))
        positions.append(pos)
        velocities.append(vel)
    if len(velocities) > 1:
        velocities[0] = velocities[1].copy()

    # The mission is flown as is: no wind, tilt or micro motion, only heading.
    rotations = [Euler((0.0, 0.0, yaw), 'XYZ') for yaw in flight_heading(velocities)]
    apply_flight(obj, sample_frames, positions, rotations, start, end)

    length = sum((b - a).length for a, b in zip(points, points[1:]))
    return {
        "start": start, "end": end, "waypoints": len(waypoints),
        "length": length, "seconds": total_time,
        "height_mode": info["height_mode"],
    }


def world_to_camera_pixel(scene, camera, world_co):
    from bpy_extras.object_utils import world_to_camera_view

    scale = scene.render.resolution_percentage / 100.0
    width = scene.render.resolution_x * scale
    height = scene.render.resolution_y * scale

    ndc = world_to_camera_view(scene, camera, world_co)
    return ndc.x * width, (1.0 - ndc.y) * height, ndc.z


def projected_bounding_box(scene, camera, obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)

    corners = [obj_eval.matrix_world @ Vector(c) for c in obj_eval.bound_box]
    projected = [world_to_camera_pixel(scene, camera, c) for c in corners]

    if not projected:
        return None

    xs = [v[0] for v in projected]
    ys = [v[1] for v in projected]

    scale = scene.render.resolution_percentage / 100.0
    width = scene.render.resolution_x * scale
    height = scene.render.resolution_y * scale

    xmin = max(0.0, min(width, min(xs)))
    ymin = max(0.0, min(height, min(ys)))
    xmax = max(0.0, min(width, max(xs)))
    ymax = max(0.0, min(height, max(ys)))

    visible = any(depth > 0 for _, _, depth in projected)
    return xmin, ymin, xmax, ymax, visible


def projected_empty_bounding_box(scene, camera, obj):
    """
    Project an Empty's displayed cube bounds into image pixels.
    Returns (xmin, ymin, xmax, ymax, visible), or None.
    """
    if obj.type != 'EMPTY' or obj.empty_display_type != 'CUBE':
        return None

    depsgraph = bpy.context.evaluated_depsgraph_get()
    empty = obj.evaluated_get(depsgraph)

    # Empty cube display size extends equally in each local direction.
    half = obj.empty_display_size

    corners = [
        Vector((x * half, y * half, z * half))
        for x in (-1, 1)
        for y in (-1, 1)
        for z in (-1, 1)
    ]

    render = scene.render
    scale = render.resolution_percentage / 100.0
    width = int(render.resolution_x * scale)
    height = int(render.resolution_y * scale)

    projected = []

    for corner in corners:
        world_corner = empty.matrix_world @ corner
        ndc = world_to_camera_view(scene, camera, world_corner)

        if ndc.z > 0:
            projected.append((
                ndc.x * width,
                (1.0 - ndc.y) * height
            ))

    if not projected:
        return None

    xmin = min(p[0] for p in projected)
    ymin = min(p[1] for p in projected)
    xmax = max(p[0] for p in projected)
    ymax = max(p[1] for p in projected)

    visible = (
        xmax >= 0 and xmin <= width
        and ymax >= 0 and ymin <= height
    )

    return xmin, ymin, xmax, ymax, visible



def raycast_ground(scene, origin, max_distance, ground_collection=None):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    direction = Vector((0, 0, -1))

    # If a collection is selected, test only its objects.
    candidates = []
    if ground_collection:
        candidates = [o for o in ground_collection.all_objects
                      if o.type == 'MESH' and o.visible_get()]
    else:
        candidates = [o for o in scene.objects
                      if o.type == 'MESH' and o.visible_get()]

    best_distance = None
    best_point = None

    for obj in candidates:
        obj_eval = obj.evaluated_get(depsgraph)
        matrix_inv = obj_eval.matrix_world.inverted()
        local_origin = matrix_inv @ origin
        local_direction = (matrix_inv.to_3x3() @ direction).normalized()

        hit, location, normal, face_index = obj_eval.ray_cast(
            local_origin, local_direction, distance=max_distance
        )

        if hit:
            world_location = obj_eval.matrix_world @ location
            distance = (origin - world_location).length
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_point = world_location

    return best_point


def save_flight_csv(scene, obj, filepath, ground_collection, ray_distance, ground_offset):
    camera = scene.camera
    if camera is None:
        raise RuntimeError("Scene camera is not set.")

    start, end = scene.frame_start, scene.frame_end
    original_frame = scene.frame_current
    fps = scene.render.fps / scene.render.fps_base

    header = [
        "frame", "time_seconds",
        "x_m", "y_m", "z_m",
        "rotation_x_deg", "rotation_y_deg", "rotation_z_deg",
        "height_above_ground_m",
        "distance_to_camera_m",
        "bbox_xmin_px", "bbox_ymin_px",
        "bbox_xmax_px", "bbox_ymax_px",
        "bbox_width_px", "bbox_height_px",
        "bbox_visible"
    ]

    count = 0
    try:
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for frame in range(start, end + 1):
                scene.frame_set(frame)

                world_pos = obj.matrix_world.translation.copy()
                rot = obj.matrix_world.to_euler('XYZ')
                camera_pos = camera.matrix_world.translation.copy()
                camera_distance = (world_pos - camera_pos).length

                ground_point = raycast_ground(
                    scene, world_pos, ray_distance, ground_collection
                )

                if ground_point is None:
                    height = -1.0
                else:
                    height = max(0.0, world_pos.z - ground_point.z + ground_offset)

                bbox = projected_empty_bounding_box(scene, camera, obj)
                if bbox:
                    xmin, ymin, xmax, ymax, visible = bbox
                    bbox_width = max(0.0, xmax - xmin)
                    bbox_height = max(0.0, ymax - ymin)
                else:
                    xmin = ymin = xmax = ymax = 0.0
                    bbox_width = bbox_height = 0.0
                    visible = False

                writer.writerow([
                    frame,
                    frame / fps,
                    world_pos.x, world_pos.y, world_pos.z,
                    math.degrees(rot.x), math.degrees(rot.y), math.degrees(rot.z),
                    height,
                    camera_distance,
                    xmin, ymin, xmax, ymax,
                    bbox_width, bbox_height,
                    int(visible),
                ])
                count += 1
    finally:
        scene.frame_set(original_frame)

    return count


CSV_REQUIRED_COLUMNS = (
    "frame", "x_m", "y_m", "z_m",
    "rotation_x_deg", "rotation_y_deg", "rotation_z_deg",
)


def read_flight_csv(filepath):
    """
    Read a CSV written by save_flight_csv.
    Returns ([(frame, loc, euler), ...] sorted by frame, fps of the CSV or None).
    """
    with open(filepath, "r", newline="", encoding="utf-8-sig") as f:
        text = f.read()

    first_line = text.split("\n", 1)[0]
    # Excel with a comma-decimal locale saves ';'-separated files with ',' decimals.
    delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)

    columns = [c.strip() for c in (reader.fieldnames or [])]
    missing = [c for c in CSV_REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise RuntimeError(f"Missing CSV columns: {', '.join(missing)}")
    reader.fieldnames = columns

    def number(row, key):
        value = row[key].strip()
        if delimiter == ";":
            value = value.replace(",", ".")
        return float(value)

    rows = {}
    time_rows = []
    for line, row in enumerate(reader, start=2):
        if not row.get("frame", "").strip():
            continue
        try:
            frame = int(round(number(row, "frame")))
            loc = Vector((number(row, "x_m"), number(row, "y_m"), number(row, "z_m")))
            rot = Euler((
                math.radians(number(row, "rotation_x_deg")),
                math.radians(number(row, "rotation_y_deg")),
                math.radians(number(row, "rotation_z_deg")),
            ), 'XYZ')
            if row.get("time_seconds", "").strip():
                time_rows.append((frame, number(row, "time_seconds")))
        except (ValueError, TypeError, AttributeError) as exc:
            raise RuntimeError(f"Bad value on CSV line {line}: {exc}")
        rows[frame] = (loc, rot)

    if len(rows) < 2:
        raise RuntimeError("The CSV must contain at least 2 frames.")

    # Frame rate the CSV was saved with (frame / time_seconds), if available.
    csv_fps = None
    rates = [fr / t for fr, t in time_rows if t > 0]
    if rates:
        csv_fps = sum(rates) / len(rates)

    return [(fr, *rows[fr]) for fr in sorted(rows)], csv_fps


def load_flight_csv(obj, filepath, start_frame=None):
    """
    Recreate a flight from a CSV saved by this add-on.
    The CSV holds world-space values; they are converted to the object's
    local transform (parent included). If start_frame is given, the flight
    is shifted so its first row lands on that frame.
    """
    scene = bpy.context.scene
    rows, csv_fps = read_flight_csv(filepath)

    offset = 0 if start_frame is None else start_frame - rows[0][0]
    frames = [fr + offset for fr, _, _ in rows]
    start, end = frames[0], frames[-1]
    if start < 0 or end > MAX_FRAME:
        raise RuntimeError(f"Frames {start}-{end} are outside 0-{MAX_FRAME}.")

    scale = obj.matrix_world.to_scale()
    positions, rotations = [], []
    prev = obj.rotation_euler.copy()

    for (src_frame, loc, rot), frame in zip(rows, frames):
        world = Matrix.LocRotScale(loc, rot, scale)
        if obj.parent:
            scene.frame_set(frame)
            world = (obj.parent.matrix_world @ obj.matrix_parent_inverse).inverted() @ world
        local_loc, local_rot, _ = world.decompose()
        # Continuous Euler angles so interpolation never spins the long way round.
        euler = local_rot.to_euler('XYZ', prev)
        positions.append(local_loc)
        rotations.append(euler)
        prev = euler

    apply_flight(obj, frames, positions, rotations, start, end)
    return {"start": start, "end": end, "rows": len(rows), "csv_fps": csv_fps}


class DroneFlightProperties(PropertyGroup):
    start_frame: IntProperty(
        name="Start Frame", default=1, min=0, max=MAX_FRAME,
        description="First frame of the generated flight"
    )
    frame_count: IntProperty(
        name="Frames", default=720, min=2, max=MAX_FRAME,
        description="Number of generated frames, start frame included"
    )
    average_speed: FloatProperty(name="Average Speed", default=8.0, min=0.1, max=100.0, unit='VELOCITY')
    altitude_min: FloatProperty(name="Altitude Min", default=5.0, min=-1000.0, max=1000.0, unit='LENGTH')
    altitude_max: FloatProperty(name="Altitude Max", default=20.0, min=-1000.0, max=1000.0, unit='LENGTH')
    direction_randomness: FloatProperty(name="Direction Randomness", default=0.45, min=0.0, max=1.5)
    direction_change_seconds: FloatProperty(name="Direction Change", default=3.5, min=0.25, max=30.0, unit='TIME')
    wind_strength: FloatProperty(name="Wind", default=0.8, min=0.0, max=20.0, unit='VELOCITY')
    wind_vertical: FloatProperty(name="Vertical Wind", default=0.2, min=0.0, max=10.0, unit='VELOCITY')
    wind_frequency: FloatProperty(name="Wind Frequency", default=0.12, min=0.01, max=2.0)
    stabilization: FloatProperty(name="Stabilization", default=0.7, min=0.0, max=5.0)
    tilt_strength: FloatProperty(name="Tilt Strength", default=0.035, min=0.0, max=0.3)
    max_roll: FloatProperty(name="Max Roll", default=0.35, min=0.0, max=1.5, unit='ROTATION')
    max_pitch: FloatProperty(name="Max Pitch", default=0.25, min=0.0, max=1.5, unit='ROTATION')
    micro_motion: FloatProperty(name="Micro Motion", default=0.008, min=0.0, max=0.1)
    seed: IntProperty(name="Seed", default=42, min=0, max=999999999)
    samples_per_second: IntProperty(name="Samples / Second", default=4, min=1, max=24)

    kmz_filepath: StringProperty(
        name="KMZ File",
        description="DJI KMZ (WPML) mission or KML path to fly",
        subtype='FILE_PATH',
        default=""
    )
    kmz_fit_frames: BoolProperty(
        name="Fit To Frames",
        default=False,
        description="Stretch/compress the mission to exactly 'Frames' frames "
                    "instead of using the speed from the file"
    )

    csv_filepath: StringProperty(
        name="CSV File",
        description="Path for exported flight CSV",
        subtype='FILE_PATH',
        default="//drone_flight.csv"
    )
    csv_use_start_frame: BoolProperty(
        name="Load At Start Frame",
        default=False,
        description="Shift the loaded flight so it begins at 'Start Frame' "
                    "instead of the frames stored in the CSV"
    )
    ground_collection: PointerProperty(
        name="Ground Collection",
        type=bpy.types.Collection,
        description="Collection containing terrain/ground meshes used for height calculation"
    )
    ground_ray_distance: FloatProperty(
        name="Raycast Distance",
        default=1000.0,
        min=1.0,
        max=100000.0,
        unit='LENGTH',
        description="Maximum downward distance used to find the ground"
    )
    ground_offset: FloatProperty(
        name="Ground Offset",
        default=0.0,
        min=-100.0,
        max=100.0,
        unit='LENGTH',
        description="Offset added to height above the ground"
    )


class DRONE_OT_generate(Operator):
    bl_idname = "drone.generate_flight"
    bl_label = "Generate Flight"
    bl_description = "Generate a randomized realistic drone flight"

    def execute(self, context):
        obj = context.active_object
        p = context.scene.drone_flight_props

        if obj is None:
            self.report({'ERROR'}, "Select a drone object first.")
            return {'CANCELLED'}
        if p.altitude_max < p.altitude_min:
            self.report({'ERROR'}, "Altitude Max must be >= Altitude Min.")
            return {'CANCELLED'}

        if p.start_frame + p.frame_count - 1 > MAX_FRAME:
            self.report({'ERROR'}, f"Last frame must not exceed {MAX_FRAME}.")
            return {'CANCELLED'}

        start, end = generate_flight(obj, p)
        self.report({'INFO'}, f"Generated drone flight for frames {start}-{end}.")
        return {'FINISHED'}


class DRONE_OT_generate_kmz(Operator):
    bl_idname = "drone.generate_from_kmz"
    bl_label = "Fly KMZ Mission"
    bl_description = "Generate the flight from a KMZ/KML mission file (takeoff point = 3D cursor)"

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.kmz;*.kml;*.wpml", options={'HIDDEN'})

    def invoke(self, context, event):
        p = context.scene.drone_flight_props
        self.filepath = bpy.path.abspath(p.kmz_filepath)
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        obj = context.active_object
        p = context.scene.drone_flight_props

        if obj is None:
            self.report({'ERROR'}, "Select a drone object first.")
            return {'CANCELLED'}

        filepath = self.filepath or bpy.path.abspath(p.kmz_filepath)
        if not os.path.isfile(filepath):
            self.report({'ERROR'}, f"File not found: {filepath}")
            return {'CANCELLED'}

        try:
            r = generate_flight_from_mission(obj, p, filepath)
        except Exception as exc:
            self.report({'ERROR'}, f"KMZ flight failed: {exc}")
            return {'CANCELLED'}

        p.kmz_filepath = bpy.path.relpath(filepath) if bpy.data.filepath else filepath
        self.report({'INFO'}, (
            f"KMZ: {r['waypoints']} waypoints, {r['length']:.0f} m, {r['seconds']:.1f} s "
            f"-> frames {r['start']}-{r['end']} (heights: {r['height_mode']})"
        ))
        return {'FINISHED'}


class DRONE_OT_save_csv(Operator):
    bl_idname = "drone.save_csv"
    bl_label = "Save CSV"
    bl_description = "Export telemetry for every animation frame"

    filepath: StringProperty(subtype='FILE_PATH')

    def invoke(self, context, event):
        p = context.scene.drone_flight_props
        self.filepath = bpy.path.abspath(p.csv_filepath)
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        scene = context.scene
        obj = context.active_object
        p = scene.drone_flight_props

        if obj is None:
            self.report({'ERROR'}, "Select the drone object first.")
            return {'CANCELLED'}
        if scene.camera is None:
            self.report({'ERROR'}, "Set an active camera first.")
            return {'CANCELLED'}

        filepath = self.filepath or bpy.path.abspath(p.csv_filepath)
        if not filepath.lower().endswith(".csv"):
            filepath += ".csv"

        try:
            count = save_flight_csv(
                scene, obj, filepath,
                p.ground_collection,
                p.ground_ray_distance,
                p.ground_offset
            )
            p.csv_filepath = bpy.path.relpath(filepath)
            self.report({'INFO'}, f"Saved {count} frames to {filepath}")
        except Exception as exc:
            self.report({'ERROR'}, f"CSV export failed: {exc}")
            return {'CANCELLED'}

        return {'FINISHED'}


class DRONE_OT_load_csv(Operator):
    bl_idname = "drone.load_csv"
    bl_label = "Load CSV"
    bl_description = "Recreate a flight from a CSV saved by this add-on"

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.csv", options={'HIDDEN'})

    def invoke(self, context, event):
        p = context.scene.drone_flight_props
        self.filepath = bpy.path.abspath(p.csv_filepath)
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        scene = context.scene
        obj = context.active_object
        p = scene.drone_flight_props

        if obj is None:
            self.report({'ERROR'}, "Select the drone object first.")
            return {'CANCELLED'}

        filepath = self.filepath or bpy.path.abspath(p.csv_filepath)
        if not os.path.isfile(filepath):
            self.report({'ERROR'}, f"File not found: {filepath}")
            return {'CANCELLED'}

        try:
            r = load_flight_csv(
                obj, filepath,
                p.start_frame if p.csv_use_start_frame else None
            )
        except Exception as exc:
            self.report({'ERROR'}, f"CSV load failed: {exc}")
            return {'CANCELLED'}

        p.csv_filepath = bpy.path.relpath(filepath) if bpy.data.filepath else filepath
        fps = scene.render.fps / scene.render.fps_base
        if r["csv_fps"] and abs(r["csv_fps"] - fps) > 0.01:
            self.report({'WARNING'}, (
                f"Loaded {r['rows']} frames ({r['start']}-{r['end']}), but the CSV was "
                f"saved at {r['csv_fps']:.3g} fps and the scene is {fps:.3g} fps."
            ))
        else:
            self.report({'INFO'}, f"Loaded {r['rows']} frames ({r['start']}-{r['end']}) from CSV.")
        return {'FINISHED'}


class DRONE_OT_clear(Operator):
    bl_idname = "drone.clear_flight"
    bl_label = "Clear Flight"

    def execute(self, context):
        obj = context.active_object
        if obj is None:
            self.report({'ERROR'}, "Select the drone object first.")
            return {'CANCELLED'}
        obj.animation_data_clear()
        self.report({'INFO'}, "Flight animation cleared.")
        return {'FINISHED'}


class DRONE_PT_panel(Panel):
    bl_label = "Drone Flight Generator"
    bl_idname = "DRONE_PT_flight_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Drone Flight"

    def draw(self, context):
        layout = self.layout
        p = context.scene.drone_flight_props
        obj = context.active_object

        if obj:
            layout.label(text=f"Selected: {obj.name}", icon='OBJECT_DATA')
        else:
            layout.label(text="Select your drone object.", icon='ERROR')

        box = layout.box()
        box.label(text="Flight", icon='ANIM')
        box.prop(p, "start_frame")
        box.prop(p, "frame_count")
        fps = context.scene.render.fps / context.scene.render.fps_base
        box.label(
            text=f"Frames {p.start_frame} - {p.start_frame + p.frame_count - 1}"
                 f" ({p.frame_count / fps:.1f} s)",
            icon='TIME'
        )
        box.prop(p, "average_speed")
        box.prop(p, "altitude_min")
        box.prop(p, "altitude_max")
        box.prop(p, "direction_change_seconds")
        box.prop(p, "direction_randomness")

        box = layout.box()
        box.label(text="Environment", icon='FORCE_WIND')
        box.prop(p, "wind_strength")
        box.prop(p, "wind_vertical")
        box.prop(p, "wind_frequency")
        box.prop(p, "stabilization")

        box = layout.box()
        box.label(text="Drone Motion", icon='CONSTRAINT')
        box.prop(p, "tilt_strength")
        box.prop(p, "max_roll")
        box.prop(p, "max_pitch")
        box.prop(p, "micro_motion")

        box = layout.box()
        box.label(text="Generation", icon='MODIFIER')
        box.prop(p, "seed")
        box.prop(p, "samples_per_second")

        box = layout.box()
        box.label(text="KMZ Mission", icon='WORLD')
        box.prop(p, "kmz_filepath")
        box.prop(p, "kmz_fit_frames")
        box.label(text="Takeoff point = 3D cursor, +Y = north.", icon='INFO')
        box.label(text="Wind and Drone Motion are not applied.", icon='INFO')
        row = box.row()
        row.scale_y = 1.4
        row.operator("drone.generate_from_kmz", icon='IMPORT')

        box = layout.box()
        box.label(text="Ground / Terrain", icon='MESH_GRID')
        box.prop(p, "ground_collection")
        box.prop(p, "ground_ray_distance")
        box.prop(p, "ground_offset")
        box.label(text="CSV height uses downward raycast.", icon='INFO')

        box = layout.box()
        box.label(text="CSV Export", icon='FILE')
        box.prop(p, "csv_filepath")
        row = box.row()
        row.scale_y = 1.4
        row.operator("drone.save_csv", icon='FILE_TICK')
        row.operator("drone.load_csv", icon='FILE_FOLDER')
        box.prop(p, "csv_use_start_frame")

        row = layout.row()
        row.scale_y = 1.5
        row.operator("drone.generate_flight", icon='PLAY')

        row = layout.row()
        row.operator("drone.clear_flight", icon='TRASH')


classes = (
    DroneFlightProperties,
    DRONE_OT_generate,
    DRONE_OT_generate_kmz,
    DRONE_OT_save_csv,
    DRONE_OT_load_csv,
    DRONE_OT_clear,
    DRONE_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.drone_flight_props = PointerProperty(type=DroneFlightProperties)


def unregister():
    if hasattr(bpy.types.Scene, "drone_flight_props"):
        del bpy.types.Scene.drone_flight_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
