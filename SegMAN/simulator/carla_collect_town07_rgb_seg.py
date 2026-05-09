"""
CARLA Town07 - RGB + Drivable/Non-Drivable Segmentation Collection
===================================================================
Spawns an ego vehicle with two synchronised cameras:
  1. RGB camera          → dataset/rgb/XXXXXXXX.png
  2. Semantic seg camera → dataset/seg/XXXXXXXX.png   (binary mask)
                         → dataset/seg_vis/XXXXXXXX.png (green overlay, debug)

Binary mask convention:
  255 = drivable   (CARLA tags: Road=1, RoadLine=24)
    0 = non-drivable (everything else)

The two cameras share the same transform so masks are pixel-aligned with RGB.

Requirements:
    pip install carla opencv-python numpy

Usage:
    python collect_town07.py --frames 5000 --out ./dataset
    python collect_town07.py --frames 5000 --out ./dataset --save-vis
"""

import carla
import cv2
import numpy as np
import argparse
import os
import time
import random
import queue

# CARLA semantic tag IDs that count as drivable surface
DRIVABLE_TAGS = {
    1,   # Road
    24,  # RoadLine
}


# ── CLI Arguments ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Collect paired RGB + binary segmentation in CARLA Town07")
    p.add_argument("--host",     default="127.0.0.1")
    p.add_argument("--port",     default=2000, type=int)
    p.add_argument("--frames",   default=5000, type=int, help="Pairs to capture")
    p.add_argument("--out",      default="./dataset", help="Root output directory")
    p.add_argument("--width",    default=1280, type=int)
    p.add_argument("--height",   default=720,  type=int)
    p.add_argument("--fov",      default=90,   type=float)
    p.add_argument("--fps",      default=10,   type=int)
    p.add_argument("--vehicles", default=30,   type=int)
    p.add_argument("--seed",     default=42,   type=int)
    p.add_argument("--save-vis", action="store_true",
                   help="Also save a green-overlay debug image for each frame")
    return p.parse_args()


# ── Camera Builders ───────────────────────────────────────────────────────────

MOUNT = carla.Transform(carla.Location(x=1.6, z=1.7))  # shared mount point

def build_camera(world, vehicle, sensor_type, width, height, fov):
    bp = world.get_blueprint_library().find(sensor_type)
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov",          str(fov))
    return world.spawn_actor(bp, MOUNT, attach_to=vehicle)


# ── Callbacks ─────────────────────────────────────────────────────────────────

def make_rgb_callback(q):
    def cb(image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        q.put((image.frame, arr[:, :, :3].copy()))  # drop alpha → BGR
    return cb


def make_seg_callback(q):
    """
    In CARLA's raw semantic segmentation image the semantic tag is stored
    in the Red channel (index 2 in BGRA layout).
    """
    def cb(image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        tag = arr[:, :, 2].copy()   # R channel = semantic tag ID
        q.put((image.frame, tag))
    return cb


def tag_to_binary_mask(tag_array):
    """HxW tag array → HxW uint8 binary mask (255=drivable, 0=non-drivable)."""
    mask = np.zeros(tag_array.shape, dtype=np.uint8)
    for t in DRIVABLE_TAGS:
        mask[tag_array == t] = 255
    return mask


def make_overlay(bgr, mask):
    """Semi-transparent green overlay on drivable pixels — for visual QC."""
    overlay    = bgr.copy()
    green      = np.zeros_like(bgr)
    green[:, :, 1] = 255
    drivable   = mask == 255
    overlay[drivable] = cv2.addWeighted(bgr, 0.5, green, 0.5, 0)[drivable]
    return overlay


# ── NPC Vehicles ──────────────────────────────────────────────────────────────

def spawn_npcs(client, world, n, rng):
    bp_lib = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    rng.shuffle(spawns)

    vehicle_bps = [
        bp for bp in bp_lib.filter("vehicle.*")
        if int(bp.get_attribute("number_of_wheels")) == 4
    ]

    tm = client.get_trafficmanager()
    tm.set_global_distance_to_leading_vehicle(2.0)

    actors = []
    for sp in spawns[:n]:
        bp = rng.choice(vehicle_bps)
        if bp.has_attribute("color"):
            bp.set_attribute("color", rng.choice(
                bp.get_attribute("color").recommended_values))
        actor = world.try_spawn_actor(bp, sp)
        if actor:
            actor.set_autopilot(True, tm.get_port())
            actors.append(actor)

    print(f"[NPC] Spawned {len(actors)} NPC vehicles.")
    return actors, tm


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rng  = random.Random(args.seed)

    # Output folders
    rgb_dir = os.path.join(args.out, "rgb")
    seg_dir = os.path.join(args.out, "seg")
    vis_dir = os.path.join(args.out, "seg_vis")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)

    print("[CARLA] Loading Town07 ...")
    world = client.load_world("Town07")
    time.sleep(2)

    # Synchronous fixed-step mode — required so both cameras fire on the same tick
    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)

    world.set_weather(carla.WeatherParameters(
        cloudiness=10.0,
        precipitation=0.0,
        sun_altitude_angle=60.0,
        fog_density=0.0,
    ))

    # Ego vehicle
    bp_lib     = world.get_blueprint_library()
    vehicle_bp = bp_lib.find("vehicle.tesla.model3")
    ego        = world.try_spawn_actor(vehicle_bp, random.choice(
                     world.get_map().get_spawn_points()))
    if ego is None:
        raise RuntimeError("Could not spawn ego vehicle.")
    print(f"[EGO] Spawned at {ego.get_location()}")

    # Two cameras at the identical mount point → pixel-aligned output
    rgb_q = queue.Queue(maxsize=200)
    seg_q = queue.Queue(maxsize=200)

    cam_rgb = build_camera(world, ego, "sensor.camera.rgb",
                           args.width, args.height, args.fov)
    cam_seg = build_camera(world, ego, "sensor.camera.semantic_segmentation",
                           args.width, args.height, args.fov)

    cam_rgb.listen(make_rgb_callback(rgb_q))
    cam_seg.listen(make_seg_callback(seg_q))

    # NPC traffic + ego autopilot
    npc_actors, tm = spawn_npcs(client, world, args.vehicles, rng)
    ego.set_autopilot(True, tm.get_port())
    tm.random_left_lanechange_percentage(ego, 20)
    tm.random_right_lanechange_percentage(ego, 20)
    tm.set_desired_speed(ego, 30)   # km/h

    print(f"[COLLECT] Capturing {args.frames} paired frames → {args.out}")
    print(f"  Mask legend: 255 = drivable (road + road lines)  |  0 = non-drivable")

    saved   = 0
    rgb_buf = {}   # frame_id → bgr array
    seg_buf = {}   # frame_id → tag array

    try:
        while saved < args.frames:
            world.tick()

            # Drain both queues into per-frame buffers
            while not rgb_q.empty():
                fid, bgr = rgb_q.get_nowait()
                rgb_buf[fid] = bgr
            while not seg_q.empty():
                fid, tag = seg_q.get_nowait()
                seg_buf[fid] = tag

            # Write frames where both cameras have delivered data
            for fid in sorted(set(rgb_buf) & set(seg_buf)):
                bgr  = rgb_buf.pop(fid)
                tag  = seg_buf.pop(fid)
                mask = tag_to_binary_mask(tag)

                cv2.imwrite(os.path.join(rgb_dir, f"{fid:08d}.png"), bgr)
                cv2.imwrite(os.path.join(seg_dir, f"{fid:08d}.png"), mask)

                if args.save_vis:
                    cv2.imwrite(os.path.join(vis_dir, f"{fid:08d}.png"),
                                make_overlay(bgr, mask))
                saved += 1
                if saved >= args.frames:
                    break

            if saved > 0 and saved % 100 == 0:
                print(f"  Saved {saved}/{args.frames} pairs")

    except KeyboardInterrupt:
        print("\n[COLLECT] Interrupted by user.")

    finally:
        print("[CLEANUP] Destroying actors ...")
        cam_rgb.stop(); cam_rgb.destroy()
        cam_seg.stop(); cam_seg.destroy()
        ego.destroy()
        for npc in npc_actors:
            npc.destroy()

        settings.synchronous_mode    = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        print(f"\n[DONE] {saved} pairs saved.")
        print(f"  RGB frames  → {rgb_dir}")
        print(f"  Seg masks   → {seg_dir}")
        if args.save_vis:
            print(f"  Overlays    → {vis_dir}")


if __name__ == "__main__":
    main()
