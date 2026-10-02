"""title_3d: extruded + bevelled 3D text (CJK ok), key/rim lights, slow camera push, transparent background (Cycles)."""
# ruff: noqa
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
from common import *
from mathutils import Vector

args = load_args()
p = args["params"]
sc = reset_scene(args)
setup_cycles(sc, args, samples=p["samples"], preview_samples=max(2, p["samples"] // 3))
frames = int(args["frames"])
aspect = args["width"] / args["height"]

fpath = font_path(p["font"])
font = bpy.data.fonts.load(fpath) if fpath else bpy.data.fonts["Bfont Regular"]
print("OPENADOBE_FONT", fpath)


def make_material(name, color, metallic, roughness=0.32):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = hex_rgba(color)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Coat Weight"].default_value = 0.0 if metallic > 0.5 else 0.4
    bsdf.inputs["Coat Roughness"].default_value = 0.15
    return mat


def make_text(body, size, color, extrude, bevel, name):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.font = font
    for attr in ("font_bold", "font_italic", "font_bold_italic"):
        setattr(cu, attr, font)
    cu.size = size
    cu.align_x = "CENTER"
    cu.align_y = "CENTER"
    cu.extrude = extrude
    cu.bevel_depth = bevel
    cu.bevel_resolution = 3
    cu.resolution_u = 6
    ob = bpy.data.objects.new(name, cu)
    sc.collection.objects.link(ob)
    ob.data.materials.append(make_material(name + "_mat", color, p["metallic"] if name == "title" else 0.0))
    return ob


def extents(ob):
    bpy.context.view_layer.update()
    xs = [v[0] for v in ob.bound_box]
    ys = [v[1] for v in ob.bound_box]
    return min(xs), max(xs), min(ys), max(ys)


H = 1.0  # title font size = 1 unit
title = make_text(p["text"], H, p["text_color"], p["extrude"] * H * 0.5, p["bevel"] * H, "title")
tx0, tx1, ty0, ty1 = extents(title)
rows = [(title, tx1 - tx0, ty1 - ty0)]
sub = None
if p["subtitle"].strip():
    sub = make_text(p["subtitle"], H * 0.36, p["subtitle_color"], p["extrude"] * H * 0.18, p["bevel"] * H * 0.5, "subtitle")
    sx0, sx1, sy0, sy1 = extents(sub)
    gap = 0.12 * H
    # title centre at 0, subtitle below
    sub.location.y = ty0 - gap - sy1
    rows.append((sub, sx1 - sx0, sy1 - sy0))
top = ty1
bottom = (sub.location.y + extents(sub)[2]) if sub else ty0
cy = (top + bottom) / 2
for ob, _, _ in rows:
    ob.location.y -= cy
width = max(w for _, w, _ in rows)
height = top - bottom

# Group stands up in XZ plane, facing -Y (towards the camera at -Y)
root = bpy.data.objects.new("title_root", None)
sc.collection.objects.link(root)
root.rotation_euler = (math.radians(90), 0, 0)
for ob, _, _ in rows:
    ob.parent = root

cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 50
cam_data.sensor_fit = "HORIZONTAL"
cam_data.sensor_width = 36
cam = bpy.data.objects.new("cam", cam_data)
sc.collection.objects.link(cam)
sc.camera = cam
target = bpy.data.objects.new("target", None)
sc.collection.objects.link(target)
trk = cam.constraints.new("TRACK_TO")
trk.target = target
trk.track_axis = "TRACK_NEGATIVE_Z"
trk.up_axis = "UP_Y"

visible_w = width / p["text_width"]
dist0 = visible_w * 50 / 36
# keep tall text inside the frame too
dist0 = max(dist0, (height / 0.8) * 50 / 36 * aspect)
dist1 = dist0 * (1.0 - p["camera_push"])
orbit = math.radians(7)  # slow arc that reveals the extrusion
for f, (d, a) in ((1, (dist0, -orbit)), (frames, (dist1, orbit * 0.3))):
    cam.location = (d * math.sin(a), -d * math.cos(a), 0.04 * d)
    cam.keyframe_insert("location", frame=f)
if frames > 1:
    for fc in cam.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "BEZIER"
            kp.handle_left_type = kp.handle_right_type = "AUTO_CLAMPED"

# lights: low-noise sun lamps + soft world fill, coloured rim from behind. A sun shines along its local -Z.
def sun(name, energy, from_pos, color=(1, 1, 1), angle=6.0):
    d = bpy.data.lights.new(name, "SUN")
    d.energy = energy
    d.color = color
    d.angle = math.radians(angle)
    ob = bpy.data.objects.new(name, d)
    ob.rotation_euler = (-Vector(from_pos)).to_track_quat("-Z", "Y").to_euler()
    sc.collection.objects.link(ob)
    return ob


# camera sits at -Y looking +Y; text faces -Y
sun("key", 3.2, (-0.7, -1.0, 0.9), (1.0, 0.97, 0.92), 8)          # front, upper left
sun("fill", 0.8, (1.0, -1.0, -0.2), (0.9, 0.95, 1.0), 30)          # front, lower right, broad
sun("rim", 3.5, (0.9, 0.9, 0.6), hex_rgba(p["rim_color"])[:3], 10)  # behind, upper right: coloured edge light

world = bpy.data.worlds.new("w")
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.55, 0.58, 0.64, 1)
bg.inputs["Strength"].default_value = 0.35
sc.world = world

fit_border(sc, cam, [ob for ob, _, _ in rows], frames)
sc.cycles.use_adaptive_sampling = True
sc.cycles.adaptive_threshold = 0.03
sc.cycles.adaptive_min_samples = max(2, p["samples"] // 4)
sc.cycles.max_bounces = 3
sc.cycles.glossy_bounces = 2
sc.cycles.transmission_bounces = 0
finish_and_render(sc, args, "CYCLES")
