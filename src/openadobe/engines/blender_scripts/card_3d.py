"""card_3d: input clip on a rounded 3D card swinging/spinning in perspective, soft drop shadow + reflection (Cycles)."""
# ruff: noqa
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bmesh
import bpy
from common import *
from mathutils import Vector

args = load_args()
p = args["params"]
sc = reset_scene(args)
setup_cycles(sc, args, samples=12, preview_samples=4)
frames = int(args["frames"])
aspect_frame = args["width"] / args["height"]

img = movie_texture(args["inputs"]["clip"], frames)
iw, ih = img.size[0] or 16, img.size[1] or 9
CH = 1.0
CW = CH * iw / ih  # card height 1 unit, width from the clip aspect


def rounded_card_mesh(name, w, h, radius, seg=10):
    r = min(radius, w / 2, h / 2)
    pts = []
    if r <= 1e-6:
        pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    else:
        for cx, cy, a0 in ((w / 2 - r, -h / 2 + r, -90), (w / 2 - r, h / 2 - r, 0),
                           (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180)):
            for i in range(seg + 1):
                a = math.radians(a0 + 90 * i / seg)
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    bm = bmesh.new()
    verts = [bm.verts.new((x, y, 0)) for x, y in pts]
    face = bm.faces.new(verts)
    uv = bm.loops.layers.uv.new("UVMap")
    for loop in face.loops:
        x, y = loop.vert.co.x, loop.vert.co.y
        loop[uv].uv = ((x + w / 2) / w, (y + h / 2) / h)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


def face_material(name, reflect=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Linear"
    tex.extension = "EXTEND"
    set_user_frames(tex, frames)
    emi = nt.nodes.new("ShaderNodeEmission")
    emi.inputs["Strength"].default_value = 1.0
    trn = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(tex.outputs["Color"], emi.inputs["Color"])
    if reflect > 0:
        uvn = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        inv = nt.nodes.new("ShaderNodeMath"); inv.operation = "SUBTRACT"; inv.inputs[0].default_value = 1.0
        pw = nt.nodes.new("ShaderNodeMath"); pw.operation = "POWER"; pw.inputs[1].default_value = 2.0
        sc1 = nt.nodes.new("ShaderNodeMath"); sc1.operation = "MULTIPLY"; sc1.inputs[1].default_value = reflect * 0.6
        m2 = nt.nodes.new("ShaderNodeMath"); m2.operation = "MULTIPLY"
        nt.links.new(uvn.outputs["UV"], sep.inputs["Vector"])
        nt.links.new(sep.outputs["Y"], inv.inputs[1])
        nt.links.new(inv.outputs["Value"], pw.inputs[0])
        nt.links.new(pw.outputs["Value"], sc1.inputs[0])
        nt.links.new(sc1.outputs["Value"], m2.inputs[0])
        nt.links.new(tex.outputs["Alpha"], m2.inputs[1])
        nt.links.new(m2.outputs["Value"], mix.inputs["Fac"])
    else:
        nt.links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
    nt.links.new(trn.outputs["BSDF"], mix.inputs[1])
    nt.links.new(emi.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


def edge_material():
    mat = bpy.data.materials.new("edge")
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = hex_rgba(p["edge_color"])
    b.inputs["Roughness"].default_value = 0.4
    return mat


radius = p["corner_radius"] * min(CW, CH)
pivot = bpy.data.objects.new("pivot", None)
sc.collection.objects.link(pivot)
root = bpy.data.objects.new("root", None)
sc.collection.objects.link(root)
root.parent = pivot
root.rotation_euler = (math.radians(90), 0, 0)  # card plane stands vertical, front faces -Y

card = bpy.data.objects.new("card", rounded_card_mesh("card", CW, CH, radius))
sc.collection.objects.link(card)
card.parent = root
card.data.materials.append(face_material("card_face"))
card.data.materials.append(edge_material())
sol = card.modifiers.new("thick", "SOLIDIFY")
sol.thickness = 0.03
sol.offset = 0.0
sol.material_offset_rim = 1

reflect = p["reflection"] if p["direction"] in ("left", "right") else 0.0
parts = [card]
if reflect > 0:
    refl = bpy.data.objects.new("reflection", rounded_card_mesh("reflection", CW, CH, radius))
    sc.collection.objects.link(refl)
    refl.parent = root
    refl.scale = (1, -1, 1)
    refl.location = (0, -(CH + 0.04), 0)
    refl.data.materials.append(face_material("reflection_face", reflect))
    parts.append(refl)

# motion
A = math.radians(p["rotation"])
axis, sign = {"right": (2, 1), "left": (2, -1), "up": (0, 1), "down": (0, -1)}[p["direction"]]
if p["motion"] == "swing":
    poses = ((1, -A * sign), (frames, A * sign))
elif p["motion"] == "spin":
    poses = ((1, 0.0), (frames + 1, 2 * math.pi * sign))
else:
    poses = ((1, A * sign), (frames, A * sign))
for f, ang in poses:
    e = [0.0, 0.0, 0.0]
    e[axis] = ang
    pivot.rotation_euler = e
    pivot.keyframe_insert("rotation_euler", frame=f)
if pivot.animation_data and p["motion"] != "static":
    for fc in pivot.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR" if p["motion"] == "spin" else "SINE"
            kp.easing = "EASE_IN_OUT" if p["motion"] == "swing" else "AUTO"

# camera: fixed, looking +Y
cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 50
cam_data.sensor_fit = "HORIZONTAL"
cam_data.sensor_width = 36
cam = bpy.data.objects.new("cam", cam_data)
sc.collection.objects.link(cam)
sc.camera = cam
fit_w = CW / p["card_scale"]
fit_h = (CH + (CH + 0.04 if reflect > 0 else 0)) / 0.9 * aspect_frame
dist = max(fit_w, fit_h * 1.0) * 50 / 36
cam.location = (0, -dist, -(0.5 * (CH + 0.04) if reflect > 0 else 0))
cam.rotation_euler = (math.radians(90), 0, 0)
if reflect > 0:  # frame card + reflection: centre between them
    cam.location.z = -0.5 * (CH + 0.04) * 0.5 - 0.0

# soft drop shadow: made in the compositor from the card's own alpha (noise-free; a path-traced shadow catcher
# cost ~5x more render time without a denoiser)
d = bpy.data.lights.new("key", "SUN")
d.energy = 3.0
d.angle = math.radians(14)
key = bpy.data.objects.new("key", d)
key.rotation_euler = (-Vector((-0.28, -1.0, 0.34))).to_track_quat("-Z", "Y").to_euler()
sc.collection.objects.link(key)
world = bpy.data.worlds.new("w")
world.use_nodes = True
bgn = world.node_tree.nodes["Background"]
bgn.inputs["Color"].default_value = (1, 1, 1, 1)
bgn.inputs["Strength"].default_value = 0.4
sc.world = world

# the border keeps the card and its reflection; the compositor shadow lives inside the margin
fit_border(sc, cam, parts, frames, margin=0.10 if p["shadow"] > 0 else 0.02)

sc.render.use_compositing = True
sc.use_nodes = True
tree = sc.node_tree
tree.nodes.clear()
N, L = tree.nodes, tree.links
rl = N.new("CompositorNodeRLayers")
comp = N.new("CompositorNodeComposite")
comp.use_alpha = True
if p["shadow"] > 0:
    W_px = sc.render.resolution_x
    sep = N.new("CompositorNodeSepRGBA")
    ramp = N.new("CompositorNodeValToRGB")  # keep the card, drop the faint reflection from the shadow source
    ramp.color_ramp.elements[0].position = 0.45
    ramp.color_ramp.elements[1].position = 0.85
    off = N.new("CompositorNodeTranslate")
    off.use_relative = False
    off.inputs["X"].default_value = 0.022 * W_px
    off.inputs["Y"].default_value = -0.032 * W_px
    blur = N.new("CompositorNodeBlur")
    blur.filter_type = "GAUSS"
    blur.use_relative = False
    blur.size_x = blur.size_y = max(2, int(0.022 * W_px))
    blk = N.new("CompositorNodeRGB")
    blk.outputs[0].default_value = (0, 0, 0, 1)
    sa = N.new("CompositorNodeSetAlpha")
    sa.mode = "REPLACE_ALPHA"
    gain = N.new("CompositorNodeMath")
    gain.operation = "MULTIPLY"
    gain.inputs[1].default_value = 0.7 * p["shadow"]
    ao = N.new("CompositorNodeAlphaOver")
    L.new(rl.outputs["Alpha"], ramp.inputs["Fac"])
    L.new(ramp.outputs["Image"], sep.inputs["Image"])
    L.new(sep.outputs["R"], off.inputs["Image"])
    L.new(off.outputs["Image"], blur.inputs["Image"])
    L.new(blur.outputs["Image"], gain.inputs[0])
    L.new(blk.outputs[0], sa.inputs["Image"])
    L.new(gain.outputs["Value"], sa.inputs["Alpha"])
    L.new(sa.outputs["Image"], ao.inputs[1])
    L.new(rl.outputs["Image"], ao.inputs[2])
    L.new(ao.outputs["Image"], comp.inputs["Image"])
else:
    L.new(rl.outputs["Image"], comp.inputs["Image"])
sc.cycles.use_adaptive_sampling = True
sc.cycles.adaptive_threshold = 0.03
sc.cycles.adaptive_min_samples = 4
sc.cycles.max_bounces = 2
sc.cycles.diffuse_bounces = 1
sc.cycles.glossy_bounces = 0
sc.cycles.transmission_bounces = 0
sc.cycles.transparent_max_bounces = 4
finish_and_render(sc, args, "CYCLES")
