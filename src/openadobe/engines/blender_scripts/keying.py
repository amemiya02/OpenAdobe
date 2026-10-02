"""keying: green/blue-screen fg (+ optional bg plate) -> compositor Keying node (+ despill, garbage matte)."""
# ruff: noqa
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
from common import *

args = load_args()
p = args["params"]
sc = reset_scene(args)
tree = setup_compositor_only(sc)
N = tree.nodes
L = tree.links
inputs = args["inputs"]

fg = movie_node(tree, inputs["fg"], args["frames"], (-900, 200))
key = N.new("CompositorNodeKeying")
key.location = (-500, 200)
key.clip_black = p["clip_black"]
key.clip_white = p["clip_white"]
key.despill_factor = p["despill"]
key.despill_balance = 0.5
key.blur_post = int(round(p["edge_blur"]))
key.blur_pre = 0
key.dilate_distance = int(p["edge_grow"])
key.edge_kernel_radius = 3
key.edge_kernel_tolerance = 0.1
key.feather_falloff = "SMOOTH"
key.feather_distance = 0
key.inputs["Key Color"].default_value = srgb_rgba(p["key_color"])
L.new(fg.outputs["Image"], key.inputs["Image"])
cur = key.outputs["Image"]

crop = p["garbage"]  # [left, right, top, bottom] fractions of frame removed
if any(c > 0 for c in crop):
    # box mask covering the *kept* region; multiply keyed alpha by it
    mask = N.new("CompositorNodeBoxMask")
    mask.location = (-500, -100)
    l, r, t, b = crop
    mask.mask_type = "ADD"
    mask.x = (l + (1 - r)) / 2
    mask.y = (b + (1 - t)) / 2
    mask.mask_width = max(0.001, 1 - l - r)
    mask.mask_height = max(0.001, 1 - t - b)
    mask.rotation = 0
    mask.inputs["Mask"].default_value = 0.0
    mask.inputs["Value"].default_value = 1.0
    sep = N.new("CompositorNodeSepRGBA"); sep.location = (-300, 200)
    mul = N.new("CompositorNodeMath"); mul.operation = "MULTIPLY"; mul.location = (-150, 100)
    comb = N.new("CompositorNodeCombRGBA"); comb.location = (50, 200)
    L.new(cur, sep.inputs["Image"])
    L.new(sep.outputs["A"], mul.inputs[0])
    L.new(mask.outputs["Mask"], mul.inputs[1])
    for ch in "RGB":
        L.new(sep.outputs[ch], comb.inputs[ch])
    L.new(mul.outputs["Value"], comb.inputs["A"])
    cur = comb.outputs["Image"]

if "bg" in inputs:
    bg = movie_node(tree, inputs["bg"], args["frames"], (-900, -300))
    ao = N.new("CompositorNodeAlphaOver"); ao.location = (300, 100)
    ao.use_premultiply = True  # fg is straight alpha from the keyer
    L.new(bg.outputs["Image"], ao.inputs[1])
    L.new(cur, ao.inputs[2])
    cur = ao.outputs["Image"]

comp = N.new("CompositorNodeComposite"); comp.location = (600, 100)
comp.use_alpha = True
L.new(cur, comp.inputs["Image"])
finish_and_render(sc, args, "compositor")
