"""glow: compositor Glare (bloom / fog glow) + subtle colour lift on an input clip."""
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

src = movie_node(tree, inputs["clip"], args["frames"], (-800, 0))
glare_type = "BLOOM" if p["style"] == "bloom" else "FOG_GLOW"

# Bloom's mix is a blend, so use an explicit add for intensity: original + intensity * glow-only
glow_only = N.new("CompositorNodeGlare"); glow_only.location = (-450, -250)
glow_only.glare_type = glare_type
glow_only.quality = "HIGH"
glow_only.threshold = p["threshold"]
glow_only.mix = 1.0
glow_only.size = int(p["size"])
L.new(src.outputs["Image"], glow_only.inputs["Image"])
gain = N.new("CompositorNodeMixRGB"); gain.blend_type = "MULTIPLY"; gain.location = (-250, -250)
gain.inputs[0].default_value = 1.0
gain.inputs[2].default_value = (p["intensity"],) * 3 + (1.0,)
L.new(glow_only.outputs["Image"], gain.inputs[1])
add = N.new("CompositorNodeMixRGB"); add.blend_type = "ADD"; add.location = (-50, 0)
add.inputs[0].default_value = 1.0
L.new(src.outputs["Image"], add.inputs[1])
L.new(gain.outputs["Image"], add.inputs[2])

# subtle colour lift: warm highlights, slight lift in the blacks, mild contrast
cb = N.new("CompositorNodeColorBalance"); cb.location = (150, 0)
cb.correction_method = "LIFT_GAMMA_GAIN"
t = srgb_rgba(p["tint"])
cb.lift = (1.0 + 0.04 * p["color_lift"],) * 3
cb.gain = tuple(1.0 + p["color_lift"] * (c - 0.8) for c in t[:3])
L.new(add.outputs["Image"], cb.inputs["Image"])

# keep the source alpha (glow must not punch holes), but let the glow extend alpha where the clip is transparent
sa = N.new("CompositorNodeSetAlpha"); sa.location = (350, 0)
sa.mode = "REPLACE_ALPHA"
sep = N.new("CompositorNodeSepRGBA"); sep.location = (-250, 250)
mx = N.new("CompositorNodeMath"); mx.operation = "MAXIMUM"; mx.location = (150, 250)
sg = N.new("CompositorNodeSepRGBA"); sg.location = (-50, 250)
L.new(src.outputs["Image"], sep.inputs["Image"])
L.new(gain.outputs["Image"], sg.inputs["Image"])
lum = N.new("CompositorNodeMath"); lum.operation = "MAXIMUM"; lum.location = (50, 330)
lum2 = N.new("CompositorNodeMath"); lum2.operation = "MAXIMUM"; lum2.location = (50, 400)
L.new(sg.outputs["R"], lum.inputs[0]); L.new(sg.outputs["G"], lum.inputs[1])
L.new(lum.outputs["Value"], lum2.inputs[0]); L.new(sg.outputs["B"], lum2.inputs[1])
clampn = N.new("CompositorNodeMath"); clampn.operation = "MINIMUM"; clampn.location = (250, 400)
clampn.inputs[1].default_value = 1.0
L.new(lum2.outputs["Value"], clampn.inputs[0])
L.new(sep.outputs["A"], mx.inputs[0]); L.new(clampn.outputs["Value"], mx.inputs[1])
L.new(cb.outputs["Image"], sa.inputs["Image"])
L.new(mx.outputs["Value"], sa.inputs["Alpha"])

comp = N.new("CompositorNodeComposite"); comp.location = (550, 0)
comp.use_alpha = True
L.new(sa.outputs["Image"], comp.inputs["Image"])
finish_and_render(sc, args, "compositor")
