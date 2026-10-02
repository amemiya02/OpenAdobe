"""Helpers shared by the bpy comp scripts. Runs inside Blender (`blender -b --python-exit-code 1 -P <script> -- args.json`)."""
# ruff: noqa
import json
import os
import subprocess
import sys

import bpy


def load_args():
    argv = sys.argv[sys.argv.index("--") + 1:]
    with open(argv[0], encoding="utf-8") as fh:
        args = json.load(fh)
    # frames are rendered to container-local scratch (bind-mounted dirs are ~20x slower to write)
    args["frames_dir"] = os.path.join(os.environ.get("OPENADOBE_WORK", "/tmp"), "frames")
    return args


def hex_rgba(value, alpha=1.0):
    """'#RRGGBB' / [r,g,b] (0-1) -> (r, g, b, a) in *linear* space (Blender colour sockets are linear)."""
    if isinstance(value, str):
        v = value.lstrip("#")
        rgb = [int(v[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    else:
        rgb = [float(x) for x in value[:3]]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return (*lin, alpha)


def srgb_rgba(value, alpha=1.0):
    """Like hex_rgba but keeps the value as-is (for compositor nodes that work on display-referred pixels)."""
    return (*hex_rgba(value)[:3], alpha)


def reset_scene(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.data.scenes[0] if bpy.data.scenes else bpy.data.scenes.new("Scene")
    bpy.context.window.scene = sc if bpy.context.window else sc
    scale = 0.5 if args.get("preview") else 1.0
    r = sc.render
    r.resolution_x = max(2, int(args["width"] * scale) // 2 * 2)
    r.resolution_y = max(2, int(args["height"] * scale) // 2 * 2)
    r.resolution_percentage = 100
    r.fps = int(args["fps_num"])
    r.fps_base = int(args["fps_den"]) if args["fps_den"] else 1
    r.fps_base = args["fps_den"]
    sc.frame_start = 1
    sc.frame_end = int(args["frames"])
    sc.frame_set(1)
    r.film_transparent = True
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGBA"
    r.image_settings.color_depth = "16"
    r.image_settings.compression = 3
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.display_settings.display_device = "sRGB"
    r.filepath = os.path.join(args["frames_dir"], "f_#####")
    r.use_file_extension = True
    return sc


def setup_cycles(sc, args, samples=24, preview_samples=6):
    sc.render.engine = "CYCLES"
    c = sc.cycles
    c.device = "CPU"
    c.samples = preview_samples if args.get("preview") else samples
    c.use_adaptive_sampling = False
    c.use_denoising = False  # this Blender build has no OpenImageDenoise
    c.max_bounces = 4
    c.diffuse_bounces = 2
    c.glossy_bounces = 2
    c.transmission_bounces = 2
    c.sample_clamp_indirect = 2.0
    c.use_animated_seed = False
    c.film_exposure = 1.0
    c.caustics_reflective = False
    c.caustics_refractive = False
    c.use_light_tree = False
    sc.render.use_persistent_data = True
    sc.render.film_transparent = True
    sc.render.filter_size = 1.2
    sc.render.threads_mode = "AUTO"
    sc.render.use_compositing = False
    sc.render.use_sequencer = False
    return sc


def setup_compositor_only(sc):
    """Comp is a pure compositor tree; the 3D render is a trivial 1-sample pass."""
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 1
    sc.cycles.use_denoising = False
    sc.render.use_compositing = True
    sc.render.use_sequencer = False
    sc.use_nodes = True
    tree = sc.node_tree
    tree.nodes.clear()
    return tree


def _load_movie(path, frames):
    img = bpy.data.images.load(path)
    img.source = "MOVIE"
    img.alpha_mode = "STRAIGHT"
    return img


def movie_node(tree, path, frames, location=(0, 0)):
    """Compositor Image node playing an intermediate clip prepared at comp fps/size with exactly `frames` frames."""
    img = _load_movie(path, frames)
    node = tree.nodes.new("CompositorNodeImage")
    node.image = img
    node.frame_start = 1
    node.frame_duration = int(frames)
    node.frame_offset = 0
    node.use_cyclic = False
    node.use_auto_refresh = True
    node.location = location
    return node


def movie_texture(path, frames):
    """Image + ImageUser settings for an Image Texture shader node."""
    return _load_movie(path, frames)


def set_user_frames(tex_node, frames):
    tex_node.image_user.frame_start = 1
    tex_node.image_user.frame_duration = int(frames)
    tex_node.image_user.frame_offset = 0
    tex_node.image_user.use_cyclic = False
    tex_node.image_user.use_auto_refresh = True


def finish_and_render(sc, args, engine_label):
    os.makedirs(args["frames_dir"], exist_ok=True)
    try:
        bpy.ops.file.make_paths_relative()
    except RuntimeError:
        pass
    bpy.ops.wm.save_as_mainfile(filepath=args["blend"], compress=True, relative_remap=True)
    bpy.ops.render.render(animation=True)
    print("OPENADOBE_ENGINE", engine_label)


def font_path(name):
    if name and os.path.exists(name):
        return name
    try:
        out = subprocess.run(["fc-match", "-f", "%{file}", name or "Noto Sans CJK SC"], capture_output=True,
                             text=True, check=False).stdout.strip()
        if out and os.path.exists(out):
            return out
    except OSError:
        pass
    return None


def fit_border(sc, cam, objs, frames, margin=0.04, extra_points=None):
    """Render only the region the objects occupy over the camera move (path tracing without a denoiser is the
    cost driver). Union of projected bounding boxes at a few frames, plus a margin; the rest stays transparent."""
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    lo = Vector((1.0, 1.0))
    hi = Vector((0.0, 0.0))
    for f in sorted({1, max(1, frames // 4), max(1, frames // 2), max(1, 3 * frames // 4), frames}):
        sc.frame_set(f)
        bpy.context.view_layer.update()
        for ob in objs:
            for corner in ob.bound_box:
                co = world_to_camera_view(sc, cam, ob.matrix_world @ Vector(corner))
                lo.x, lo.y = min(lo.x, co.x), min(lo.y, co.y)
                hi.x, hi.y = max(hi.x, co.x), max(hi.y, co.y)
                for extra in (extra_points(ob.matrix_world @ Vector(corner)) if extra_points else []):
                    co = world_to_camera_view(sc, cam, extra)
                    lo.x, lo.y = min(lo.x, co.x), min(lo.y, co.y)
                    hi.x, hi.y = max(hi.x, co.x), max(hi.y, co.y)
    sc.frame_set(1)
    r = sc.render
    r.use_border = True
    r.use_crop_to_border = False
    r.border_min_x, r.border_min_y = max(0.0, lo.x - margin), max(0.0, lo.y - margin)
    r.border_max_x, r.border_max_y = min(1.0, hi.x + margin), min(1.0, hi.y + margin)
