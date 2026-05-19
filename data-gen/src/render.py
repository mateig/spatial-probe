"""Render stage-0 style spatial QA scenes with Blender."""

import json
import math
import os
import random
import time

import bpy
import numpy as np
from mathutils import Matrix
from PIL import Image

from src import description, qa, relations, sample


def _look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    e = np.asarray(eye, dtype=float)
    fwd = np.asarray(target, dtype=float) - e
    fwd /= np.linalg.norm(fwd)
    up_ = np.asarray(up, dtype=float)
    right = np.cross(fwd, up_)
    if np.linalg.norm(right) < 1e-8:
        up_ = np.array((0.0, 1.0, 0.0))
        right = np.cross(fwd, up_)
    right /= np.linalg.norm(right)
    M = np.eye(4)
    M[:3, 0] = right
    M[:3, 1] = np.cross(right, fwd)
    M[:3, 2] = -fwd
    M[:3, 3] = e
    return M


def _material(name, color=(0.55, 0.55, 0.6), roughness=0.5, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def _setup_render(cfg):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    scene.render.engine = "CYCLES"
    scene.cycles.samples = cfg.get("samples", 16)
    scene.cycles.use_denoising = False

    w, h = cfg["resolution"]
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    return scene


def _setup_world(scene):
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Strength"].default_value = 0.8
    bg.inputs["Color"].default_value = (0.08, 0.08, 0.09, 1.0)
    scene.world = world
    return bg


def _setup_compositor(scene):
    scene.use_nodes = True
    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True
    view_layer.use_pass_object_index = True
    view_layer.update_render_passes()

    tree = scene.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers")
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(_output_socket(rl, "Image"), composite.inputs["Image"])

    depth = tree.nodes.new("CompositorNodeOutputFile")
    depth.format.file_format = "OPEN_EXR"
    depth.format.color_depth = "32"
    depth.file_slots[0].path = "depth_"
    tree.links.new(_output_socket(rl, "Depth"), depth.inputs[0])

    seg = tree.nodes.new("CompositorNodeOutputFile")
    seg.format.file_format = "OPEN_EXR"
    seg.format.color_depth = "32"
    seg.file_slots[0].path = "seg_"
    tree.links.new(_output_socket(rl, "IndexOB"), seg.inputs[0])
    return depth, seg


def _output_socket(node, name: str):
    for socket in node.outputs:
        if socket.name == name or socket.identifier == name:
            return socket
    names = ", ".join(socket.name for socket in node.outputs)
    raise KeyError(f"output socket {name!r} not found; available: {names}")


def _add_camera(scene):
    data = bpy.data.cameras.new("Camera")
    data.lens_unit = "FOV"
    data.sensor_fit = "HORIZONTAL"
    data.angle = math.radians(60.0)
    obj = bpy.data.objects.new("Camera", data)
    scene.collection.objects.link(obj)
    scene.camera = obj
    return obj


def _add_sun(scene):
    light = bpy.data.lights.new("Sun", "SUN")
    light.energy = 2.0
    light.angle = math.radians(8.0)
    obj = bpy.data.objects.new("Sun", light)
    obj.matrix_world = Matrix(_look_at((0, 0, 0), (0.4, 0.6, -1.0)).tolist())
    scene.collection.objects.link(obj)


def _add_area_light(scene, name, location, target, energy, size):
    light = bpy.data.lights.new(name, "AREA")
    light.energy = energy
    light.size = size
    obj = bpy.data.objects.new(name, light)
    obj.location = location
    obj.matrix_world = Matrix(_look_at(location, target).tolist())
    scene.collection.objects.link(obj)


def _add_soft_lighting(scene):
    _add_sun(scene)
    _add_area_light(
        scene,
        "Key_Area",
        location=(-2.5, -3.0, 4.0),
        target=(0.0, 0.0, 0.8),
        energy=260.0,
        size=5.0,
    )
    _add_area_light(
        scene,
        "Fill_Area",
        location=(3.0, 2.5, 3.0),
        target=(0.0, 0.0, 0.7),
        energy=120.0,
        size=6.0,
    )
    _add_area_light(
        scene,
        "Top_Area",
        location=(0.0, 0.0, 5.0),
        target=(0.0, 0.0, 0.6),
        energy=90.0,
        size=7.0,
    )


def _add_ground(scene, size=20.0):
    s = size / 2
    mesh = bpy.data.meshes.new("Ground")
    mesh.from_pydata(
        [(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)]
    )
    mesh.update()
    mesh.materials.append(_material("GroundMat", (0.35, 0.35, 0.38), roughness=0.9))
    obj = bpy.data.objects.new("Ground", mesh)
    obj.location = (0, 0, -0.005)
    obj.pass_index = 0
    scene.collection.objects.link(obj)


def _add_object_pool(scene, n):
    objs = []
    for i in range(n):
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.object
        obj.name = f"object_{i}"
        obj.data.materials.append(_material(f"object_{i}_mat", roughness=0.55))
        obj.pass_index = i + 1
        objs.append(obj)
    return objs


def _apply_shape(obj, shape):
    if obj.get("shape") == shape:
        return
    mat = obj.data.materials[0] if obj.data.materials else None
    if shape == "cube":
        bpy.ops.mesh.primitive_cube_add(size=1.0)
    elif shape == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=0.5)
    elif shape == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.5, depth=1.0)
    else:
        raise ValueError(shape)
    new_mesh = bpy.context.object.data
    new_mesh.name = f"{shape}_mesh"
    bpy.data.objects.remove(bpy.context.object, do_unlink=True)
    old = obj.data
    obj.data = new_mesh
    if mat is not None:
        obj.data.materials.append(mat)
    obj["shape"] = shape
    if old.users == 0:
        bpy.data.meshes.remove(old)


def _set_object(obj, spec: sample.Object):
    _apply_shape(obj, spec.shape)
    s = spec.half_extent
    obj.hide_render = False
    obj.hide_viewport = False
    obj.location = spec.position
    obj.rotation_euler = (0.0, 0.0, 0.0)
    if spec.shape == "cube":
        obj.scale = (2.0 * s, 2.0 * s, 2.0 * s)
    elif spec.shape == "sphere":
        obj.scale = (2.0 * s, 2.0 * s, 2.0 * s)
    else:
        obj.scale = (2.0 * s, 2.0 * s, 2.0 * s)
    obj.data.materials[0].node_tree.nodes["Principled BSDF"].inputs[
        "Base Color"
    ].default_value = (*spec.color_rgb, 1.0)


def _clipped_depth(depth: np.ndarray, clip: float) -> np.ndarray:
    d = np.where(np.isfinite(depth), depth, clip)
    return np.clip(d, 0.0, clip).astype(np.float32)


def _save_depth_png(depth: np.ndarray, path: str, clip: float) -> None:
    d = _clipped_depth(depth, clip)
    mask = d < clip
    if mask.any():
        lo, hi = float(d[mask].min()), float(d[mask].max())
    else:
        lo, hi = 0.0, clip
    span = max(hi - lo, 1e-6)
    norm = np.clip((d - lo) / span, 0.0, 1.0)
    Image.fromarray((norm * 255.0).astype(np.uint8)).save(path)


def _save_segmentation(seg_float: np.ndarray, path: str) -> None:
    seg = np.rint(seg_float).astype(np.int32)
    rng = np.random.default_rng(0)
    n = int(max(seg.max(), 0)) + 2
    palette = rng.integers(0, 255, size=(n, 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)
    img = palette[np.clip(seg, 0, n - 1)]
    Image.fromarray(img.astype(np.uint8)).save(path)


def _load_exr(path: str, w: int, h: int) -> np.ndarray:
    img = bpy.data.images.load(path)
    arr = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    bpy.data.images.remove(img)
    os.remove(path)
    return np.flipud(arr[:, :, 0])


def _scene_json(scene: sample.Scene, rels: list[relations.PairRelation], cam_T, intrinsics):
    depths = relations.object_depths(scene)
    objects = []
    for o in scene.objects:
        d = depths[o.id]
        objects.append(
            {
                "id": o.id,
                "shape": o.shape,
                "color": o.color,
                "color_rgb": list(o.color_rgb),
                "size": o.size,
                "half_extent": o.half_extent,
                "x": o.x,
                "y": o.y,
                "z": o.z,
                "position_world": list(o.position),
                "lateral_x": d["lateral_x"],
                "vertical_y": d["vertical_y"],
                "centroid_depth": d["centroid_depth"],
                "nearest_surface_depth": d["nearest_surface_depth"],
            }
        )
    return {
        "scene_id": scene.scene_id,
        "regime": scene.regime,
        "camera": {
            "position_world": list(scene.camera.position),
            "look_at_world": list(scene.camera.target),
            "fov_degrees": scene.camera.fov_degrees,
            "resolution": list(scene.camera.resolution),
            "matrix_world": cam_T.tolist(),
            "intrinsics": intrinsics,
        },
        "objects": objects,
        "pair_relations": relations.relations_to_dict(rels),
    }


def _write_text(path: str, text: str):
    with open(path, "w") as f:
        f.write(text)


def _write_json(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def run(cfg, count, seed, *, start=0, step=1, worker_id=0):
    cfg = dict(cfg)
    cfg["output_dir"] = os.path.abspath(cfg["output_dir"])

    scene = _setup_render(cfg)
    _setup_world(scene)
    depth_out, seg_out = _setup_compositor(scene)
    cam = _add_camera(scene)
    _add_soft_lighting(scene)
    _add_ground(scene)
    object_pool = _add_object_pool(scene, 6)

    w, h = cfg["resolution"]
    fov = math.radians(cfg["camera_fov_degrees"])
    fx = fy = 0.5 * w / math.tan(0.5 * fov)
    cx, cy = 0.5 * w, 0.5 * h
    intrinsics = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]

    t0 = time.time()
    os.makedirs(cfg["output_dir"], exist_ok=True)
    indices = range(start, count, step)
    worker_total = len(indices)
    for worker_done, i in enumerate(indices, start=1):
        rng = random.Random(seed + i)
        s = sample.scene(rng, cfg, scene_id=i)
        rels = relations.derive(s)

        cam_T = _look_at(s.camera.position, s.camera.target)
        cam.matrix_world = Matrix(cam_T.tolist())
        cam.data.angle = math.radians(s.camera.fov_degrees)

        for j, obj_spec in enumerate(s.objects):
            _set_object(object_pool[j], obj_spec)
        for j in range(len(s.objects), len(object_pool)):
            object_pool[j].hide_render = True
            object_pool[j].hide_viewport = True

        prefix = f"{cfg['output_dir']}/{i:06d}"
        scene.render.filepath = f"{prefix}_rgb.png"
        depth_out.base_path = cfg["output_dir"]
        seg_out.base_path = cfg["output_dir"]
        depth_out.file_slots[0].path = f"{i:06d}_depth_"
        seg_out.file_slots[0].path = f"{i:06d}_segmentation_"
        bpy.ops.render.render(write_still=True)

        depth_exr = f"{prefix}_depth_{scene.frame_current:04d}.exr"
        seg_exr = f"{prefix}_segmentation_{scene.frame_current:04d}.exr"
        depth = _load_exr(depth_exr, w, h)
        seg = _load_exr(seg_exr, w, h)
        depth = np.where(depth > 999, np.nan, depth)

        _save_depth_png(depth, f"{prefix}_depth.png", cfg["depth_clip"])
        _save_segmentation(seg, f"{prefix}_segmentation.png")

        _write_json(f"{prefix}_scene.json", _scene_json(s, rels, cam_T, intrinsics))
        _write_text(f"{prefix}_description.txt", description.serialize(s, rels))
        _write_json(
            f"{prefix}_qa.json",
            qa.generate(s, rels, rng, n_target=cfg["qa_per_scene"]),
        )

        done = worker_done
        elapsed = time.time() - t0
        eta = elapsed / done * (worker_total - done)
        print(
            f"[worker {worker_id}] {done}/{worker_total} "
            f"sample={i} elapsed={elapsed:.0f}s eta={eta:.0f}s",
            flush=True,
        )
