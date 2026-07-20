import argparse
import fnmatch
import json
import math
import os
import random
import subprocess
import sys
import traceback
from pathlib import Path

try:
    import bpy
    from mathutils import Matrix, Vector
except ImportError:
    bpy = None
    Matrix = None
    Vector = None


TARGET_VIEWS = {
    "target_030.png": 30,
    "target_090.png": 90,
    "target_150.png": 150,
    "target_210.png": 210,
    "target_270.png": 270,
    "target_330.png": 330,
}
ZERO123PLUS_TARGET_AZIMUTHS = [30, 90, 150, 210, 270, 330]
ZERO123PLUS_TARGET_ELEVATIONS = {
    "v1.1": [30, -20, 30, -20, 30, -20],
    "v1.2": [20, -10, 20, -10, 20, -10],
}
REQUIRED_TARGET_IMAGES = ["cond.png", *TARGET_VIEWS.keys()]
SUPPORTED_EXTS = {".glb", ".gltf", ".obj", ".fbx"}


def get_zero123plus_target_poses(version):
    if version not in ZERO123PLUS_TARGET_ELEVATIONS:
        raise ValueError(f"Unknown Zero123++ pose version: {version}")
    return list(ZERO123PLUS_TARGET_AZIMUTHS), list(ZERO123PLUS_TARGET_ELEVATIONS[version])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="data/source_models")
    parser.add_argument("--output_dir", default="data/reference_zero123plus_tiny")
    parser.add_argument("--max_objects", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--render_size", type=int, default=768)
    parser.add_argument(
        "--exposure",
        type=float,
        default=-0.75,
        help="Blender render exposure. Use a lower value, such as -1.5, for overly bright models.",
    )
    parser.add_argument(
        "--postprocess_python",
        default=sys.executable,
        help="Python executable with Pillow used for white compositing and Lanczos resize.",
    )
    parser.add_argument("--quality_report_samples", type=int, default=5)
    parser.add_argument(
        "--rerender_list",
        default=None,
        help="Optional text file containing object IDs to rerender in place.",
    )
    parser.add_argument(
        "--skip_existing_good",
        action="store_true",
        help="Keep existing valid object folders instead of overwriting them.",
    )
    parser.add_argument("--zero123plus_pose_version", choices=["v1.1", "v1.2"], default="v1.2")
    parser.add_argument("--cond_elevation", type=float, default=0.0)
    parser.add_argument("--elevation", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--fov", type=float, default=30.0)
    parser.add_argument("--camera_padding", type=float, default=0.86)
    parser.add_argument("--target_size", type=float, default=1.6)
    parser.add_argument("--min_occupancy", type=float, default=0.03)
    parser.add_argument("--max_occupancy", type=float, default=0.92)
    parser.add_argument("--ref_sampling_mode", choices=["biased_side_back", "fixed"], default="biased_side_back")
    parser.add_argument("--ref_num", type=int, default=3)
    parser.add_argument("--ref_front_prob", type=float, default=0.2)
    parser.add_argument("--ref_side_prob", type=float, default=0.4)
    parser.add_argument("--ref_back_prob", type=float, default=0.4)
    parser.add_argument("--ref_min_target_gap", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--orientation_metadata",
        default=None,
        help="Optional JSON mapping model filenames to yaw corrections in degrees.",
    )
    parser.add_argument(
        "--default_yaw_offset",
        type=float,
        default=0.0,
        help="Yaw correction applied when a model has no orientation metadata entry.",
    )
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--blender_path", default="blender")
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    elif bpy is None:
        argv = argv[1:]
    else:
        argv = []
    args = parser.parse_args(argv)
    if args.elevation is not None:
        args.cond_elevation = args.elevation
    return args


def run_blender_wrapper(args):
    script_path = Path(__file__).resolve()
    cmd = [
        args.blender_path, "--background", "--python", str(script_path), "--",
        "--source_dir", args.source_dir,
        "--output_dir", args.output_dir,
        "--max_objects", str(args.max_objects),
        "--image_size", str(args.image_size),
        "--render_size", str(args.render_size),
        "--exposure", str(args.exposure),
        "--postprocess_python", args.postprocess_python,
        "--quality_report_samples", str(args.quality_report_samples),
        "--zero123plus_pose_version", args.zero123plus_pose_version,
        "--cond_elevation", str(args.cond_elevation),
        "--radius", str(args.radius),
        "--fov", str(args.fov),
        "--camera_padding", str(args.camera_padding),
        "--target_size", str(args.target_size),
        "--min_occupancy", str(args.min_occupancy),
        "--max_occupancy", str(args.max_occupancy),
        "--ref_sampling_mode", args.ref_sampling_mode,
        "--ref_num", str(args.ref_num),
        "--ref_front_prob", str(args.ref_front_prob),
        "--ref_side_prob", str(args.ref_side_prob),
        "--ref_back_prob", str(args.ref_back_prob),
        "--ref_min_target_gap", str(args.ref_min_target_gap),
        "--seed", str(args.seed),
        "--default_yaw_offset", str(args.default_yaw_offset),
    ]
    if args.orientation_metadata:
        cmd.extend(["--orientation_metadata", args.orientation_metadata])
    if args.rerender_list:
        cmd.extend(["--rerender_list", args.rerender_list])
    if args.skip_existing_good:
        cmd.append("--skip_existing_good")
    if args.white_background:
        cmd.append("--white_background")
    print("[render-wrapper] launching Blender:")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
    subprocess.run(cmd, check=True)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_model(path):
    ext = path.suffix.lower()
    if ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif ext == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    else:
        raise ValueError(f"Unsupported model extension: {path}")


def mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def remove_imported_non_meshes():
    for obj in list(bpy.context.scene.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)


def remove_named_meshes(patterns):
    removed = []
    for obj in list(bpy.context.scene.objects):
        if obj.type != "MESH":
            continue
        if any(fnmatch.fnmatchcase(obj.name, pattern) for pattern in patterns):
            removed.append(obj.name)
            bpy.data.objects.remove(obj, do_unlink=True)
    if removed:
        print(f"[render] removed configured meshes: {removed}")
    return removed


def force_materials_opaque():
    changed = []
    for material in bpy.data.materials:
        if not material.use_nodes or material.node_tree is None:
            continue
        nodes = material.node_tree.nodes
        output = next((node for node in nodes if node.type == "OUTPUT_MATERIAL"), None)
        emission = next((node for node in nodes if node.type == "EMISSION"), None)
        if output is None or emission is None:
            continue
        surface = output.inputs.get("Surface")
        if surface is None:
            continue
        for link in list(surface.links):
            material.node_tree.links.remove(link)
        material.node_tree.links.new(emission.outputs[0], surface)
        if hasattr(material, "surface_render_method"):
            material.surface_render_method = "DITHERED"
        changed.append(material.name)
    if changed:
        print(f"[render] forced materials opaque: {changed}")
    return changed


def hierarchy_mesh_vertex_count(root):
    objects = [root, *list(root.children_recursive)]
    return sum(len(obj.data.vertices) for obj in objects if obj.type == "MESH")


def remove_low_detail_outlier_roots():
    """Remove obvious disconnected helper geometry without pruning model parts."""
    roots = [obj for obj in bpy.context.scene.objects if obj.parent is None]
    root_counts = [(root, hierarchy_mesh_vertex_count(root)) for root in roots]
    root_counts = [(root, count) for root, count in root_counts if count > 0]
    total_vertices = sum(count for _, count in root_counts)
    if len(root_counts) < 2 or total_vertices <= 0:
        return []

    dominant_root, dominant_count = max(root_counts, key=lambda item: item[1])
    if dominant_count / total_vertices < 0.80:
        return []

    removed = []
    for root, count in root_counts:
        if root is dominant_root or count / total_vertices > 0.05:
            continue
        removed.append({"root": root.name, "mesh_vertices": count})
        hierarchy = [*reversed(list(root.children_recursive)), root]
        for obj in hierarchy:
            bpy.data.objects.remove(obj, do_unlink=True)
    if removed:
        print(f"[render] removed disconnected low-detail helper roots: {removed}")
    return removed


def bbox_for_objects(objects):
    min_corner = Vector((float("inf"), float("inf"), float("inf")))
    max_corner = Vector((float("-inf"), float("-inf"), float("-inf")))
    bpy.context.view_layer.update()
    for obj in objects:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            min_corner.x = min(min_corner.x, world.x)
            min_corner.y = min(min_corner.y, world.y)
            min_corner.z = min(min_corner.z, world.z)
            max_corner.x = max(max_corner.x, world.x)
            max_corner.y = max(max_corner.y, world.y)
            max_corner.z = max(max_corner.z, world.z)
    return min_corner, max_corner


def normalize_scene(target_size, yaw_offset_degrees=0.0):
    objects = mesh_objects()
    if not objects:
        raise ValueError("No mesh objects imported.")

    min_before, max_before = bbox_for_objects(objects)
    center_before = (min_before + max_before) * 0.5
    size_before_vec = max_before - min_before
    max_dim = max(size_before_vec.x, size_before_vec.y, size_before_vec.z, 1e-6)
    applied_scale = target_size / max_dim

    # GLB/FBX files commonly keep meshes below rotated or scaled EMPTY/ARMATURE
    # parents. Moving mesh children by a world-space bounding-box center applies
    # that offset in parent-local space and can send the model far from camera.
    # Normalize the complete imported hierarchy through one new root instead.
    normalization_root = bpy.data.objects.new("DatasetNormalizationRoot", None)
    bpy.context.scene.collection.objects.link(normalization_root)
    imported_roots = [
        obj for obj in bpy.context.scene.objects
        if obj is not normalization_root and obj.parent is None
    ]
    for obj in imported_roots:
        world_matrix = obj.matrix_world.copy()
        obj.parent = normalization_root
        obj.matrix_world = world_matrix

    normalization_root.scale = (applied_scale,) * 3
    normalization_root.location = Vector((
        -center_before.x * applied_scale,
        -center_before.y * applied_scale,
        -min_before.z * applied_scale,
    ))

    # Rotate around the normalized model center. Keeping orientation separate
    # from normalization avoids disturbing imported parent/armature transforms.
    orientation_root = bpy.data.objects.new("DatasetOrientationRoot", None)
    bpy.context.scene.collection.objects.link(orientation_root)
    normalization_root.parent = orientation_root
    orientation_root.rotation_euler[2] = math.radians(yaw_offset_degrees)
    bpy.context.view_layer.update()

    min_after, max_after = bbox_for_objects(objects)
    center_after = (min_after + max_after) * 0.5
    size_after_vec = max_after - min_after
    return {
        "bbox_center_before": list(center_before),
        "bbox_size_before": list(size_before_vec),
        "bbox_center_after": list(center_after),
        "bbox_size_after": list(size_after_vec),
        "applied_scale": applied_scale,
        "yaw_offset_degrees": yaw_offset_degrees,
        "look_at": list(center_after),
    }


def load_orientation_metadata(path):
    if not path:
        return {}
    metadata_path = Path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Orientation metadata does not exist: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Orientation metadata must be a JSON object keyed by model filename.")
    return data


def resolve_yaw_offset(model_path, orientation_metadata, default_yaw_offset=0.0):
    model_path = Path(model_path)
    keys = (model_path.name, model_path.stem, str(model_path), str(model_path.resolve()))
    entry = next((orientation_metadata[key] for key in keys if key in orientation_metadata), None)
    if entry is None:
        return float(default_yaw_offset)
    if isinstance(entry, dict):
        if "yaw_degrees" in entry:
            entry = entry["yaw_degrees"]
        elif "yaw" in entry:
            entry = entry["yaw"]
        else:
            raise ValueError(
                f"Orientation entry for {model_path.name} needs 'yaw_degrees' or 'yaw'."
            )
    if not isinstance(entry, (int, float)):
        raise ValueError(f"Yaw correction for {model_path.name} must be a number, got {entry!r}.")
    return float(entry)


def resolve_model_render_options(model_path, orientation_metadata, default_yaw_offset=0.0):
    model_path = Path(model_path)
    keys = (model_path.name, model_path.stem, str(model_path), str(model_path.resolve()))
    entry = next((orientation_metadata[key] for key in keys if key in orientation_metadata), None)
    options = {
        "yaw_degrees": float(default_yaw_offset),
        "force_opaque": False,
        "remove_meshes": [],
    }
    if entry is None:
        return options
    if isinstance(entry, (int, float)):
        options["yaw_degrees"] = float(entry)
        return options
    if not isinstance(entry, dict):
        raise ValueError(f"Render options for {model_path.name} must be a number or object.")
    if "yaw_degrees" in entry:
        options["yaw_degrees"] = float(entry["yaw_degrees"])
    elif "yaw" in entry:
        options["yaw_degrees"] = float(entry["yaw"])
    options["force_opaque"] = bool(entry.get("force_opaque", False))
    remove_meshes = entry.get("remove_meshes", [])
    if isinstance(remove_meshes, str):
        remove_meshes = [remove_meshes]
    if not isinstance(remove_meshes, list) or not all(isinstance(item, str) for item in remove_meshes):
        raise ValueError(f"remove_meshes for {model_path.name} must be a string list.")
    options["remove_meshes"] = remove_meshes
    return options


def add_area_light(name, location, energy, size, color=(1.0, 1.0, 1.0)):
    light_data = bpy.data.lights.new(name, type="AREA")
    light = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light)
    light.location = location
    light.data.energy = energy
    light.data.size = size
    light.data.color = color
    light.rotation_euler = (0, 0, 0)
    direction = Vector((0, 0, 0.7)) - light.location
    light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return light


def setup_render(render_size, fov, exposure):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 96
    scene.render.resolution_x = render_size
    scene.render.resolution_y = render_size
    scene.render.resolution_percentage = 100
    # Render RGBA internally for clean antialiased edges, then composite to an
    # RGB white PNG in the postprocessing step.
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    if hasattr(scene.render, "use_motion_blur"):
        scene.render.use_motion_blur = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = exposure
    scene.world = scene.world or bpy.data.worlds.new("World")
    scene.world.color = (1.0, 1.0, 1.0)
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        background.inputs["Strength"].default_value = 0.35

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    camera.data.angle = math.radians(fov)
    camera.data.dof.use_dof = False

    # Four broad lights keep front, side, and back details readable while
    # retaining enough directionality to show plush shape.
    add_area_light("Front_Key", (0, -4.5, 4.5), 440, 4.0)
    add_area_light("Rear_Fill", (0, 4.5, 3.2), 340, 4.5)
    add_area_light("Left_Fill", (-4.0, 0, 3.0), 285, 4.0)
    add_area_light("Right_Fill", (4.0, 0, 3.0), 285, 4.0)
    add_area_light("Top_Soft", (0, 0, 6.0), 190, 5.0)
    return camera


def point_camera(camera, azimuth, elevation, radius, look_at):
    az = math.radians(azimuth)
    el = math.radians(elevation)
    look_at = Vector(look_at)
    camera.location = look_at + Vector((
        radius * math.cos(el) * math.sin(az),
        -radius * math.cos(el) * math.cos(az),
        radius * math.sin(el),
    ))
    direction = look_at - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def postprocess_render(raw_path, output_path, report_path, output_size, python_executable):
    helper = Path(__file__).resolve().parent / "postprocess_reference_render.py"
    cmd = [
        python_executable,
        str(helper),
        "--input", str(raw_path),
        "--output", str(output_path),
        "--size", str(output_size),
        "--report", str(report_path),
    ]
    subprocess.run(cmd, check=True)
    with open(report_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def render_view(camera, output_path, azimuth, elevation, radius, look_at, args):
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_name(f".{output_path.stem}_raw.png")
    report_path = output_path.with_name(f".{output_path.stem}_quality.json")
    point_camera(camera, azimuth, elevation, radius, look_at)
    bpy.context.scene.render.filepath = str(raw_path)
    print(f"[render] saving {output_path} azimuth={azimuth} elevation={elevation}")
    result = bpy.ops.render.render(write_still=True)
    if "FINISHED" not in result:
        raise RuntimeError(f"Blender render did not finish for {output_path}: {result}")
    if not raw_path.exists():
        raise FileNotFoundError(f"Render finished but raw PNG was not written: {raw_path}")
    report = postprocess_render(
        raw_path,
        output_path,
        report_path,
        args.image_size,
        args.postprocess_python,
    )
    raw_path.unlink(missing_ok=True)
    report_path.unlink(missing_ok=True)
    return report


def validate_quality_report(filename, report, expected_size, min_occupancy, max_occupancy):
    if tuple(report["resolution"]) != expected_size:
        return False, f"{filename}: size mismatch {report['resolution']} != {expected_size}"
    if report["mode"] != "RGB":
        return False, f"{filename}: expected RGB PNG, got mode={report['mode']}"
    if report["has_alpha"]:
        return False, f"{filename}: final PNG unexpectedly contains alpha"
    occupancy = report["bbox_occupancy"]
    if occupancy < min_occupancy:
        return False, f"{filename}: object too small occupancy={occupancy:.4f}"
    if occupancy > max_occupancy:
        return False, f"{filename}: object likely clipped occupancy={occupancy:.4f}"
    bbox = report.get("bbox")
    if bbox is None:
        return False, f"{filename}: no foreground detected against white background"
    margin = 2
    width, height = expected_size
    if bbox[0] <= margin or bbox[1] <= margin or bbox[2] >= width - margin or bbox[3] >= height - margin:
        return False, f"{filename}: object mostly out of frame bbox={bbox}"
    return True, "ok"


def validate_object_images(object_dir, image_size, min_occupancy, max_occupancy, ref_files, reports):
    expected_size = (image_size, image_size)
    required_images = REQUIRED_TARGET_IMAGES + list(ref_files)
    for filename in required_images:
        path = object_dir / filename
        if not path.exists():
            return False, f"missing {filename}"
        report = reports.get(filename)
        if report is None:
            return False, f"missing quality report for {filename}"
        ok, reason = validate_quality_report(
            filename, report, expected_size, min_occupancy, max_occupancy
        )
        if not ok:
            return False, reason
    return True, "ok"


def quality_warnings(filename, report, azimuth):
    warnings = []
    if report["bbox_width_ratio"] < 0.50:
        warnings.append(f"foreground width is only {report['bbox_width_ratio']:.1%}")
    if report["bbox_height_ratio"] < 0.50:
        warnings.append(f"foreground height is only {report['bbox_height_ratio']:.1%}")
    is_side_or_back = 60 <= (azimuth % 360) <= 300
    if is_side_or_back and report["foreground_luminance"] < 0.10:
        warnings.append(
            f"side/back foreground luminance is low ({report['foreground_luminance']:.3f})"
        )
    if report["edge_variance"] < 8.0:
        warnings.append(f"low edge sharpness score ({report['edge_variance']:.2f})")
    return warnings


def compute_camera_radius(norm, fov_degrees, min_radius, padding):
    size = norm["bbox_size_after"]
    diagonal = math.sqrt(size[0] ** 2 + size[1] ** 2 + size[2] ** 2)
    bounding_radius = max(diagonal * 0.5, 1e-6)
    half_fov = math.radians(max(fov_degrees, 1e-3) * 0.5)
    fit_radius = bounding_radius / max(math.sin(half_fov), 1e-6)
    return max(min_radius, fit_radius * padding)


def angular_distance(a, b):
    return min(abs(a - b), 360 - abs(a - b))


def sample_from_ranges(ranges):
    start, end = random.choice(ranges)
    if end >= start:
        return random.uniform(start, end)
    span = (360 - start) + end
    value = (start + random.uniform(0, span)) % 360
    return value


def avoid_exact_target(angle, min_gap):
    targets = [30, 90, 150, 210, 270, 330]
    if min_gap <= 0:
        return angle % 360
    for target in targets:
        if angular_distance(angle, target) < min_gap:
            direction = 1 if ((angle - target) % 360) < 180 else -1
            return (target + direction * min_gap) % 360
    return angle % 360


def target_pose_items(version):
    azimuths, elevations = get_zero123plus_target_poses(version)
    filenames = [
        "target_030.png",
        "target_090.png",
        "target_150.png",
        "target_210.png",
        "target_270.png",
        "target_330.png",
    ]
    return list(zip(filenames, azimuths, elevations))


def sample_reference_azimuths(args):
    if args.ref_sampling_mode == "fixed":
        return [0, 90, 180][:args.ref_num]

    total = args.ref_front_prob + args.ref_side_prob + args.ref_back_prob
    if total <= 0:
        raise ValueError("Reference sampling probabilities must sum to > 0.")
    front_cut = args.ref_front_prob / total
    side_cut = front_cut + args.ref_side_prob / total

    azimuths = []
    seen = set()
    attempts = 0
    while len(azimuths) < args.ref_num:
        attempts += 1
        if attempts > args.ref_num * 100:
            raise RuntimeError("Could not sample enough unique reference azimuths.")
        draw = random.random()
        if draw < front_cut:
            angle = sample_from_ranges([(330, 360), (0, 30)])
        elif draw < side_cut:
            angle = sample_from_ranges([(60, 120), (240, 300)])
        else:
            angle = sample_from_ranges([(135, 225)])
        azimuth = int(round(avoid_exact_target(angle, args.ref_min_target_gap))) % 360
        if azimuth in seen:
            continue
        seen.add(azimuth)
        azimuths.append(azimuth)
    return azimuths


def sample_reference_elevations(args):
    _, target_elevations = get_zero123plus_target_poses(args.zero123plus_pose_version)
    low = min(target_elevations)
    high = max(target_elevations)
    return [int(round(random.uniform(low, high))) for _ in range(args.ref_num)]


def write_meta(path, meta):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def ref_filename(azimuth):
    return f"ref_{int(round(azimuth)) % 360:03d}.png"


def build_rendered_views(target_items, ref_files, ref_azimuths, ref_elevations, cond_elevation):
    return {
        "cond.png": {"azimuth": 0, "elevation": cond_elevation},
        **{
            filename: {"azimuth": azimuth, "elevation": elevation}
            for filename, azimuth, elevation in target_items
        },
        **{
            filename: {"azimuth": azimuth, "elevation": elevation}
            for filename, azimuth, elevation in zip(ref_files, ref_azimuths, ref_elevations)
        },
    }


def render_object(model_path, object_dir, rejected_dir, object_id, args, render_options=None):
    render_options = render_options or {}
    yaw_offset_degrees = float(render_options.get("yaw_degrees", 0.0))
    for stale_png in object_dir.glob("*.png"):
        stale_png.unlink()

    target_items = target_pose_items(args.zero123plus_pose_version)
    target_azimuths = [azimuth for _, azimuth, _ in target_items]
    target_elevations = [elevation for _, _, elevation in target_items]
    ref_azimuths = sample_reference_azimuths(args)
    ref_elevations = sample_reference_elevations(args)
    ref_files = [ref_filename(azimuth) for azimuth in ref_azimuths]
    rendered_views = build_rendered_views(
        target_items, ref_files, ref_azimuths, ref_elevations, args.cond_elevation
    )
    meta_path = object_dir / "meta.json"
    meta = {
        "object_id": object_id,
        "source_model": str(model_path),
        "image_size": args.image_size,
        "render_size": args.render_size,
        "resampling": "Lanczos",
        "final_image_mode": "RGB",
        "final_background": "white",
        "depth_of_field": False,
        "motion_blur": False,
        "exposure": args.exposure,
        "zero123plus_pose_version": args.zero123plus_pose_version,
        "cond_azimuth": 0,
        "cond_elevation": args.cond_elevation,
        "yaw_offset_degrees": yaw_offset_degrees,
        "render_options": render_options,
        "radius": args.radius,
        "fov": args.fov,
        "camera_padding": args.camera_padding,
        "target_azimuths": target_azimuths,
        "target_elevations": target_elevations,
        "target_order": target_azimuths,
        "rendered_views": rendered_views,
        "ref_sampling_mode": args.ref_sampling_mode,
        "ref_azimuths": ref_azimuths,
        "ref_elevations": ref_elevations,
        "status": "started",
    }
    write_meta(meta_path, meta)

    clear_scene()
    import_model(model_path)
    remove_imported_non_meshes()
    removed_configured_meshes = remove_named_meshes(render_options.get("remove_meshes", []))
    forced_opaque_materials = force_materials_opaque() if render_options.get("force_opaque") else []
    meta["removed_configured_meshes"] = removed_configured_meshes
    meta["forced_opaque_materials"] = forced_opaque_materials
    removed_helper_roots = remove_low_detail_outlier_roots()
    meta["removed_helper_roots"] = removed_helper_roots
    norm = normalize_scene(args.target_size, yaw_offset_degrees=yaw_offset_degrees)
    meta.update(norm)
    camera_radius = compute_camera_radius(norm, args.fov, args.radius, args.camera_padding)
    meta["camera_radius"] = camera_radius
    camera = setup_render(args.render_size, args.fov, args.exposure)
    quality_reports = {}

    quality_reports["cond.png"] = render_view(
        camera, object_dir / "cond.png", 0, args.cond_elevation, camera_radius, norm["look_at"], args
    )
    quality_reports["cond.png"]["azimuth"] = 0
    quality_reports["cond.png"]["elevation"] = args.cond_elevation
    for filename, azimuth, elevation in target_items:
        quality_reports[filename] = render_view(
            camera, object_dir / filename, azimuth, elevation, camera_radius, norm["look_at"], args
        )
        quality_reports[filename]["azimuth"] = azimuth
        quality_reports[filename]["elevation"] = elevation
    for filename, azimuth, elevation in zip(ref_files, ref_azimuths, ref_elevations):
        quality_reports[filename] = render_view(
            camera, object_dir / filename, azimuth, elevation, camera_radius, norm["look_at"], args
        )
        quality_reports[filename]["azimuth"] = azimuth
        quality_reports[filename]["elevation"] = elevation

    quality_summary = {
        "object_id": object_id,
        "source_model": str(model_path),
        "render_resolution": [args.render_size, args.render_size],
        "saved_resolution": [args.image_size, args.image_size],
        "saved_mode": "RGB",
        "background": "white",
        "resampling": "Lanczos",
        "depth_of_field": False,
        "motion_blur": False,
        "exposure": args.exposure,
        "views": quality_reports,
    }
    write_meta(object_dir / "quality_report.json", quality_summary)
    meta["quality_report"] = "quality_report.json"
    meta["quality_summary"] = quality_reports

    object_number = int(object_id.rsplit("_", 1)[-1])
    if object_number <= args.quality_report_samples:
        print(
            f"[quality] {object_id}: render={args.render_size}x{args.render_size} "
            f"saved={args.image_size}x{args.image_size} mode=RGB background=white "
            f"resampling=Lanczos dof=false motion_blur=false exposure={args.exposure:g}"
        )
        for filename, report in quality_reports.items():
            warnings = quality_warnings(filename, report, report["azimuth"])
            print(
                f"[quality] {filename}: bbox={report['bbox']} "
                f"fill={report['bbox_width_ratio']:.1%}x{report['bbox_height_ratio']:.1%} "
                f"luminance={report['foreground_luminance']:.3f} "
                f"sharpness={report['edge_variance']:.2f}"
            )
            for warning in warnings:
                print(f"[quality:warning] {filename}: {warning}")

    ok, reason = validate_object_images(
        object_dir,
        args.image_size,
        args.min_occupancy,
        args.max_occupancy,
        ref_files,
        quality_reports,
    )
    meta["validation_passed"] = ok
    meta["validation_reason"] = reason
    meta["status"] = "ok" if ok else "rejected"
    write_meta(meta_path, meta)
    if not ok:
        rejected_dir.mkdir(parents=True, exist_ok=True)
        write_meta(rejected_dir / f"{object_id}.json", meta)
        raise ValueError(reason)
    return ref_files, ref_azimuths, ref_elevations, target_azimuths, target_elevations


def manifest_record(object_id, ref_files, ref_azimuths, ref_elevations, target_azimuths, target_elevations):
    return {
        "object_id": object_id,
        "cond_img": f"{object_id}/cond.png",
        "target_azimuths": target_azimuths,
        "target_elevations": target_elevations,
        "target_imgs": {
            "30": f"{object_id}/target_030.png",
            "90": f"{object_id}/target_090.png",
            "150": f"{object_id}/target_150.png",
            "210": f"{object_id}/target_210.png",
            "270": f"{object_id}/target_270.png",
            "330": f"{object_id}/target_330.png",
        },
        "ref_imgs": [
            f"{object_id}/{filename}" for filename in ref_files
        ],
        "ref_view_labels": ["unknown"] * len(ref_files),
        "ref_azimuths": ref_azimuths,
        "ref_elevations": ref_elevations,
    }


def write_manifest(records, output_dir):
    manifest_path = output_dir / "train.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def load_manifest(path):
    records = {}
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            object_id = record.get("object_id")
            if not object_id:
                raise ValueError(f"Missing object_id in {path} line {line_number}")
            records[object_id] = record
    return records


def load_object_id_list(path):
    if not path:
        return None
    list_path = Path(path)
    if not list_path.exists():
        raise FileNotFoundError(f"Rerender list does not exist: {list_path}")
    object_ids = set()
    with open(list_path, "r", encoding="utf-8") as handle:
        for line in handle:
            value = line.split("#", 1)[0].strip()
            if value:
                object_ids.add(value)
    if not object_ids:
        raise ValueError(
            f"Rerender list {list_path} contains no object IDs. "
            "Lines beginning with '#' are comments."
        )
    return object_ids


def main():
    args = parse_args()
    if bpy is None:
        run_blender_wrapper(args)
        return

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    rejected_dir = output_dir / "rejected"
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    orientation_metadata = load_orientation_metadata(args.orientation_metadata)
    rerender_ids = load_object_id_list(args.rerender_list)
    existing_records = load_manifest(output_dir / "train.jsonl")

    all_models = [
        path for path in sorted(source_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
    ]
    models = all_models if rerender_ids is not None else all_models[: args.max_objects]
    if not models:
        raise ValueError(f"No supported 3D models found in {source_dir}")

    records_by_id = dict(existing_records) if (rerender_ids is not None or args.skip_existing_good) else {}
    rejected = []
    kept = 0
    skipped = 0
    rerendered = 0
    requested_found = set()
    for idx, model_path in enumerate(models, start=1):
        object_id = f"object_{idx:06d}"
        object_dir = output_dir / object_id
        if rerender_ids is not None and object_id not in rerender_ids:
            skipped += 1
            if object_id in existing_records:
                kept += 1
            continue
        if rerender_ids is not None:
            requested_found.add(object_id)
        elif args.skip_existing_good and object_id in existing_records and object_dir.exists():
            meta_path = object_dir / "meta.json"
            try:
                existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                existing_meta = {}
            if existing_meta.get("status") == "ok" and existing_meta.get("validation_passed") is True:
                skipped += 1
                kept += 1
                print(f"[render:skip] {object_id}: existing valid object kept")
                continue
        if object_dir.exists():
            import shutil
            shutil.rmtree(object_dir)
        object_dir.mkdir(parents=True, exist_ok=True)
        rerendered += 1
        print(f"[render] {object_id}: {model_path}")
        try:
            render_options = resolve_model_render_options(
                model_path,
                orientation_metadata,
                default_yaw_offset=args.default_yaw_offset,
            )
            print(
                f"[render] {object_id} orientation yaw correction: "
                f"{render_options['yaw_degrees']:g} degrees"
            )
            print(f"[render] {object_id} model render options: {render_options}")
            ref_files, ref_azimuths, ref_elevations, target_azimuths, target_elevations = render_object(
                model_path,
                object_dir,
                rejected_dir,
                object_id,
                args,
                render_options=render_options,
            )
            print(
                f"[render] {object_id} sampled reference poses: "
                f"{list(zip(ref_azimuths, ref_elevations))}"
            )
        except Exception as exc:
            meta_path = object_dir / "meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as handle:
                        meta = json.load(handle)
                except (OSError, ValueError):
                    meta = {}
            else:
                meta = {}
            meta.update({
                "object_id": object_id,
                "source_model": str(model_path),
                "status": "error",
                "validation_passed": False,
                "validation_reason": str(exc),
                "traceback": traceback.format_exc(),
            })
            write_meta(meta_path, meta)
            rejected_dir.mkdir(parents=True, exist_ok=True)
            write_meta(rejected_dir / f"{object_id}.json", meta)
            rejected.append({"object_id": object_id, "source_model": str(model_path), "reason": str(exc)})
            records_by_id.pop(object_id, None)
            print(f"[render:rejected] {object_id}: {exc}")
            continue
        records_by_id[object_id] = manifest_record(
            object_id,
            ref_files,
            ref_azimuths,
            ref_elevations,
            target_azimuths,
            target_elevations,
        )

    if rerender_ids is not None:
        missing_requested = sorted(rerender_ids - requested_found)
        if missing_requested:
            print(f"[render:warning] requested IDs not found in source-model index: {missing_requested}")

    records = [records_by_id[key] for key in sorted(records_by_id)]
    write_manifest(records, output_dir)
    with open(output_dir / "rejected.json", "w", encoding="utf-8") as handle:
        json.dump(rejected, handle, indent=2)
    print(f"[render] wrote {output_dir / 'train.jsonl'} with {len(records)} valid sample(s)")
    print(
        f"[render] repair summary: kept={kept} skipped={skipped} "
        f"rerendered={rerendered} rejected={len(rejected)}"
    )
    if not records:
        raise RuntimeError("No objects rendered successfully. Check rejected.json and per-object meta.json.")


if __name__ == "__main__":
    main()
