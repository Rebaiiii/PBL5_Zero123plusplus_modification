import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from PIL import Image, ImageOps


ZERO123PLUS_TARGET_AZIMUTHS = [30, 90, 150, 210, 270, 330]
ZERO123PLUS_TARGET_ELEVATIONS = {
    "v1.1": [30, -20, 30, -20, 30, -20],
    "v1.2": [20, -10, 20, -10, 20, -10],
}
REFERENCE_SLOT_LABELS = ("view_30", "view_90", "view_150", "view_210", "view_270", "view_330", "unknown")
REFERENCE_UNKNOWN_VIEW_ID = 6
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
COARSE_SLOT_WEIGHTS = {
    "front": [1.0, 0.05, 0.05, 0.05, 0.05, 1.0],
    "side": [0.05, 1.0, 0.35, 0.35, 1.0, 0.05],
    "back": [0.05, 0.05, 1.0, 1.0, 0.05, 0.05],
    "unknown": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
}
AZIMUTH_SLOT_WEIGHTS = {
    0: [1.0, 0.05, 0.05, 0.05, 0.05, 1.0],
    90: [0.05, 1.0, 0.35, 0.05, 0.35, 0.05],
    180: [0.05, 0.05, 1.0, 1.0, 0.05, 0.05],
}


@dataclass
class ReferenceImage:
    image: Image.Image
    path: str
    view_id: int = REFERENCE_UNKNOWN_VIEW_ID
    view_label: str = "unknown"
    azimuth: Optional[float] = None
    elevation: Optional[float] = None
    pose_source: str = "unknown"


def normalize_coarse_label(label: str) -> str:
    label = (label or "unknown").strip().lower()
    if label in ("front", "side", "back"):
        return label
    if label in REFERENCE_SLOT_LABELS:
        return label
    return "unknown"


def infer_label_from_filename(path: str) -> str:
    name = os.path.basename(path).lower().replace("-", "_").replace(".", "_")
    parts = name.split("_")
    if any(part in ("front", "frontal") for part in parts):
        return "front"
    if any(part in ("back", "rear") for part in parts):
        return "back"
    if any(part in ("side", "left", "right", "profile") for part in parts):
        return "side"
    for idx, slot in enumerate(REFERENCE_SLOT_LABELS[:6]):
        if slot in name or str(idx) in parts:
            return slot
    return "unknown"


def label_to_view_id(label: str) -> int:
    label = normalize_coarse_label(label)
    if label == "front":
        return 0
    if label == "side":
        return 1
    if label == "back":
        return 2
    if label in REFERENCE_SLOT_LABELS:
        return REFERENCE_SLOT_LABELS.index(label)
    return REFERENCE_UNKNOWN_VIEW_ID


def label_to_slot_weights(label: str) -> List[float]:
    label = normalize_coarse_label(label)
    if label in COARSE_SLOT_WEIGHTS:
        return COARSE_SLOT_WEIGHTS[label]
    if label in REFERENCE_SLOT_LABELS[:6]:
        weights = [0.05] * 6
        weights[REFERENCE_SLOT_LABELS.index(label)] = 1.0
        return weights
    return COARSE_SLOT_WEIGHTS["unknown"]


def azimuth_to_slot_weights(azimuth: float) -> List[float]:
    azimuth = int(round(float(azimuth))) % 360
    if azimuth in AZIMUTH_SLOT_WEIGHTS:
        return AZIMUTH_SLOT_WEIGHTS[azimuth]
    weights = []
    for target in ZERO123PLUS_TARGET_AZIMUTHS:
        distance = min(abs(target - azimuth), 360 - abs(target - azimuth))
        if distance <= 30:
            weight = 1.0
        elif distance <= 75:
            weight = 0.35
        else:
            weight = 0.05
        weights.append(weight)
    return weights


def get_zero123plus_target_poses(version: str = "v1.2") -> tuple[List[int], List[int]]:
    if version not in ZERO123PLUS_TARGET_ELEVATIONS:
        raise ValueError(f"Unknown Zero123++ pose version: {version}")
    return list(ZERO123PLUS_TARGET_AZIMUTHS), list(ZERO123PLUS_TARGET_ELEVATIONS[version])


def circular_azimuth_distance(a: float, b: float) -> float:
    diff = abs(float(a) - float(b)) % 360.0
    return min(diff, 360.0 - diff)


def wide_pose_to_slot_weights(
    azimuth: float,
    elevation: float,
    target_azimuths: Optional[Sequence[float]] = None,
    target_elevations: Optional[Sequence[float]] = None,
    sigma_deg: float = 80.0,
    min_weight: float = 0.05,
    normalize: bool = True,
    elevation_weight: float = 0.25,
) -> List[float]:
    target_azimuths = list(target_azimuths or ZERO123PLUS_TARGET_AZIMUTHS)
    target_elevations = list(target_elevations or ZERO123PLUS_TARGET_ELEVATIONS["v1.2"])
    if len(target_azimuths) != 6 or len(target_elevations) != 6:
        raise ValueError("Zero123++ target azimuth/elevation lists must both have length 6.")
    if sigma_deg <= 0:
        raise ValueError("sigma_deg must be positive.")
    if elevation_weight < 0:
        raise ValueError("elevation_weight must be non-negative.")

    weights = []
    for target_azimuth, target_elevation in zip(target_azimuths, target_elevations):
        azimuth_dist = circular_azimuth_distance(azimuth, target_azimuth)
        elevation_dist = abs(float(elevation) - float(target_elevation))
        distance = math.sqrt(azimuth_dist ** 2 + elevation_weight * elevation_dist ** 2)
        weights.append(max(float(min_weight), math.exp(-(distance ** 2) / (2 * sigma_deg ** 2))))
    if normalize:
        peak = max(weights)
        if peak > 0:
            weights = [weight / peak for weight in weights]
    return weights


def propagate_slot_weights(
    weights: Sequence[float],
    target_azimuths: Optional[Sequence[float]] = None,
    strength: float = 0.3,
    neighbor_degrees: float = 120.0,
) -> List[float]:
    """Spread part of each slot's influence to nearby azimuth slots."""
    target_azimuths = list(target_azimuths or ZERO123PLUS_TARGET_AZIMUTHS)
    if len(weights) != 6 or len(target_azimuths) != 6:
        raise ValueError("slot weights and target azimuths must both have length 6.")
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0:
        return [float(weight) for weight in weights]
    if neighbor_degrees <= 0:
        raise ValueError("neighbor_degrees must be positive.")

    original = [float(weight) for weight in weights]
    propagated = [(1.0 - strength) * weight for weight in original]
    for source_index, source_weight in enumerate(original):
        neighbor_scores = []
        for target_index, target_azimuth in enumerate(target_azimuths):
            if target_index == source_index:
                continue
            distance = circular_azimuth_distance(target_azimuths[source_index], target_azimuth)
            if distance <= neighbor_degrees:
                neighbor_scores.append((target_index, max(0.0, 1.0 - distance / neighbor_degrees)))
        score_sum = sum(score for _, score in neighbor_scores)
        if score_sum <= 0:
            propagated[source_index] += strength * source_weight
            continue
        for target_index, score in neighbor_scores:
            propagated[target_index] += strength * source_weight * score / score_sum
    return [min(1.0, max(0.0, weight)) for weight in propagated]


def pose_to_slot_weights(
    azimuth: float,
    elevation: float,
    target_azimuths: Optional[Sequence[float]] = None,
    target_elevations: Optional[Sequence[float]] = None,
    azimuth_sigma: float = 45.0,
    elevation_sigma: float = 25.0,
    min_weight: float = 0.05,
) -> List[float]:
    target_azimuths = list(target_azimuths or ZERO123PLUS_TARGET_AZIMUTHS)
    target_elevations = list(target_elevations or ZERO123PLUS_TARGET_ELEVATIONS["v1.2"])
    if len(target_azimuths) != 6 or len(target_elevations) != 6:
        raise ValueError("Zero123++ target azimuth/elevation lists must both have length 6.")
    if azimuth_sigma <= 0 or elevation_sigma <= 0:
        raise ValueError("azimuth_sigma and elevation_sigma must be positive.")

    weights = []
    for target_azimuth, target_elevation in zip(target_azimuths, target_elevations):
        azimuth_dist = circular_azimuth_distance(azimuth, target_azimuth)
        elevation_dist = abs(float(elevation) - float(target_elevation))
        exponent = -(
            (azimuth_dist ** 2) / (2 * azimuth_sigma ** 2)
            + (elevation_dist ** 2) / (2 * elevation_sigma ** 2)
        )
        weights.append(max(min_weight, math.exp(exponent)))
    return weights


def routed_pose_to_slot_weights(
    azimuth: float,
    elevation: float,
    target_azimuths: Optional[Sequence[float]] = None,
    target_elevations: Optional[Sequence[float]] = None,
    mode: str = "local",
    azimuth_sigma: float = 45.0,
    elevation_sigma: float = 25.0,
    sigma_deg: float = 80.0,
    min_weight: float = 0.05,
    normalize: bool = True,
    elevation_weight: float = 0.25,
    cross_view_propagation_enabled: bool = False,
    cross_view_propagation_strength: float = 0.3,
    cross_view_neighbor_degrees: float = 120.0,
) -> List[float]:
    mode = (mode or "local").strip().lower()
    target_azimuths = list(target_azimuths or ZERO123PLUS_TARGET_AZIMUTHS)
    target_elevations = list(target_elevations or ZERO123PLUS_TARGET_ELEVATIONS["v1.2"])
    if mode == "local":
        weights = pose_to_slot_weights(
            azimuth,
            elevation,
            target_azimuths=target_azimuths,
            target_elevations=target_elevations,
            azimuth_sigma=azimuth_sigma,
            elevation_sigma=elevation_sigma,
            min_weight=min_weight,
        )
    elif mode == "wide":
        weights = wide_pose_to_slot_weights(
            azimuth,
            elevation,
            target_azimuths=target_azimuths,
            target_elevations=target_elevations,
            sigma_deg=sigma_deg,
            min_weight=min_weight,
            normalize=normalize,
            elevation_weight=elevation_weight,
        )
    else:
        raise ValueError(f"Unknown reference_slot_weight_mode: {mode}")
    if cross_view_propagation_enabled:
        weights = propagate_slot_weights(
            weights,
            target_azimuths=target_azimuths,
            strength=cross_view_propagation_strength,
            neighbor_degrees=cross_view_neighbor_degrees,
        )
    return weights


def parse_reference_view_labels(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {os.path.basename(key): normalize_coarse_label(value) for key, value in raw.items()}


def parse_reference_pose_metadata(path: Optional[str]) -> Dict[str, Dict[str, float]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    metadata = {}
    for filename, pose in raw.items():
        if not isinstance(pose, dict) or "azimuth" not in pose or "elevation" not in pose:
            raise ValueError(
                f"Reference metadata for {filename} must contain numeric azimuth and elevation."
            )
        metadata[os.path.basename(filename)] = {
            "azimuth": float(pose["azimuth"]) % 360.0,
            "elevation": float(pose["elevation"]),
        }
    return metadata


def infer_pose_from_filename(path: str) -> Optional[tuple[float, float]]:
    name = os.path.basename(path).lower()
    match = re.search(r"(?:^|_)ref_(\d{1,3})_el([+-]?\d+(?:\.\d+)?)", name)
    if not match:
        return None
    return float(match.group(1)) % 360.0, float(match.group(2))


def load_reference_images(
    refs_path: str,
    view_labels_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    pose_version: str = "v1.2",
    max_image_size: int = 1024,
) -> List[ReferenceImage]:
    label_map = parse_reference_view_labels(view_labels_path)
    pose_map = parse_reference_pose_metadata(metadata_path)
    target_azimuths, target_elevations = get_zero123plus_target_poses(pose_version)
    if os.path.isdir(refs_path):
        paths = [
            os.path.join(refs_path, filename)
            for filename in sorted(os.listdir(refs_path))
            if filename.lower().endswith(IMAGE_EXTENSIONS)
        ]
    else:
        paths = [refs_path]

    references = []
    for path in paths:
        image = Image.open(path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_image_size, max_image_size), Image.Resampling.LANCZOS)
        filename = os.path.basename(path)
        label = label_map.get(filename) or infer_label_from_filename(path)
        view_id = label_to_view_id(label)
        pose = pose_map.get(filename)
        pose_source = "metadata" if pose is not None else "unknown"
        if pose is None:
            parsed_pose = infer_pose_from_filename(path)
            if parsed_pose is not None:
                pose = {"azimuth": parsed_pose[0], "elevation": parsed_pose[1]}
                pose_source = "filename"
        azimuth = pose["azimuth"] if pose is not None else None
        elevation = pose["elevation"] if pose is not None else None
        if azimuth is not None and elevation is not None:
            pose_weights = pose_to_slot_weights(
                azimuth,
                elevation,
                target_azimuths=target_azimuths,
                target_elevations=target_elevations,
            )
            view_id = max(range(6), key=lambda slot: pose_weights[slot])
        references.append(ReferenceImage(
            image=image.copy(),
            path=path,
            view_id=view_id,
            view_label=label,
            azimuth=azimuth,
            elevation=elevation,
            pose_source=pose_source,
        ))
    return references


def references_to_pil(references: Iterable[ReferenceImage]) -> List[Image.Image]:
    return [reference.image for reference in references]


def references_to_view_ids(references: Sequence[ReferenceImage]) -> List[int]:
    return [reference.view_id for reference in references]


def references_to_slot_weights(
    references: Sequence[ReferenceImage],
    pose_version: str = "v1.2",
    azimuth_sigma: float = 45.0,
    elevation_sigma: float = 25.0,
    mode: str = "local",
    sigma_deg: float = 80.0,
    min_weight: float = 0.05,
    normalize: bool = True,
    elevation_weight: float = 0.25,
    cross_view_propagation_enabled: bool = False,
    cross_view_propagation_strength: float = 0.3,
    cross_view_neighbor_degrees: float = 120.0,
) -> List[List[float]]:
    target_azimuths, target_elevations = get_zero123plus_target_poses(pose_version)
    return [
        routed_pose_to_slot_weights(
            reference.azimuth,
            reference.elevation,
            target_azimuths=target_azimuths,
            target_elevations=target_elevations,
            mode=mode,
            azimuth_sigma=azimuth_sigma,
            elevation_sigma=elevation_sigma,
            sigma_deg=sigma_deg,
            min_weight=min_weight,
            normalize=normalize,
            elevation_weight=elevation_weight,
            cross_view_propagation_enabled=cross_view_propagation_enabled,
            cross_view_propagation_strength=cross_view_propagation_strength,
            cross_view_neighbor_degrees=cross_view_neighbor_degrees,
        )
        if reference.azimuth is not None and reference.elevation is not None
        else label_to_slot_weights(reference.view_label)
        for reference in references
    ]
