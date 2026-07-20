import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps


ZERO123PLUS_AZIMUTHS = [30, 90, 150, 210, 270, 330]
VIEW_TO_TILE_INDICES = {
    "front": [0, 5],
    "side": [1, 4],
    "back": [2, 3],
    "unknown": [0, 1, 2, 3, 4, 5],
}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")


@dataclass
class ReferenceImage:
    image: Image.Image
    path: str
    view_label: str = "unknown"


@dataclass
class CandidateScore:
    seed: int
    total_score: float
    input_preservation_score: float
    rag_reference_score: float
    inconsistency_penalty: float


def normalize_view_label(label: str) -> str:
    label = (label or "unknown").strip().lower()
    return label if label in VIEW_TO_TILE_INDICES else "unknown"


def infer_view_label_from_filename(path: str) -> str:
    name = os.path.basename(path).lower().replace("-", "_").replace(".", "_")
    parts = name.split("_")
    if any(part in ("front", "frontal") for part in parts):
        return "front"
    if any(part in ("back", "rear") for part in parts):
        return "back"
    if any(part in ("side", "left", "right", "profile") for part in parts):
        return "side"
    return "unknown"


def parse_seed_list(seed_text: Optional[str], default_seed: int, num_seeds: int) -> List[int]:
    if seed_text:
        return [int(seed.strip()) for seed in seed_text.split(",") if seed.strip()]
    return [default_seed + offset for offset in range(num_seeds)]


def load_view_label_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {os.path.basename(key): normalize_view_label(value) for key, value in raw.items()}


def load_reference_images(
    refs_path: str,
    view_labels_path: Optional[str] = None,
    max_image_size: int = 1024,
) -> List[ReferenceImage]:
    label_map = load_view_label_map(view_labels_path)
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
        view_label = normalize_view_label(label_map.get(filename) or infer_view_label_from_filename(path))
        references.append(ReferenceImage(image=image.copy(), path=path, view_label=view_label))
    return references


def split_zero123plus_sheet(sheet: Image.Image) -> List[Image.Image]:
    """Split the fixed Zero123++ 3x2 sheet into six row-major target-view tiles."""
    sheet = sheet.convert("RGB")
    width, height = sheet.size
    tile_width = width // 2
    tile_height = height // 3
    tiles = []
    for row in range(3):
        for col in range(2):
            left = col * tile_width
            upper = row * tile_height
            tiles.append(sheet.crop((left, upper, left + tile_width, upper + tile_height)))
    return tiles


def generate_zero123plus_candidate(
    pipeline,
    input_image: Image.Image,
    num_inference_steps: int,
    device: torch.device,
    seed: int,
    **pipeline_kwargs,
) -> Image.Image:
    """Generate one unmodified Zero123++ candidate.

    Safe RAG baseline: retrieved images are not passed into this function and are
    not injected into Zero123++ conditioning. Retrieval is used only after this
    call to rerank complete generated sheets.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    return pipeline(
        input_image,
        num_inference_steps=num_inference_steps,
        generator=generator,
        **pipeline_kwargs,
    ).images[0]


@torch.no_grad()
def encode_images_clip(
    images: Sequence[Image.Image],
    feature_extractor,
    vision_encoder,
    device: torch.device,
) -> torch.Tensor:
    if not images:
        return torch.empty((0, 1), device=device)
    dtype = next(vision_encoder.parameters()).dtype
    pixel_values = feature_extractor(images=[image.convert("RGB") for image in images], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device=device, dtype=dtype)
    embeds = vision_encoder(pixel_values, output_hidden_states=False).image_embeds
    return torch.nn.functional.normalize(embeds.float(), dim=-1)


def _pairwise_mean_similarity(embeds: torch.Tensor) -> float:
    # TODO: use a geometry-aware consistency metric. Pairwise CLIP similarity can
    # over-penalize correct front/back view differences, so the safe baseline
    # leaves the inconsistency penalty disabled.
    del embeds
    return 0.0


def compute_rag_score(
    input_embed: torch.Tensor,
    tile_embeds: torch.Tensor,
    ref_embeds: torch.Tensor,
    references: Sequence[ReferenceImage],
    rag_weight: float = 0.2,
) -> CandidateScore:
    input_sims = tile_embeds @ input_embed.squeeze(0)
    input_preservation_score = float(input_sims.mean().item())

    rag_scores = []
    for ref_embed, reference in zip(ref_embeds, references):
        tile_indices = VIEW_TO_TILE_INDICES[normalize_view_label(reference.view_label)]
        view_weight = 0.5 if normalize_view_label(reference.view_label) == "unknown" else 1.0
        similarities = tile_embeds[tile_indices] @ ref_embed
        rag_scores.append(float(similarities.max().item()) * view_weight)
    rag_reference_score = float(np.mean(rag_scores)) if rag_scores else 0.0

    inconsistency_penalty = _pairwise_mean_similarity(tile_embeds)
    total_score = input_preservation_score + rag_weight * rag_reference_score - inconsistency_penalty
    return CandidateScore(
        seed=-1,
        total_score=total_score,
        input_preservation_score=input_preservation_score,
        rag_reference_score=rag_reference_score,
        inconsistency_penalty=inconsistency_penalty,
    )


def select_best_candidate(
    candidates: Sequence[Tuple[int, Image.Image]],
    input_image: Image.Image,
    references: Sequence[ReferenceImage],
    feature_extractor,
    vision_encoder,
    device: torch.device,
    rag_weight: float = 0.2,
) -> Tuple[Image.Image, CandidateScore, List[CandidateScore]]:
    input_embed = encode_images_clip([input_image], feature_extractor, vision_encoder, device)
    ref_embeds = encode_images_clip([reference.image for reference in references], feature_extractor, vision_encoder, device)

    scored_candidates: List[CandidateScore] = []
    best_image = None
    best_score = None

    for seed, sheet in candidates:
        tiles = split_zero123plus_sheet(sheet)
        tile_embeds = encode_images_clip(tiles, feature_extractor, vision_encoder, device)
        score = compute_rag_score(input_embed, tile_embeds, ref_embeds, references, rag_weight=rag_weight)
        score.seed = seed
        scored_candidates.append(score)
        if best_score is None or score.total_score > best_score.total_score:
            best_score = score
            best_image = sheet

    if best_image is None or best_score is None:
        raise ValueError("No candidates were provided for RAG reranking.")
    return best_image, best_score, scored_candidates


def log_reference_images(references: Iterable[ReferenceImage]) -> None:
    for reference in references:
        print(f"[RAG-RERANK] reference={reference.path} view_label={reference.view_label}")


def log_candidate_scores(scores: Iterable[CandidateScore]) -> None:
    for score in scores:
        print(
            "[RAG-RERANK] "
            f"seed={score.seed} "
            f"score={score.total_score:.4f} "
            f"input={score.input_preservation_score:.4f} "
            f"rag={score.rag_reference_score:.4f} "
            f"penalty={score.inconsistency_penalty:.4f}"
        )
