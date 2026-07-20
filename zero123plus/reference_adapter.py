from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


COARSE_TO_STRONG_SLOTS = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [],
}


def map_coarse_label_to_slot_weights(
    ref_view_ids: torch.Tensor,
    match_scale: float = 1.0,
    near_scale: float = 0.35,
    nonmatch_scale: float = 0.05,
    unknown_scale: float = 0.1,
) -> torch.Tensor:
    """Return weights shaped (B, R, 6) for fixed Zero123++ target slots."""
    weights = torch.full(
        (*ref_view_ids.shape, 6),
        fill_value=nonmatch_scale,
        device=ref_view_ids.device,
        dtype=torch.float32,
    )
    # Fixed slot ids.
    for slot_id in range(6):
        weights[..., slot_id] = torch.where(
            ref_view_ids == slot_id,
            torch.as_tensor(match_scale, device=ref_view_ids.device),
            weights[..., slot_id],
        )

    # Coarse labels encoded as unknown are weak global only. The current loader
    # maps direct fixed-slot labels to 0..5; coarse front/side/back can be mapped
    # by passing view ids for the strongest slots in the dataset/JSON.
    unknown = ref_view_ids == 6
    weights = torch.where(unknown.unsqueeze(-1), torch.as_tensor(unknown_scale, device=ref_view_ids.device), weights)

    # Add near-view softness between paired front, side, and back slots.
    near_pairs = ((0, 5), (1, 4), (2, 3))
    for a, b in near_pairs:
        weights[..., b] = torch.where(
            ref_view_ids == a,
            torch.maximum(weights[..., b], torch.as_tensor(near_scale, device=ref_view_ids.device)),
            weights[..., b],
        )
        weights[..., a] = torch.where(
            ref_view_ids == b,
            torch.maximum(weights[..., a], torch.as_tensor(near_scale, device=ref_view_ids.device)),
            weights[..., a],
        )
    return weights


class ReferenceAdapter(nn.Module):
    """View-aware reference-token adapter with optional fixed spatial gating.

    This is IP-Adapter-inspired reference conditioning, not a full IP-Adapter:
    it does not add decoupled image cross-attention layers throughout the UNet.
    """

    def __init__(self, embed_dim: int, num_view_ids: int = 7):
        super().__init__()
        self.ref_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.view_embed = nn.Embedding(num_view_ids, embed_dim)

    def forward(
        self,
        ref_embeds: torch.Tensor,
        ref_view_ids: torch.Tensor,
        ref_slot_weights: Optional[torch.Tensor] = None,
        ref_valid_mask: Optional[torch.Tensor] = None,
        token_scale: float = 0.1,
        global_scale: float = 0.05,
        match_scale: float = 1.0,
        near_scale: float = 0.35,
        nonmatch_scale: float = 0.05,
        unknown_scale: float = 0.1,
        global_token_enabled: bool = True,
    ) -> torch.Tensor:
        if ref_embeds.numel() == 0:
            return ref_embeds.new_zeros((ref_embeds.shape[0], 0, ref_embeds.shape[-1]))

        ref_view_ids = ref_view_ids.to(device=ref_embeds.device, dtype=torch.long)
        ref_tokens = self.ref_proj(ref_embeds) + self.view_embed(ref_view_ids)
        if ref_valid_mask is None:
            valid_mask = torch.ones(
                ref_embeds.shape[:2], device=ref_embeds.device, dtype=ref_tokens.dtype
            )
        else:
            valid_mask = ref_valid_mask.to(device=ref_embeds.device, dtype=ref_tokens.dtype)
            if valid_mask.shape != ref_embeds.shape[:2]:
                raise ValueError(
                    "ref_valid_mask must have shape (B, R), got "
                    f"{tuple(valid_mask.shape)} for ref_embeds {tuple(ref_embeds.shape)}"
                )
            valid_mask = valid_mask.clamp(0.0, 1.0)

        valid_count = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        global_token = (
            (ref_tokens * valid_mask.unsqueeze(-1)).sum(dim=1, keepdim=True)
            / valid_count.unsqueeze(-1)
        )
        if global_token_enabled:
            global_token = global_token * global_scale
        else:
            global_token = torch.zeros_like(global_token)

        if ref_slot_weights is not None:
            route_weights = ref_slot_weights.to(device=ref_tokens.device, dtype=ref_tokens.dtype)
        else:
            route_weights = map_coarse_label_to_slot_weights(
                ref_view_ids,
                match_scale=match_scale,
                near_scale=near_scale,
                nonmatch_scale=nonmatch_scale,
                unknown_scale=unknown_scale,
            ).to(dtype=ref_tokens.dtype)
        route_weights = route_weights * valid_mask.unsqueeze(-1)
        weighted_tokens = route_weights.unsqueeze(-1) * ref_tokens.unsqueeze(2)
        denom = route_weights.sum(dim=1).clamp_min(1e-6).unsqueeze(-1)
        slot_tokens = weighted_tokens.sum(dim=1) / denom
        slot_tokens = slot_tokens * token_scale
        return torch.cat([global_token, slot_tokens], dim=1)


def append_reference_tokens_to_prompt(
    prompt_embeds: torch.Tensor,
    reference_tokens: Optional[torch.Tensor],
) -> torch.Tensor:
    if reference_tokens is None or reference_tokens.numel() == 0:
        return prompt_embeds
    reference_tokens = reference_tokens.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
    return torch.cat([prompt_embeds, reference_tokens], dim=1)


def make_zero123plus_tile_masks(
    latent_height: int,
    latent_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Fixed row-major masks for the Zero123++ 3x2 sheet layout.

    Slot order is [30, 90, 150, 210, 270, 330]:
      row 0: 30, 90
      row 1: 150, 210
      row 2: 270, 330
    """
    masks = torch.zeros((6, 1, latent_height, latent_width), device=device, dtype=dtype)
    row_edges = [round(latent_height * idx / 3) for idx in range(4)]
    col_edges = [round(latent_width * idx / 2) for idx in range(3)]
    slot = 0
    for row in range(3):
        for col in range(2):
            masks[
                slot,
                :,
                row_edges[row]:row_edges[row + 1],
                col_edges[col]:col_edges[col + 1],
            ] = 1.0
            slot += 1
    return masks


def ref_slot_weights_to_spatial_mask(
    ref_slot_weights: torch.Tensor,
    latent_shape: tuple[int, int, int, int],
    gate_scale: float = 1.0,
) -> torch.Tensor:
    """Convert pose-aware ref weights (B,R,6) into fixed latent tile gates (B,1,H,W)."""
    batch_size, _, latent_height, latent_width = latent_shape
    slot_weights = ref_slot_weights.to(dtype=torch.float32)
    if slot_weights.ndim != 3 or slot_weights.shape[-1] != 6:
        raise ValueError(f"ref_slot_weights must have shape (B, R, 6), got {tuple(slot_weights.shape)}")
    slot_weights = slot_weights.max(dim=1).values.clamp(0.0, 1.0) * gate_scale
    slot_weights = slot_weights.clamp(0.0, 1.0)
    masks = make_zero123plus_tile_masks(
        latent_height,
        latent_width,
        device=slot_weights.device,
        dtype=slot_weights.dtype,
    )
    spatial_mask = (slot_weights.view(batch_size, 6, 1, 1, 1) * masks.unsqueeze(0)).sum(dim=1)
    return spatial_mask


def expand_spatial_mask_batch(spatial_mask: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Repeat a spatial mask across prediction batches with explicit shape validation."""
    mask_batch = spatial_mask.shape[0]
    prediction_batch = prediction.shape[0]
    if mask_batch <= 0 or prediction_batch % mask_batch != 0:
        raise ValueError(
            "Prediction batch size must be divisible by spatial mask batch size; "
            f"prediction shape={tuple(prediction.shape)}, spatial mask shape={tuple(spatial_mask.shape)}"
        )
    if mask_batch == prediction_batch:
        return spatial_mask
    return spatial_mask.repeat_interleave(prediction_batch // mask_batch, dim=0)


def maybe_unfreeze_cross_attention(unet: nn.Module, limit: int = 4) -> int:
    unfrozen = 0
    for name, param in unet.named_parameters():
        lowered = name.lower()
        if "attn2" in lowered and ("to_k" in lowered or "to_v" in lowered):
            param.requires_grad = True
            unfrozen += 1
            if unfrozen >= limit:
                break
    return unfrozen
