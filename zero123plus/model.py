import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from torchvision.transforms import v2
from torchvision.utils import make_grid, save_image
from einops import rearrange

from src.utils.train_util import instantiate_from_config
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, DDPMScheduler, UNet2DConditionModel
from .pipeline import RefOnlyNoisedUNet
from .reference_adapter import (
    ReferenceAdapter,
    append_reference_tokens_to_prompt,
    ref_slot_weights_to_spatial_mask,
    maybe_unfreeze_cross_attention,
)


def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents


def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents


def scale_image(image):
    image = image * 0.5 / 0.8
    return image


def unscale_image(image):
    image = image / 0.5 * 0.8
    return image


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class MVDiffusion(pl.LightningModule):
    def __init__(
        self,
        stable_diffusion_config,
        drop_cond_prob=0.1,
        enable_reference_adapter=False,
        reference_token_scale=0.1,
        reference_global_scale=0.05,
        reference_global_token_enabled=True,
        reference_global_token_scale=None,
        reference_match_scale=1.0,
        reference_near_scale=0.35,
        reference_nonmatch_scale=0.05,
        reference_unknown_scale=0.1,
        reference_train_adapter_only=True,
        reference_unfreeze_crossattn=False,
        reference_spatial_gating=False,
        reference_spatial_gate_scale=1.0,
        reference_debug_dump=False,
        train_debug_image_interval=500,
        reference_debug_metrics_enabled=False,
        reference_debug_metrics_interval=20,
        reference_debug_metrics_filename='reference_debug_metrics.jsonl',
        reference_validation_num_inference_steps=75,
        reference_validation_generate_images=True,
        reference_slot_weight_mode='local',
        reference_slot_weight_sigma_deg=80.0,
        reference_slot_weight_min=0.05,
        reference_slot_weight_normalize=True,
        reference_slot_weight_elevation_weight=0.25,
        reference_cross_view_propagation_enabled=False,
        reference_cross_view_propagation_strength=0.3,
        reference_cross_view_neighbor_degrees=120.0,
    ):
        super(MVDiffusion, self).__init__()

        self.drop_cond_prob = drop_cond_prob
        self.enable_reference_adapter = enable_reference_adapter
        self.reference_token_scale = reference_token_scale
        self.reference_global_scale = reference_global_scale
        self.reference_global_token_enabled = bool(reference_global_token_enabled)
        self.reference_global_token_scale = (
            float(reference_global_scale)
            if reference_global_token_scale is None
            else float(reference_global_token_scale)
        )
        self.reference_match_scale = reference_match_scale
        self.reference_near_scale = reference_near_scale
        self.reference_nonmatch_scale = reference_nonmatch_scale
        self.reference_unknown_scale = reference_unknown_scale
        self.reference_train_adapter_only = reference_train_adapter_only
        self.reference_unfreeze_crossattn = reference_unfreeze_crossattn
        self.reference_spatial_gating = reference_spatial_gating
        self.reference_spatial_gate_scale = reference_spatial_gate_scale
        self.reference_debug_dump = reference_debug_dump
        self.train_debug_image_interval = int(train_debug_image_interval)
        self.reference_debug_metrics_enabled = bool(reference_debug_metrics_enabled or reference_debug_dump)
        self.reference_debug_metrics_interval = max(1, int(reference_debug_metrics_interval))
        self.reference_debug_metrics_filename = str(reference_debug_metrics_filename)
        self.reference_validation_num_inference_steps = int(reference_validation_num_inference_steps)
        self.reference_validation_generate_images = bool(reference_validation_generate_images)
        self.reference_slot_weight_mode = str(reference_slot_weight_mode)
        self.reference_slot_weight_sigma_deg = float(reference_slot_weight_sigma_deg)
        self.reference_slot_weight_min = float(reference_slot_weight_min)
        self.reference_slot_weight_normalize = bool(reference_slot_weight_normalize)
        self.reference_slot_weight_elevation_weight = float(reference_slot_weight_elevation_weight)
        self.reference_cross_view_propagation_enabled = bool(reference_cross_view_propagation_enabled)
        self.reference_cross_view_propagation_strength = float(reference_cross_view_propagation_strength)
        self.reference_cross_view_neighbor_degrees = float(reference_cross_view_neighbor_degrees)
        self._reference_batch_debug_printed = False
        self._reference_optimizer_debug_printed = False
        self._reference_grad_debug_printed = False
        self._reference_mask_debug_printed = False
        self._reference_spatial_train_debug_printed = False
        self._reference_validation_first_generation_seconds = None
        self._reference_forward_debug_metrics = None
        self._reference_pending_debug_metrics = None
        self._target_debug_printed = False

        self.register_schedule()

        # init modules
        pipeline = DiffusionPipeline.from_pretrained(**stable_diffusion_config)
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing='trailing'
        )
        self.pipeline = pipeline

        train_sched = DDPMScheduler.from_config(self.pipeline.scheduler.config)
        if isinstance(self.pipeline.unet, UNet2DConditionModel):
            self.pipeline.unet = RefOnlyNoisedUNet(self.pipeline.unet, train_sched, self.pipeline.scheduler)

        self.train_scheduler = train_sched      # use ddpm scheduler during training

        self.unet = pipeline.unet
        self.reference_adapter = None
        if self.enable_reference_adapter:
            embed_dim = getattr(self.pipeline.vision_encoder.config, "projection_dim", None)
            if embed_dim is None:
                embed_dim = getattr(self.pipeline.vision_encoder.config, "hidden_size", None)
            if embed_dim is None:
                raise ValueError("Could not infer CLIP vision embedding dimension for reference adapter.")
            self.reference_adapter = ReferenceAdapter(embed_dim=embed_dim)
            if self.reference_train_adapter_only:
                for module in (self.pipeline.unet, self.pipeline.vae, self.pipeline.vision_encoder, self.pipeline.text_encoder):
                    for param in module.parameters():
                        param.requires_grad = False
                for param in self.reference_adapter.parameters():
                    param.requires_grad = True
            if self.reference_unfreeze_crossattn:
                unfrozen = maybe_unfreeze_cross_attention(self.pipeline.unet)
                print(f"[REFERENCE-ADAPTER] unfroze {unfrozen} cross-attention parameter tensors")
            if self.reference_spatial_gating:
                print("[REFERENCE-ADAPTER] spatial tile gating enabled; additional trainable params: 0")

        # validation output buffer
        self.validation_step_outputs = []

    def register_schedule(self):
        self.num_timesteps = 1000

        # replace scaled_linear schedule with linear schedule as Zero123++
        beta_start = 0.00085
        beta_end = 0.0120
        betas = torch.linspace(beta_start, beta_end, 1000, dtype=torch.float32)
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]], 0)

        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod).float())
        
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod).float())
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1).float())
    
    def on_fit_start(self):
        device = torch.device(f'cuda:{self.global_rank}')
        self.pipeline.to(device)
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, 'validation'), exist_ok=True)
    
    def prepare_batch_data(self, batch):
        # prepare stable diffusion input
        cond_imgs = batch['cond_imgs']      # (B, C, H, W)
        cond_imgs = cond_imgs.to(self.device)

        # random resize the condition image
        cond_size = np.random.randint(128, 513)
        cond_imgs = v2.functional.resize(cond_imgs, cond_size, interpolation=3, antialias=True).clamp(0, 1)

        target_imgs = batch['target_imgs']  # (B, 6, C, H, W), ordered [30, 90, 150, 210, 270, 330]
        if target_imgs.ndim == 4:
            raise ValueError(
                "target_imgs appears to be an already-concatenated sheet with shape "
                f"{tuple(target_imgs.shape)}. Training expects six separate views shaped "
                "(B, 6, 3, H, W), ordered [30, 90, 150, 210, 270, 330]."
            )
        if target_imgs.ndim != 5 or target_imgs.shape[1] != 6 or target_imgs.shape[2] != 3:
            raise ValueError(
                "target_imgs must have shape (B, 6, 3, H, W); "
                f"got {tuple(target_imgs.shape)}."
            )
        target_shape_before = tuple(target_imgs.shape)
        batch_size, _, channels, _, _ = target_imgs.shape
        target_imgs = v2.functional.resize(target_imgs, 320, interpolation=3, antialias=True).clamp(0, 1)
        # Match the original Zero123++ training/inference sheet layout:
        # row 0: [30, 90], row 1: [150, 210], row 2: [270, 330].
        target_imgs = rearrange(target_imgs, 'b (x y) c h w -> b c (x h) (y w)', x=3, y=2)    # (B, C, 3H, 2W)
        expected_sheet_shape = (batch_size, channels, 3 * 320, 2 * 320)
        if tuple(target_imgs.shape) != expected_sheet_shape:
            raise RuntimeError(
                "Zero123++ target sheet layout conversion failed: expected "
                f"{expected_sheet_shape}, got {tuple(target_imgs.shape)}."
            )
        target_imgs = target_imgs.to(self.device)
        if not self._target_debug_printed and self.global_rank == 0:
            print(f"[ZERO123-TARGET] target_imgs before preprocessing: {target_shape_before}")
            print(f"[ZERO123-TARGET] target sheet after preprocessing: {tuple(target_imgs.shape)}")
            print("[ZERO123-TARGET] target order/layout: row-major [30, 90] / [150, 210] / [270, 330]")
            self._target_debug_printed = True

        return cond_imgs, target_imgs
    
    @torch.no_grad()
    def forward_vision_encoder(self, images):
        dtype = next(self.pipeline.vision_encoder.parameters()).dtype
        image_pil = [v2.functional.to_pil_image(images[i]) for i in range(images.shape[0])]
        image_pt = self.pipeline.feature_extractor_clip(images=image_pil, return_tensors="pt").pixel_values
        image_pt = image_pt.to(device=self.device, dtype=dtype)
        global_embeds = self.pipeline.vision_encoder(image_pt, output_hidden_states=False).image_embeds
        global_embeds = global_embeds.unsqueeze(-2)

        encoder_hidden_states = self.pipeline._encode_prompt("", self.device, 1, False)[0]
        ramp = global_embeds.new_tensor(self.pipeline.config.ramping_coefficients).unsqueeze(-1)
        encoder_hidden_states = encoder_hidden_states + global_embeds * ramp

        return encoder_hidden_states

    @torch.no_grad()
    def encode_reference_images(self, ref_imgs):
        if ref_imgs is None or not self.enable_reference_adapter:
            return None
        if ref_imgs.dim() == 4:
            ref_imgs = ref_imgs.unsqueeze(1)
        bsz, num_refs = ref_imgs.shape[:2]
        flat_refs = rearrange(ref_imgs, 'b r c h w -> (b r) c h w').to(self.device)
        dtype = next(self.pipeline.vision_encoder.parameters()).dtype
        image_pil = [v2.functional.to_pil_image(flat_refs[i].clamp(0, 1)) for i in range(flat_refs.shape[0])]
        image_pt = self.pipeline.feature_extractor_clip(images=image_pil, return_tensors="pt").pixel_values
        image_pt = image_pt.to(device=self.device, dtype=dtype)
        ref_embeds = self.pipeline.vision_encoder(image_pt, output_hidden_states=False).image_embeds
        ref_embeds = ref_embeds.reshape(bsz, num_refs, -1)
        return ref_embeds

    def build_reference_tokens(self, batch):
        if not self.enable_reference_adapter or self.reference_adapter is None or 'ref_imgs' not in batch:
            return None
        ref_imgs = batch['ref_imgs'].to(self.device)
        ref_embeds = self.encode_reference_images(ref_imgs)
        if ref_embeds is None:
            return None
        ref_view_ids = batch.get('ref_view_labels', None)
        if ref_view_ids is None:
            ref_view_ids = torch.full(ref_embeds.shape[:2], 6, device=self.device, dtype=torch.long)
        elif not torch.is_tensor(ref_view_ids):
            ref_view_ids = torch.as_tensor(ref_view_ids, device=self.device, dtype=torch.long)
        else:
            ref_view_ids = ref_view_ids.to(self.device, dtype=torch.long)
        ref_slot_weights = batch.get('ref_slot_weights', None)
        if ref_slot_weights is not None:
            ref_slot_weights = ref_slot_weights.to(self.device, dtype=torch.float32)
            ref_valid_mask = batch.get('ref_valid_mask', None)
            if ref_valid_mask is not None:
                ref_valid_mask = ref_valid_mask.to(self.device, dtype=ref_slot_weights.dtype)
                ref_slot_weights = ref_slot_weights * ref_valid_mask.unsqueeze(-1)
        ref_valid_mask = batch.get('ref_valid_mask', None)
        if ref_valid_mask is None:
            ref_valid_mask = torch.ones(ref_embeds.shape[:2], device=self.device, dtype=torch.float32)
        else:
            ref_valid_mask = ref_valid_mask.to(self.device, dtype=torch.float32)
        if self.reference_debug_dump and self.global_rank == 0 and not self._reference_mask_debug_printed:
            valid_count = int(ref_valid_mask.sum().item())
            total_count = ref_valid_mask.numel()
            print(f"[REFERENCE-ADAPTER] references_valid {valid_count}/{total_count}")
            if valid_count < total_count:
                print("[REFERENCE-ADAPTER] WARNING: padded references detected and excluded from token aggregation")
            print("[REFERENCE-ADAPTER] padded refs contribute neither slot tokens nor the global reference mean")
            self._reference_mask_debug_printed = True
        if not torch.any(ref_valid_mask > 0):
            return None
        reference_tokens = self.reference_adapter(
            ref_embeds,
            ref_view_ids,
            ref_slot_weights=ref_slot_weights,
            ref_valid_mask=ref_valid_mask,
            token_scale=self.reference_token_scale,
            global_scale=self.reference_global_token_scale,
            match_scale=self.reference_match_scale,
            near_scale=self.reference_near_scale,
            nonmatch_scale=self.reference_nonmatch_scale,
            unknown_scale=self.reference_unknown_scale,
            global_token_enabled=self.reference_global_token_enabled,
        )
        if self.reference_debug_dump and self.global_rank == 0:
            print(
                "[REFERENCE-ADAPTER] "
                f"ref_embeds={tuple(ref_embeds.shape)} "
                f"ref_view_ids={ref_view_ids.detach().cpu().tolist()} "
                f"ref_slot_weights={None if ref_slot_weights is None else tuple(ref_slot_weights.shape)} "
                f"reference_tokens={tuple(reference_tokens.shape)}"
            )
        return reference_tokens

    @torch.no_grad()
    def encode_condition_image(self, images):
        dtype = next(self.pipeline.vae.parameters()).dtype
        image_pil = [v2.functional.to_pil_image(images[i]) for i in range(images.shape[0])]
        image_pt = self.pipeline.feature_extractor_vae(images=image_pil, return_tensors="pt").pixel_values
        image_pt = image_pt.to(device=self.device, dtype=dtype)
        latents = self.pipeline.vae.encode(image_pt).latent_dist.sample()
        return latents
    
    @torch.no_grad()
    def encode_target_images(self, images):
        dtype = next(self.pipeline.vae.parameters()).dtype
        # equals to scaling images to [-1, 1] first and then call scale_image
        images = (images - 0.5) / 0.8   # [-0.625, 0.625]
        posterior = self.pipeline.vae.encode(images.to(dtype)).latent_dist
        latents = posterior.sample() * self.pipeline.vae.config.scaling_factor
        latents = scale_latents(latents)
        if self.enable_reference_adapter and self.reference_debug_dump and self.global_rank == 0:
            print(f"[ZERO123-TARGET] target latent shape after VAE encoding: {tuple(latents.shape)}")
        return latents
    
    def forward_unet(
        self,
        latents,
        t,
        prompt_embeds,
        cond_latents,
        reference_tokens=None,
        condition_noise=None,
    ):
        dtype = next(self.pipeline.unet.parameters()).dtype
        latents = latents.to(dtype)
        prompt_embeds = prompt_embeds.to(dtype)
        prompt_embeds = append_reference_tokens_to_prompt(prompt_embeds, reference_tokens)
        cond_latents = cond_latents.to(dtype)
        cross_attention_kwargs = dict(cond_lat=cond_latents)
        if condition_noise is not None:
            cross_attention_kwargs['reference_condition_noise'] = condition_noise
        pred_noise = self.pipeline.unet(
            latents,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]
        return pred_noise

    def _should_collect_reference_metrics(self):
        if not self.reference_debug_metrics_enabled or self.global_rank != 0:
            return False
        step = int(self.global_step) + 1
        return step % self.reference_debug_metrics_interval == 0

    def _capture_reference_prediction_metrics(self, base_pred, ref_pred, final_pred):
        if not self._should_collect_reference_metrics():
            self._reference_forward_debug_metrics = None
            return
        with torch.no_grad():
            self._reference_forward_debug_metrics = {
                'base_pred_mean': float(base_pred.detach().float().mean().item()),
                'ref_pred_mean': float(ref_pred.detach().float().mean().item()),
                'final_pred_mean': float(final_pred.detach().float().mean().item()),
                'reference_delta_mean': float((ref_pred - base_pred).detach().float().abs().mean().item()),
                'final_delta_mean': float((final_pred - base_pred).detach().float().abs().mean().item()),
            }

    @staticmethod
    def _module_grad_norm(module, prefix):
        total_sq = 0.0
        found = False
        for name, param in module.named_parameters():
            if name.startswith(prefix) and param.grad is not None:
                total_sq += float(param.grad.detach().float().pow(2).sum().item())
                found = True
        return total_sq ** 0.5 if found else None

    def _write_reference_debug_metrics(self, metrics):
        if self.global_rank != 0:
            return
        path = os.path.join(self.logdir, self.reference_debug_metrics_filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + '\n')
        print(f"[REFERENCE-METRICS] {json.dumps(metrics, sort_keys=True)}")
        print(f"[REFERENCE-METRICS] appended: {path}")

    def forward_unet_spatial_gated(self, latents, t, prompt_embeds, cond_latents, reference_tokens, ref_slot_weights):
        if (
            not self.reference_spatial_gating
            or reference_tokens is None
            or ref_slot_weights is None
        ):
            prediction = self.forward_unet(
                latents,
                t,
                prompt_embeds,
                cond_latents,
                reference_tokens=reference_tokens,
            )
            # No valid references means the conceptual base/ref/final predictions
            # are identical without paying for a redundant second UNet call.
            if self.reference_spatial_gating and reference_tokens is None:
                if self.reference_adapter is not None:
                    zero_adapter_link = sum(
                        parameter.sum() * 0.0 for parameter in self.reference_adapter.parameters()
                    )
                    prediction = prediction + zero_adapter_link
                self._capture_reference_prediction_metrics(prediction, prediction, prediction)
            else:
                self._reference_forward_debug_metrics = None
            return prediction

        # Both branches must see exactly the same noised condition latent. The
        # wrapped UNet deterministically derives it from this shared noise.
        condition_noise = torch.randn_like(cond_latents)

        with torch.no_grad():
            base_pred = self.forward_unet(
                latents,
                t,
                prompt_embeds,
                cond_latents,
                reference_tokens=None,
                condition_noise=condition_noise,
            )
        effective_reference_tokens = reference_tokens
        global_scale = getattr(
            self,
            'reference_global_token_scale',
            getattr(self, 'reference_global_scale', 0.05),
        )
        if self.reference_token_scale == 0.0 and (
            not getattr(self, 'reference_global_token_enabled', True)
            or global_scale == 0.0
        ):
            effective_reference_tokens = None
        ref_pred = self.forward_unet(
            latents,
            t,
            prompt_embeds,
            cond_latents,
            reference_tokens=effective_reference_tokens,
            condition_noise=condition_noise,
        )
        spatial_mask = ref_slot_weights_to_spatial_mask(
            ref_slot_weights.to(device=ref_pred.device),
            ref_pred.shape,
            gate_scale=self.reference_spatial_gate_scale,
        ).to(device=ref_pred.device, dtype=ref_pred.dtype)
        final_pred = base_pred + spatial_mask * (ref_pred - base_pred)
        self._capture_reference_prediction_metrics(base_pred, ref_pred, final_pred)
        if self.reference_debug_dump and self.global_rank == 0 and not self._reference_spatial_train_debug_printed:
            print(f"[REFERENCE-ADAPTER] spatial gate mask shape: {tuple(spatial_mask.shape)}")
            print(
                f"[REFERENCE-ADAPTER] spatial gate min/max: "
                f"{float(spatial_mask.min().item()):.4f}/{float(spatial_mask.max().item()):.4f}"
            )
            mean_abs_delta = float((ref_pred - base_pred).detach().abs().mean().item())
            print("[REFERENCE-ADAPTER] spatial-gating training reuses identical condition noise for base/ref branches")
            print(f"[REFERENCE-ADAPTER] mean(abs(ref_pred - base_pred)): {mean_abs_delta:.8f}")
            if effective_reference_tokens is None:
                print("[REFERENCE-ADAPTER] zero/disabled reference identity check: base/ref predictions use identical conditioning")
            self._reference_spatial_train_debug_printed = True
        return final_pred
    
    def predict_start_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def get_v(self, x, noise, t):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )
    
    def training_step(self, batch, batch_idx):
        # get input
        cond_imgs, target_imgs = self.prepare_batch_data(batch)
        if self.enable_reference_adapter and not self._reference_batch_debug_printed and self.global_rank == 0:
            print(f"[REFERENCE-ADAPTER] batch has ref_imgs: {'ref_imgs' in batch}")
            if 'ref_imgs' in batch:
                print(f"[REFERENCE-ADAPTER] ref_imgs shape: {tuple(batch['ref_imgs'].shape)}")
            if 'ref_view_labels' in batch:
                print(f"[REFERENCE-ADAPTER] ref_view_labels shape: {tuple(batch['ref_view_labels'].shape)}")
            if 'ref_slot_weights' in batch:
                print(f"[REFERENCE-ADAPTER] ref_slot_weights shape: {tuple(batch['ref_slot_weights'].shape)}")
            if 'ref_valid_mask' in batch:
                valid_count = int(batch['ref_valid_mask'].sum().item())
                total_count = batch['ref_valid_mask'].numel()
                print(f"[REFERENCE-ADAPTER] references_valid {valid_count}/{total_count}")
                if valid_count < total_count:
                    print("[REFERENCE-ADAPTER] WARNING: padded references are present and will be masked out")
            self._reference_batch_debug_printed = True

        # sample random timestep
        B = cond_imgs.shape[0]
        
        t = torch.randint(0, self.num_timesteps, size=(B,)).long().to(self.device)

        # classifier-free guidance
        if np.random.rand() < self.drop_cond_prob:
            prompt_embeds = self.pipeline._encode_prompt([""]*B, self.device, 1, False)
            cond_latents = self.encode_condition_image(torch.zeros_like(cond_imgs))
        else:
            prompt_embeds = self.forward_vision_encoder(cond_imgs)
            cond_latents = self.encode_condition_image(cond_imgs)
        reference_tokens = self.build_reference_tokens(batch)
        ref_slot_weights = batch.get('ref_slot_weights', None)
        if ref_slot_weights is not None:
            ref_slot_weights = ref_slot_weights.to(self.device, dtype=torch.float32)
            ref_valid_mask = batch.get('ref_valid_mask', None)
            if ref_valid_mask is not None:
                ref_slot_weights = ref_slot_weights * ref_valid_mask.to(
                    self.device, dtype=ref_slot_weights.dtype
                ).unsqueeze(-1)

        latents = self.encode_target_images(target_imgs)
        noise = torch.randn_like(latents)
        latents_noisy = self.train_scheduler.add_noise(latents, noise, t)
        
        v_pred = self.forward_unet_spatial_gated(
            latents_noisy,
            t,
            prompt_embeds,
            cond_latents,
            reference_tokens=reference_tokens,
            ref_slot_weights=ref_slot_weights,
        )
        v_target = self.get_v(latents, noise, t)

        loss, loss_dict = self.compute_loss(v_pred, v_target)

        if self._reference_forward_debug_metrics is not None:
            ref_valid_mask = batch.get('ref_valid_mask', None)
            if ref_valid_mask is None and 'ref_imgs' in batch:
                ref_valid_mask = torch.ones(
                    batch['ref_imgs'].shape[:2], device=self.device, dtype=torch.bool
                )
            elif ref_valid_mask is not None:
                ref_valid_mask = ref_valid_mask.to(self.device, dtype=torch.bool)
            ref_valid_count = int(ref_valid_mask.sum().item()) if ref_valid_mask is not None else 0
            valid_slot_weights = None
            raw_slot_weights = batch.get('ref_slot_weights', None)
            if raw_slot_weights is not None:
                raw_slot_weights = raw_slot_weights.to(self.device, dtype=torch.float32)
                if ref_valid_mask is not None:
                    valid_slot_weights = raw_slot_weights[ref_valid_mask]
                else:
                    valid_slot_weights = raw_slot_weights.reshape(-1, raw_slot_weights.shape[-1])
            if valid_slot_weights is None or valid_slot_weights.numel() == 0:
                slot_min = slot_max = slot_mean = 0.0
            else:
                slot_min = float(valid_slot_weights.min().item())
                slot_max = float(valid_slot_weights.max().item())
                slot_mean = float(valid_slot_weights.mean().item())
            self._reference_pending_debug_metrics = {
                'step': int(self.global_step) + 1,
                'loss': float(loss.detach().float().item()),
                **self._reference_forward_debug_metrics,
                'ref_valid_count': ref_valid_count,
                'ref_slot_weights_min': slot_min,
                'ref_slot_weights_max': slot_max,
                'ref_slot_weights_mean': slot_mean,
                'no_valid_reference_check': (
                    'PASS_NEAR_ZERO'
                    if ref_valid_count == 0 and self._reference_forward_debug_metrics['reference_delta_mean'] <= 1e-7
                    else 'NOT_APPLICABLE'
                ),
                'valid_reference_effect_check': (
                    'PASS_FINITE_NONZERO'
                    if ref_valid_count > 0
                    and np.isfinite(self._reference_forward_debug_metrics['reference_delta_mean'])
                    and self._reference_forward_debug_metrics['reference_delta_mean'] > 0.0
                    else 'WARNING_ZERO_OR_NONFINITE' if ref_valid_count > 0 else 'NOT_APPLICABLE'
                ),
            }

        # logging
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if (
            self.train_debug_image_interval > 0
            and self.global_step % self.train_debug_image_interval == 0
            and self.global_rank == 0
        ):
            with torch.no_grad():
                latents_pred = self.predict_start_from_z_and_v(latents_noisy, t, v_pred)
                pred_x0_latents = unscale_latents(latents_pred)
                pred_x0_images = unscale_image(self.pipeline.vae.decode(
                    pred_x0_latents / self.pipeline.vae.config.scaling_factor,
                    return_dict=False,
                )[0])
                pred_x0_images = (pred_x0_images * 0.5 + 0.5).clamp(0, 1)

                image_dir = os.path.join(self.logdir, 'images')
                step_name = f'{self.global_step:07d}'
                save_image(
                    make_grid(cond_imgs, nrow=cond_imgs.shape[0], normalize=True, value_range=(0, 1)),
                    os.path.join(image_dir, f'train_{step_name}_cond_image.png'),
                )
                save_image(
                    make_grid(target_imgs, nrow=target_imgs.shape[0], normalize=True, value_range=(0, 1)),
                    os.path.join(image_dir, f'train_{step_name}_target_ground_truth_3x2_sheet.png'),
                )
                if 'ref_imgs' in batch:
                    ref_imgs = batch['ref_imgs'].to(self.device)
                    ref_imgs = rearrange(ref_imgs, 'b r c h w -> (b r) c h w')
                    save_image(
                        make_grid(ref_imgs, nrow=batch['ref_imgs'].shape[1], normalize=True, value_range=(0, 1)),
                        os.path.join(image_dir, f'train_{step_name}_reference_images.png'),
                    )
                save_image(
                    make_grid(pred_x0_images, nrow=pred_x0_images.shape[0], normalize=True, value_range=(0, 1)),
                    os.path.join(image_dir, f'train_{step_name}_pred_x0_preview_not_final_generation.png'),
                )
                print(
                    f"[TRAIN-PREVIEW] global_step={self.global_step} "
                    f"timestep={t.detach().cpu().tolist()} "
                    "preview_type=one_step_predicted_x0 "
                    f"adapter_enabled={self.enable_reference_adapter} "
                    f"spatial_gating_enabled={self.reference_spatial_gating} "
                    "final_quality=false"
                )
                print(
                    "[TRAIN-PREVIEW] pred_x0_preview_not_final_generation is a one-step "
                    "prediction at a random training timestep and should not be judged as final quality."
                )

        return loss

    def on_after_backward(self):
        if not self.enable_reference_adapter or self.reference_adapter is None:
            return
        if self.global_rank != 0:
            return
        ref_proj_grad = self._module_grad_norm(self.reference_adapter, 'ref_proj')
        view_embed_grad = self._module_grad_norm(self.reference_adapter, 'view_embed')
        if not self._reference_grad_debug_printed:
            print(f"[REFERENCE-ADAPTER] ref_proj grad_norm: {ref_proj_grad}")
            print(f"[REFERENCE-ADAPTER] view_embed grad_norm: {view_embed_grad}")
            print(f"[REFERENCE-ADAPTER] ref_proj grad nonzero: {ref_proj_grad is not None and ref_proj_grad > 0}")
            print(f"[REFERENCE-ADAPTER] view_embed grad nonzero: {view_embed_grad is not None and view_embed_grad > 0}")
            self._reference_grad_debug_printed = True
        if self._reference_pending_debug_metrics is not None:
            metrics = dict(self._reference_pending_debug_metrics)
            metrics['ref_proj_grad_norm'] = ref_proj_grad
            metrics['view_embed_grad_norm'] = view_embed_grad
            self._write_reference_debug_metrics(metrics)
            self._reference_pending_debug_metrics = None
        
    def compute_loss(self, noise_pred, noise_gt):
        loss = F.mse_loss(noise_pred, noise_gt)

        prefix = 'train'
        loss_dict = {}
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def _build_validation_reference_kwargs(self, batch, sample_index):
        """Build per-sample pipeline kwargs and truthfully report whether reference is applied."""
        if not self.enable_reference_adapter or self.reference_adapter is None or 'ref_imgs' not in batch:
            return {}, False

        ref_imgs = batch['ref_imgs'][sample_index]
        valid_mask = batch.get('ref_valid_mask', None)
        if valid_mask is None:
            valid_mask = torch.ones(ref_imgs.shape[0], dtype=torch.bool, device=ref_imgs.device)
        else:
            valid_mask = valid_mask[sample_index].to(device=ref_imgs.device, dtype=torch.bool)
        valid_indices = valid_mask.nonzero(as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            return {}, False

        reference_images = [
            v2.functional.to_pil_image(ref_imgs[index].detach().cpu().clamp(0, 1))
            for index in valid_indices.tolist()
        ]
        ref_view_ids = batch.get('ref_view_labels', None)
        if ref_view_ids is None:
            selected_view_ids = [6] * len(reference_images)
        else:
            selected_view_ids = ref_view_ids[sample_index, valid_indices].detach().cpu().tolist()
        ref_slot_weights = batch.get('ref_slot_weights', None)
        selected_slot_weights = None
        if ref_slot_weights is not None:
            selected_slot_weights = ref_slot_weights[
                sample_index, valid_indices
            ].detach().cpu().tolist()

        return {
            'reference_images': reference_images,
            'reference_view_ids': selected_view_ids,
            'reference_slot_weights': selected_slot_weights,
            'reference_valid_mask': [1.0] * len(reference_images),
            'reference_adapter': self.reference_adapter,
            'reference_token_scale': self.reference_token_scale,
            'reference_global_scale': self.reference_global_token_scale,
            'reference_global_token_enabled': self.reference_global_token_enabled,
            'reference_match_scale': self.reference_match_scale,
            'reference_near_scale': self.reference_near_scale,
            'reference_nonmatch_scale': self.reference_nonmatch_scale,
            'reference_unknown_scale': self.reference_unknown_scale,
            'reference_spatial_gating': self.reference_spatial_gating and selected_slot_weights is not None,
            'reference_spatial_gate_scale': self.reference_spatial_gate_scale,
            'reference_debug_dump': self.reference_debug_dump,
        }, True

    @staticmethod
    def _module_device(module):
        if module is None:
            return 'none'
        try:
            return str(next(module.parameters()).device)
        except StopIteration:
            return 'no_parameters'

    @staticmethod
    def _tensor_device(value):
        if value is None:
            return 'none'
        if torch.is_tensor(value):
            return str(value.device)
        return type(value).__name__

    def _validation_branch_name(self, uses_reference):
        return 'reference' if uses_reference else 'baseline'

    def _log_validation_generation_devices(self, batch, sample_index, branch_name, cond_img):
        ref_imgs = batch.get('ref_imgs', None)
        ref_slot_weights = batch.get('ref_slot_weights', None)
        ref_valid_mask = batch.get('ref_valid_mask', None)
        print(
            "[VALIDATION-GEN-DEVICE] "
            f"batch_idx={getattr(self, '_current_validation_batch_idx', 'unknown')} "
            f"sample_index={sample_index} "
            f"branch={branch_name} "
            f"unet={self._module_device(self.pipeline.unet)} "
            f"vae={self._module_device(self.pipeline.vae)} "
            f"image_encoder={self._module_device(getattr(self.pipeline, 'vision_encoder', None))} "
            f"reference_adapter={self._module_device(self.reference_adapter)} "
            f"cond_tensor={self._tensor_device(cond_img)} "
            f"ref_tensor={self._tensor_device(None if ref_imgs is None else ref_imgs[sample_index])} "
            f"ref_slot_weights={self._tensor_device(None if ref_slot_weights is None else ref_slot_weights[sample_index])} "
            f"ref_valid_mask={self._tensor_device(None if ref_valid_mask is None else ref_valid_mask[sample_index])}"
        )

    def _prepare_validation_generation_devices(self):
        target_device = self.device
        self.pipeline.to(target_device)
        if self.reference_adapter is not None:
            self.reference_adapter.to(target_device)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # get input
        cond_imgs, target_imgs = self.prepare_batch_data(batch)
        batch_size = cond_imgs.shape[0]

        prompt_embeds = self.forward_vision_encoder(cond_imgs)
        cond_latents = self.encode_condition_image(cond_imgs)
        reference_tokens = self.build_reference_tokens(batch)
        ref_slot_weights = batch.get('ref_slot_weights', None)
        if ref_slot_weights is not None:
            ref_slot_weights = ref_slot_weights.to(self.device, dtype=torch.float32)
            ref_valid_mask = batch.get('ref_valid_mask', None)
            if ref_valid_mask is not None:
                ref_slot_weights = ref_slot_weights * ref_valid_mask.to(
                    self.device, dtype=ref_slot_weights.dtype
                ).unsqueeze(-1)

        t = torch.randint(0, self.num_timesteps, size=(batch_size,), device=self.device).long()
        latents = self.encode_target_images(target_imgs)
        noise = torch.randn_like(latents)
        latents_noisy = self.train_scheduler.add_noise(latents, noise, t)
        v_pred = self.forward_unet_spatial_gated(
            latents_noisy,
            t,
            prompt_embeds,
            cond_latents,
            reference_tokens=reference_tokens,
            ref_slot_weights=ref_slot_weights,
        )
        v_target = self.get_v(latents, noise, t)
        val_loss, _ = self.compute_loss(v_pred, v_target)
        self.log(
            'val/loss',
            val_loss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        if not self.reference_validation_generate_images:
            if self.global_rank == 0:
                print(
                    f"[VALIDATION-PREVIEW] skipped image generation at global_step={self.global_step}; "
                    "reference_validation_generate_images=false"
                )
            return

        self._current_validation_batch_idx = int(batch_idx)
        self._prepare_validation_generation_devices()
        images_pil = [v2.functional.to_pil_image(cond_imgs[i]) for i in range(cond_imgs.shape[0])]

        outputs = []
        reference_applied = []
        for sample_index, cond_img in enumerate(images_pil):
            reference_kwargs, sample_uses_reference = self._build_validation_reference_kwargs(batch, sample_index)
            branch_name = self._validation_branch_name(sample_uses_reference)
            if self.global_rank == 0:
                self._log_validation_generation_devices(batch, sample_index, branch_name, cond_imgs[sample_index])
            start_time = time.perf_counter()
            latent = self.pipeline(
                cond_img,
                num_inference_steps=self.reference_validation_num_inference_steps,
                output_type='latent',
                **reference_kwargs,
            ).images
            elapsed = time.perf_counter() - start_time
            seconds_per_step = elapsed / max(1, int(self.reference_validation_num_inference_steps))
            if self.global_rank == 0:
                print(
                    "[VALIDATION-GEN-TIMING] "
                    f"batch_idx={batch_idx} "
                    f"sample_index={sample_index} "
                    f"branch={branch_name} "
                    f"inference_steps={self.reference_validation_num_inference_steps} "
                    f"elapsed_seconds={elapsed:.3f} "
                    f"seconds_per_step={seconds_per_step:.3f}"
                )
                if self._reference_validation_first_generation_seconds is None:
                    self._reference_validation_first_generation_seconds = elapsed
                elif elapsed > 2.0 * self._reference_validation_first_generation_seconds:
                    print(
                        "[VALIDATION-GEN-TIMING] WARNING: generation took more than 2x first generation "
                        f"first_seconds={self._reference_validation_first_generation_seconds:.3f} "
                        f"current_seconds={elapsed:.3f} "
                        f"branch={branch_name}"
                    )
            image = unscale_image(self.pipeline.vae.decode(latent / self.pipeline.vae.config.scaling_factor, return_dict=False)[0])   # [-1, 1]
            image = (image * 0.5 + 0.5).clamp(0, 1)
            outputs.append(image)
            reference_applied.append(sample_uses_reference)
        outputs = torch.cat(outputs, dim=0).to(self.device)
        self.validation_step_outputs.append({
            'cond': cond_imgs,
            'target': target_imgs,
            'full_generation': outputs,
            'reference_applied': reference_applied,
        })
    
    @torch.no_grad()
    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        cond_images = torch.cat([item['cond'] for item in self.validation_step_outputs], dim=0)
        target_images = torch.cat([item['target'] for item in self.validation_step_outputs], dim=0)
        full_generations = torch.cat([item['full_generation'] for item in self.validation_step_outputs], dim=0)
        local_reference_count = sum(sum(item['reference_applied']) for item in self.validation_step_outputs)
        local_output_count = sum(len(item['reference_applied']) for item in self.validation_step_outputs)

        all_cond = rearrange(self.all_gather(cond_images), 'r b c h w -> (r b) c h w')
        all_targets = rearrange(self.all_gather(target_images), 'r b c h w -> (r b) c h w')
        all_generations = rearrange(self.all_gather(full_generations), 'r b c h w -> (r b) c h w')
        all_counts = self.all_gather(torch.tensor(
            [local_reference_count, local_output_count], device=self.device, dtype=torch.long
        )).reshape(-1, 2).sum(dim=0)

        if self.global_rank == 0:
            image_dir = os.path.join(self.logdir, 'validation')
            os.makedirs(image_dir, exist_ok=True)
            step_name = f'{self.global_step:07d}'
            reference_count, output_count = (int(value.item()) for value in all_counts)
            if reference_count == output_count and output_count > 0:
                generation_kind = 'reference'
            elif reference_count == 0:
                generation_kind = 'baseline'
            else:
                generation_kind = 'mixed'
            save_image(
                make_grid(all_cond, nrow=8, normalize=True, value_range=(0, 1)),
                os.path.join(image_dir, f'val_{step_name}_cond_image.png'),
            )
            save_image(
                make_grid(all_targets, nrow=8, normalize=True, value_range=(0, 1)),
                os.path.join(image_dir, f'val_{step_name}_target_ground_truth_3x2_sheet.png'),
            )
            save_image(
                make_grid(all_generations, nrow=8, normalize=True, value_range=(0, 1)),
                os.path.join(
                    image_dir,
                    f'val_{step_name}_{generation_kind}_full_{self.reference_validation_num_inference_steps}_step_generation.png',
                ),
            )
            validation_report = {
                'global_step': int(self.global_step),
                'generation_kind': generation_kind,
                'reference_outputs': reference_count,
                'output_count': output_count,
                'spatial_gating_requested': bool(self.reference_spatial_gating),
                'num_inference_steps': int(self.reference_validation_num_inference_steps),
            }
            with open(os.path.join(image_dir, f'val_{step_name}_report.json'), 'w', encoding='utf-8') as handle:
                json.dump(validation_report, handle, indent=2, sort_keys=True)
            print(
                f"[VALIDATION-PREVIEW] global_step={self.global_step} "
                f"timestep=full_schedule preview_type=full_{self.reference_validation_num_inference_steps}_step_generation "
                f"generation_kind={generation_kind} "
                f"reference_outputs={reference_count}/{output_count} "
                f"spatial_gating_requested={self.reference_spatial_gating} "
                f"num_inference_steps={self.reference_validation_num_inference_steps} "
                "final_quality=true"
            )
            if self.enable_reference_adapter and reference_count == 0:
                print("[VALIDATION-PREVIEW] WARNING: no references were applied; outputs are baseline")

        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        lr = self.learning_rate

        params = [param for param in self.parameters() if param.requires_grad]
        if not params:
            raise ValueError("No trainable parameters found. Check reference adapter training flags.")
        if self.enable_reference_adapter and not self._reference_optimizer_debug_printed:
            trainable_count = sum(param.numel() for param in params)
            ref_proj_trainable = any(
                name.startswith('reference_adapter.ref_proj') and param.requires_grad
                for name, param in self.named_parameters()
            )
            view_embed_trainable = any(
                name.startswith('reference_adapter.view_embed') and param.requires_grad
                for name, param in self.named_parameters()
            )
            print(f"[REFERENCE-ADAPTER] trainable parameter count: {trainable_count}")
            print(f"[REFERENCE-ADAPTER] optimizer includes ref_proj: {ref_proj_trainable}")
            print(f"[REFERENCE-ADAPTER] optimizer includes view_embed: {view_embed_trainable}")
            self._reference_optimizer_debug_printed = True
        optimizer = torch.optim.AdamW(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 3000, eta_min=lr/4)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
