import os
from typing import Any, Dict, Optional
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers

import numpy
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.distributed
import transformers
from collections import OrderedDict
from PIL import Image
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    UNet2DConditionModel,
    ImagePipelineOutput
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.attention_processor import Attention, AttnProcessor, XFormersAttnProcessor, AttnProcessor2_0
from diffusers.utils.import_utils import is_xformers_available
from .reference_adapter import (
    append_reference_tokens_to_prompt,
    expand_spatial_mask_batch,
    ref_slot_weights_to_spatial_mask,
)


def should_use_xformers():
    return is_xformers_available() and os.environ.get("ZERO123PLUS_DISABLE_XFORMERS", "").lower() not in ("1", "true", "yes")


def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        img = numpy.random.randint(255, 256, size=[rgba.size[1], rgba.size[0], 3], dtype=numpy.uint8)
        img = Image.fromarray(img, 'RGB')
        img.paste(rgba, mask=rgba.getchannel('A'))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)


class ReferenceOnlyAttnProc(torch.nn.Module):
    def __init__(
        self,
        chained_proc,
        enabled=False,
        name=None
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.chained_proc = chained_proc
        self.name = name

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None,
        mode="w", ref_dict: dict = None, is_cfg_guidance = False
    ) -> Any:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        if self.enabled and is_cfg_guidance:
            res0 = self.chained_proc(attn, hidden_states[:1], encoder_hidden_states[:1], attention_mask)
            hidden_states = hidden_states[1:]
            encoder_hidden_states = encoder_hidden_states[1:]
        if self.enabled:
            if mode == 'w':
                ref_dict[self.name] = encoder_hidden_states
            elif mode == 'r':
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref_dict.pop(self.name)], dim=1)
            elif mode == 'm':
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref_dict[self.name]], dim=1)
            else:
                assert False, mode
        res = self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask)
        if self.enabled and is_cfg_guidance:
            res = torch.cat([res0, res])
        return res


class RefOnlyNoisedUNet(torch.nn.Module):
    def __init__(self, unet: UNet2DConditionModel, train_sched: DDPMScheduler, val_sched: EulerAncestralDiscreteScheduler) -> None:
        super().__init__()
        self.unet = unet
        self.train_sched = train_sched
        self.val_sched = val_sched
        self._reference_val_scheduler_debug_printed = False

        unet_lora_attn_procs = dict()
        for name, _ in unet.attn_processors.items():
            if torch.__version__ >= '2.0':
                default_attn_proc = AttnProcessor2_0()
            elif should_use_xformers():
                default_attn_proc = XFormersAttnProcessor()
            else:
                default_attn_proc = AttnProcessor()
            unet_lora_attn_procs[name] = ReferenceOnlyAttnProc(
                default_attn_proc, enabled=name.endswith("attn1.processor"), name=name
            )
        unet.set_attn_processor(unet_lora_attn_procs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward_cond(self, noisy_cond_lat, timestep, encoder_hidden_states, class_labels, ref_dict, is_cfg_guidance, **kwargs):
        if is_cfg_guidance:
            encoder_hidden_states = encoder_hidden_states[1:]
            class_labels = class_labels[1:]
        self.unet(
            noisy_cond_lat, timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            cross_attention_kwargs=dict(mode="w", ref_dict=ref_dict),
            **kwargs
        )

    @staticmethod
    def _scheduler_timestep_membership(timestep, scheduler):
        scheduler_timesteps = getattr(scheduler, "timesteps", None)
        if scheduler_timesteps is None or scheduler_timesteps.numel() == 0:
            return torch.zeros_like(timestep.reshape(-1), dtype=torch.bool)
        timestep_flat = timestep.detach().reshape(-1).to(device=scheduler_timesteps.device)
        timestep_flat = timestep_flat.to(dtype=scheduler_timesteps.dtype)
        return (timestep_flat.unsqueeze(-1) == scheduler_timesteps.reshape(1, -1)).any(dim=1)

    @staticmethod
    def _scheduler_timestep_range(scheduler):
        scheduler_timesteps = getattr(scheduler, "timesteps", None)
        if scheduler_timesteps is None or scheduler_timesteps.numel() == 0:
            return None, None, 0
        scheduler_timesteps = scheduler_timesteps.detach().float()
        return (
            float(scheduler_timesteps.min().item()),
            float(scheduler_timesteps.max().item()),
            int(scheduler_timesteps.numel()),
        )

    def _condition_noise_scheduler(self, timestep):
        if self.training:
            return self.train_sched, "training_noise", None

        membership = self._scheduler_timestep_membership(timestep, self.val_sched)
        if not torch.is_floating_point(timestep):
            return self.train_sched, "validation_supervised_noise", membership
        if bool(membership.all().item()):
            return self.val_sched, "validation_inference", membership
        return self.train_sched, "validation_supervised_noise", membership

    def _log_condition_scheduler_choice(self, timestep, scheduler, reason, membership):
        if self.training or self._reference_val_scheduler_debug_printed:
            return
        val_min, val_max, val_count = self._scheduler_timestep_range(self.val_sched)
        timestep_values = timestep.detach().reshape(-1).float().cpu().tolist()
        found_values = None if membership is None else membership.detach().cpu().tolist()
        print("[REFERENCE-VALIDATION] condition scheduler choice")
        print(f"[REFERENCE-VALIDATION] timestep={timestep_values}")
        print(f"[REFERENCE-VALIDATION] val_sched_timesteps_min={val_min} max={val_max} count={val_count}")
        print(f"[REFERENCE-VALIDATION] timestep_found_in_val_sched={found_values}")
        print(f"[REFERENCE-VALIDATION] scheduler_class={scheduler.__class__.__name__}")
        print(f"[REFERENCE-VALIDATION] validation_inference_steps={val_count}")
        print(f"[REFERENCE-VALIDATION] reason={reason}")
        self._reference_val_scheduler_debug_printed = True

    def _noise_condition_latents(self, cond_lat, noise, timestep):
        scheduler, reason, membership = self._condition_noise_scheduler(timestep)
        timestep_flat = timestep.reshape(-1)
        if scheduler is self.val_sched and membership is not None and not bool(membership.all().item()):
            missing = timestep_flat[~membership.to(device=timestep_flat.device)].detach().cpu().tolist()
            val_min, val_max, val_count = self._scheduler_timestep_range(self.val_sched)
            raise ValueError(
                "Validation inference scheduler timestep mismatch: "
                f"missing={missing}, val_sched_min={val_min}, "
                f"val_sched_max={val_max}, val_sched_count={val_count}. "
                "Use the training noise scheduler for supervised validation loss, "
                "or choose timesteps from val_sched.timesteps for image generation."
            )

        self._log_condition_scheduler_choice(timestep, scheduler, reason, membership)
        noisy_cond_lat = scheduler.add_noise(cond_lat, noise, timestep_flat)
        if scheduler is self.val_sched and timestep_flat.numel() > 1 and noisy_cond_lat.shape[0] == timestep_flat.numel():
            return torch.cat(
                [
                    scheduler.scale_model_input(
                        noisy_cond_lat[index : index + 1],
                        timestep_flat[index],
                    )
                    for index in range(timestep_flat.numel())
                ],
                dim=0,
            )
        if scheduler is self.val_sched:
            return scheduler.scale_model_input(noisy_cond_lat, timestep_flat[0])
        return scheduler.scale_model_input(noisy_cond_lat, timestep)

    def forward(
        self, sample, timestep, encoder_hidden_states, class_labels=None,
        *args, cross_attention_kwargs,
        down_block_res_samples=None, mid_block_res_sample=None,
        base_down_block_res_samples=None, base_mid_block_res_sample=None,
        **kwargs
    ):
        cond_lat = cross_attention_kwargs['cond_lat']
        is_cfg_guidance = cross_attention_kwargs.get('is_cfg_guidance', False)
        noise = cross_attention_kwargs.get('reference_condition_noise', None)
        if noise is None:
            noise = torch.randn_like(cond_lat)
        elif noise.shape != cond_lat.shape:
            raise ValueError(
                "reference_condition_noise must match cond_lat shape; "
                f"noise={tuple(noise.shape)}, cond_lat={tuple(cond_lat.shape)}"
            )
        noisy_cond_lat = self._noise_condition_latents(cond_lat, noise, timestep)
        weight_dtype = self.unet.dtype

        def run_prediction(
            current_encoder_hidden_states,
            current_down_block_res_samples=down_block_res_samples,
            current_mid_block_res_sample=mid_block_res_sample,
        ):
            ref_dict = {}
            self.forward_cond(
                noisy_cond_lat, timestep,
                current_encoder_hidden_states, class_labels,
                ref_dict, is_cfg_guidance, **kwargs
            )
            return self.unet(
                sample, timestep,
                current_encoder_hidden_states, *args,
                class_labels=class_labels,
                cross_attention_kwargs=dict(mode="r", ref_dict=ref_dict, is_cfg_guidance=is_cfg_guidance),
                down_block_additional_residuals=[
                    residual.to(dtype=weight_dtype) for residual in current_down_block_res_samples
                ] if current_down_block_res_samples is not None else None,
                mid_block_additional_residual=(
                    current_mid_block_res_sample.to(dtype=weight_dtype)
                    if current_mid_block_res_sample is not None else None
                ),
                **kwargs
            )

        spatial_mask = cross_attention_kwargs.get('reference_spatial_mask', None)
        base_encoder_hidden_states = cross_attention_kwargs.get('reference_base_encoder_hidden_states', None)
        if spatial_mask is None or base_encoder_hidden_states is None:
            return run_prediction(encoder_hidden_states)

        ref_out = run_prediction(encoder_hidden_states)
        base_out = run_prediction(
            base_encoder_hidden_states,
            current_down_block_res_samples=(
                base_down_block_res_samples
                if base_down_block_res_samples is not None
                else down_block_res_samples
            ),
            current_mid_block_res_sample=(
                base_mid_block_res_sample
                if base_mid_block_res_sample is not None
                else mid_block_res_sample
            ),
        )
        if not isinstance(ref_out, tuple) or not isinstance(base_out, tuple):
            raise TypeError("reference spatial gating expects UNet return_dict=False tuple outputs during inference.")

        ref_pred = ref_out[0]
        base_pred = base_out[0]
        spatial_mask = spatial_mask.to(device=ref_pred.device, dtype=ref_pred.dtype)
        spatial_mask = expand_spatial_mask_batch(spatial_mask, ref_pred)
        final_pred = base_pred + spatial_mask * (ref_pred - base_pred)

        if cross_attention_kwargs.get('reference_debug_dump', False) and not getattr(self, "_reference_inference_spatial_debug_printed", False):
            print("[REFERENCE-ADAPTER] spatial gating enabled during inference")
            print("[REFERENCE-ADAPTER] expected extra inference cost: about 2x UNet calls per denoising step")
            print(f"[REFERENCE-ADAPTER] latent spatial mask shape: {tuple(spatial_mask.shape)}")
            print(f"[REFERENCE-ADAPTER] base_pred shape: {tuple(base_pred.shape)}")
            print(f"[REFERENCE-ADAPTER] ref_pred shape: {tuple(ref_pred.shape)}")
            print(f"[REFERENCE-ADAPTER] final_pred shape: {tuple(final_pred.shape)}")
            self._reference_inference_spatial_debug_printed = True

        return (final_pred, *ref_out[1:])


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


class DepthControlUNet(torch.nn.Module):
    def __init__(self, unet: RefOnlyNoisedUNet, controlnet: Optional[diffusers.ControlNetModel] = None, conditioning_scale=1.0) -> None:
        super().__init__()
        self.unet = unet
        if controlnet is None:
            self.controlnet = diffusers.ControlNetModel.from_unet(unet.unet)
        else:
            self.controlnet = controlnet
        DefaultAttnProc = AttnProcessor2_0
        if should_use_xformers():
            DefaultAttnProc = XFormersAttnProcessor
        self.controlnet.set_attn_processor(DefaultAttnProc())
        self.conditioning_scale = conditioning_scale

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward(self, sample, timestep, encoder_hidden_states, class_labels=None, *args, cross_attention_kwargs: dict, **kwargs):
        cross_attention_kwargs = dict(cross_attention_kwargs)
        control_depth = cross_attention_kwargs.pop('control_depth')
        def run_controlnet(current_encoder_hidden_states):
            return self.controlnet(
                sample,
                timestep,
                encoder_hidden_states=current_encoder_hidden_states,
                controlnet_cond=control_depth,
                conditioning_scale=self.conditioning_scale,
                return_dict=False,
            )

        down_block_res_samples, mid_block_res_sample = run_controlnet(encoder_hidden_states)
        base_down_block_res_samples = None
        base_mid_block_res_sample = None
        base_encoder_hidden_states = cross_attention_kwargs.get('reference_base_encoder_hidden_states')
        if cross_attention_kwargs.get('reference_spatial_mask') is not None and base_encoder_hidden_states is not None:
            base_down_block_res_samples, base_mid_block_res_sample = run_controlnet(base_encoder_hidden_states)
            if (
                cross_attention_kwargs.get('reference_debug_dump', False)
                and not getattr(self, "_reference_controlnet_spatial_debug_printed", False)
            ):
                print("[REFERENCE-ADAPTER] ControlNet residuals computed separately for base/ref branches")
                print("[REFERENCE-ADAPTER] ControlNet base branch uses REFERENCE-free encoder conditioning")
                self._reference_controlnet_spatial_debug_printed = True
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample,
            base_down_block_res_samples=base_down_block_res_samples,
            base_mid_block_res_sample=base_mid_block_res_sample,
            cross_attention_kwargs=cross_attention_kwargs
        )


class ModuleListDict(torch.nn.Module):
    def __init__(self, procs: dict) -> None:
        super().__init__()
        self.keys = sorted(procs.keys())
        self.values = torch.nn.ModuleList(procs[k] for k in self.keys)

    def __getitem__(self, key):
        return self.values[self.keys.index(key)]


class SuperNet(torch.nn.Module):
    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        super().__init__()
        state_dict = OrderedDict((k, state_dict[k]) for k in sorted(state_dict.keys()))
        self.layers = torch.nn.ModuleList(state_dict.values())
        self.mapping = dict(enumerate(state_dict.keys()))
        self.rev_mapping = {v: k for k, v in enumerate(state_dict.keys())}

        # .processor for unet, .self_attn for text encoder
        self.split_keys = [".processor", ".self_attn"]

        # we add a hook to state_dict() and load_state_dict() so that the
        # naming fits with `unet.attn_processors`
        def map_to(module, state_dict, *args, **kwargs):
            new_state_dict = {}
            for key, value in state_dict.items():
                num = int(key.split(".")[1])  # 0 is always "layers"
                new_key = key.replace(f"layers.{num}", module.mapping[num])
                new_state_dict[new_key] = value

            return new_state_dict

        def remap_key(key, state_dict):
            for k in self.split_keys:
                if k in key:
                    return key.split(k)[0] + k
            return key.split('.')[0]

        def map_from(module, state_dict, *args, **kwargs):
            all_keys = list(state_dict.keys())
            for key in all_keys:
                replace_key = remap_key(key, state_dict)
                new_key = key.replace(replace_key, f"layers.{module.rev_mapping[replace_key]}")
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        self._register_state_dict_hook(map_to)
        self._register_load_state_dict_pre_hook(map_from, with_module=True)


class Zero123PlusPipeline(diffusers.StableDiffusionPipeline):
    tokenizer: transformers.CLIPTokenizer
    text_encoder: transformers.CLIPTextModel
    vision_encoder: transformers.CLIPVisionModelWithProjection

    feature_extractor_clip: transformers.CLIPImageProcessor
    unet: UNet2DConditionModel
    scheduler: diffusers.schedulers.KarrasDiffusionSchedulers

    vae: AutoencoderKL
    ramping: nn.Linear

    feature_extractor_vae: transformers.CLIPImageProcessor

    depth_transforms_multi = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        vision_encoder: transformers.CLIPVisionModelWithProjection,
        feature_extractor_clip: CLIPImageProcessor, 
        feature_extractor_vae: CLIPImageProcessor,
        ramping_coefficients: Optional[list] = None,
        safety_checker=None,
    ):
        DiffusionPipeline.__init__(self)

        self.register_modules(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            unet=unet, scheduler=scheduler, safety_checker=None,
            vision_encoder=vision_encoder,
            feature_extractor_clip=feature_extractor_clip,
            feature_extractor_vae=feature_extractor_vae
        )
        self.register_to_config(ramping_coefficients=ramping_coefficients)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def prepare(self):
        train_sched = DDPMScheduler.from_config(self.scheduler.config)
        if isinstance(self.unet, UNet2DConditionModel):
            self.unet = RefOnlyNoisedUNet(self.unet, train_sched, self.scheduler).eval()

    def add_controlnet(self, controlnet: Optional[diffusers.ControlNetModel] = None, conditioning_scale=1.0):
        self.prepare()
        self.unet = DepthControlUNet(self.unet, controlnet, conditioning_scale)
        return SuperNet(OrderedDict([('controlnet', self.unet.controlnet)]))

    def encode_condition_image(self, image: torch.Tensor):
        image = self.vae.encode(image).latent_dist.sample()
        return image

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image = None,
        prompt = "",
        *args,
        num_images_per_prompt: Optional[int] = 1,
        guidance_scale=4.0,
        depth_image: Image.Image = None,
        reference_images=None,
        reference_view_ids=None,
        reference_slot_weights=None,
        reference_valid_mask=None,
        reference_adapter=None,
        reference_token_scale: float = 0.1,
        reference_global_scale: float = 0.05,
        reference_global_token_enabled: bool = True,
        reference_match_scale: float = 1.0,
        reference_near_scale: float = 0.35,
        reference_nonmatch_scale: float = 0.05,
        reference_unknown_scale: float = 0.1,
        reference_spatial_gating: bool = False,
        reference_spatial_gate_scale: float = 1.0,
        reference_debug_dump: bool = False,
        output_type: Optional[str] = "pil",
        width=640,
        height=960,
        num_inference_steps=28,
        return_dict=True,
        **kwargs
    ):
        self.prepare()
        if image is None:
            raise ValueError("Inputting embeddings not supported for this pipeline. Please pass an image.")
        assert not isinstance(image, torch.Tensor)
        image = to_rgb_image(image)
        image_1 = self.feature_extractor_vae(images=image, return_tensors="pt").pixel_values
        image_2 = self.feature_extractor_clip(images=image, return_tensors="pt").pixel_values
        if depth_image is not None and hasattr(self.unet, "controlnet"):
            depth_image = to_rgb_image(depth_image)
            depth_image = self.depth_transforms_multi(depth_image).to(
                device=self.unet.controlnet.device, dtype=self.unet.controlnet.dtype
            )
        image = image_1.to(device=self.vae.device, dtype=self.vae.dtype)
        image_2 = image_2.to(device=self.vae.device, dtype=self.vae.dtype)
        cond_lat = self.encode_condition_image(image)
        if guidance_scale > 1:
            negative_lat = self.encode_condition_image(torch.zeros_like(image))
            cond_lat = torch.cat([negative_lat, cond_lat])
        encoded = self.vision_encoder(image_2, output_hidden_states=False)
        global_embeds = encoded.image_embeds
        global_embeds = global_embeds.unsqueeze(-2)
        
        if hasattr(self, "encode_prompt"):
            encoder_hidden_states = self.encode_prompt(
                prompt,
                self.device,
                num_images_per_prompt,
                False
            )[0]
        else:
            encoder_hidden_states = self._encode_prompt(
                prompt,
                self.device,
                num_images_per_prompt,
                False
            )
        ramp = global_embeds.new_tensor(self.config.ramping_coefficients).unsqueeze(-1)
        encoder_hidden_states = encoder_hidden_states + global_embeds * ramp
        base_encoder_hidden_states = encoder_hidden_states
        negative_prompt_embeds = None
        reference_spatial_mask = None
        if reference_adapter is not None and reference_images:
            ref_images = [to_rgb_image(ref_image) for ref_image in reference_images]
            ref_pixels = self.feature_extractor_clip(images=ref_images, return_tensors="pt").pixel_values
            ref_pixels = ref_pixels.to(device=self.vae.device, dtype=self.vae.dtype)
            ref_embeds = self.vision_encoder(ref_pixels, output_hidden_states=False).image_embeds
            adapter_dtype = next(reference_adapter.parameters()).dtype
            ref_embeds = ref_embeds.to(dtype=adapter_dtype)
            ref_embeds = ref_embeds.unsqueeze(0)
            if reference_view_ids is None:
                reference_view_ids = [6] * len(ref_images)
            reference_view_ids = torch.as_tensor(reference_view_ids, device=ref_embeds.device, dtype=torch.long).unsqueeze(0)
            if reference_slot_weights is not None:
                reference_slot_weights = torch.as_tensor(
                    reference_slot_weights,
                    device=ref_embeds.device,
                    dtype=ref_embeds.dtype,
                ).unsqueeze(0)
            if reference_valid_mask is None:
                reference_valid_mask = torch.ones(
                    ref_embeds.shape[:2], device=ref_embeds.device, dtype=ref_embeds.dtype
                )
            else:
                reference_valid_mask = torch.as_tensor(
                    reference_valid_mask,
                    device=ref_embeds.device,
                    dtype=ref_embeds.dtype,
                ).reshape(1, -1)
            if reference_valid_mask.shape != ref_embeds.shape[:2]:
                raise ValueError(
                    "reference_valid_mask must match the number of reference images; "
                    f"mask={tuple(reference_valid_mask.shape)}, refs={tuple(ref_embeds.shape[:2])}"
                )
            if reference_slot_weights is not None:
                reference_slot_weights = reference_slot_weights * reference_valid_mask.unsqueeze(-1)
            reference_tokens = reference_adapter(
                ref_embeds,
                reference_view_ids,
                ref_slot_weights=reference_slot_weights,
                ref_valid_mask=reference_valid_mask,
                token_scale=reference_token_scale,
                global_scale=reference_global_scale,
                match_scale=reference_match_scale,
                near_scale=reference_near_scale,
                nonmatch_scale=reference_nonmatch_scale,
                unknown_scale=reference_unknown_scale,
                global_token_enabled=reference_global_token_enabled,
            )
            encoder_hidden_states = append_reference_tokens_to_prompt(encoder_hidden_states, reference_tokens)
            if reference_spatial_gating:
                if reference_slot_weights is None:
                    raise ValueError("reference_spatial_gating requires reference_slot_weights.")
                vae_scale_factor = getattr(
                    self,
                    "vae_scale_factor",
                    2 ** (len(self.vae.config.block_out_channels) - 1),
                )
                latent_shape = (
                    reference_slot_weights.shape[0],
                    self.unet.config.in_channels,
                    height // vae_scale_factor,
                    width // vae_scale_factor,
                )
                reference_spatial_mask = ref_slot_weights_to_spatial_mask(
                    reference_slot_weights,
                    latent_shape,
                    gate_scale=reference_spatial_gate_scale,
                )
            if guidance_scale > 1:
                if hasattr(self, "encode_prompt"):
                    negative_base = self.encode_prompt(
                        "",
                        self.device,
                        num_images_per_prompt,
                        False
                    )[0]
                else:
                    negative_base = self._encode_prompt(
                        "",
                        self.device,
                        num_images_per_prompt,
                        False
                    )
                negative_base = negative_base.to(device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
                negative_ref_tokens = torch.zeros(
                    negative_base.shape[0],
                    reference_tokens.shape[1],
                    negative_base.shape[-1],
                    device=negative_base.device,
                    dtype=negative_base.dtype,
                )
                negative_prompt_embeds = torch.cat([negative_base, negative_ref_tokens], dim=1)
                base_encoder_hidden_states = torch.cat([negative_base, base_encoder_hidden_states], dim=0)
            if reference_debug_dump:
                print(
                    "[REFERENCE-ADAPTER] appended reference tokens "
                    f"ref_embeds={tuple(ref_embeds.shape)} "
                    f"view_ids={reference_view_ids.detach().cpu().tolist()} "
                    f"slot_weights={None if reference_slot_weights is None else tuple(reference_slot_weights.shape)} "
                    f"references_valid={int(reference_valid_mask.sum().item())}/{reference_valid_mask.numel()} "
                    f"encoder_hidden_states={tuple(encoder_hidden_states.shape)} "
                    f"negative_prompt_embeds={None if negative_prompt_embeds is None else tuple(negative_prompt_embeds.shape)}"
                )
        cak = dict(cond_lat=cond_lat)
        if reference_spatial_mask is not None:
            cak['reference_spatial_mask'] = reference_spatial_mask
            cak['reference_base_encoder_hidden_states'] = base_encoder_hidden_states
            cak['reference_debug_dump'] = reference_debug_dump
        if hasattr(self.unet, "controlnet"):
            cak['control_depth'] = depth_image
        latents: torch.Tensor = super().__call__(
            None,
            *args,
            cross_attention_kwargs=cak,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=encoder_hidden_states,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            output_type='latent',
            width=width,
            height=height,
            **kwargs
        ).images
        latents = unscale_latents(latents)
        if not output_type == "latent":
            image = unscale_image(self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0])
        else:
            image = latents

        image = self.image_processor.postprocess(image, output_type=output_type)
        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
