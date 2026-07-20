import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .reference_utils import (
    REFERENCE_SLOT_LABELS,
    ZERO123PLUS_TARGET_AZIMUTHS,
    load_reference_images,
    references_to_pil,
    references_to_slot_weights,
    references_to_view_ids,
)


def _adapter_state_dict(checkpoint):
    state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("Adapter checkpoint must be a state_dict or contain a 'state_dict' entry.")
    adapter_state = {}
    for key, value in state_dict.items():
        for prefix in (
            'reference_adapter.',
            'model.reference_adapter.',
            'rag_adapter.',
            'model.rag_adapter.',
        ):
            if key.startswith(prefix):
                adapter_state[key[len(prefix):]] = value
                break
    if adapter_state:
        return adapter_state
    if any(key.startswith(('ref_proj.', 'view_embed.')) for key in state_dict):
        return state_dict
    raise ValueError("No reference adapter weights found in smoke-test checkpoint.")


def _mean_abs_array_diff(left_array, right_array):
    if left_array.shape != right_array.shape:
        raise ValueError(f"Smoke-test image shapes differ: {left_array.shape} vs {right_array.shape}")
    return float(np.abs(left_array - right_array).mean())


def _output_to_image_and_array(output):
    if torch.is_tensor(output):
        tensor = output.detach().float().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        array = tensor.clamp(0, 1).permute(1, 2, 0).numpy()
        image = Image.fromarray(np.rint(array * 255.0).astype(np.uint8), 'RGB')
        return image, array
    if isinstance(output, np.ndarray):
        array = output[0] if output.ndim == 4 else output
        array = np.clip(array.astype(np.float32), 0.0, 1.0)
        return Image.fromarray(np.rint(array * 255.0).astype(np.uint8), 'RGB'), array
    image = output.convert('RGB')
    return image, np.asarray(image, dtype=np.float32) / 255.0


def _save_comparison_grid(images, path):
    rows = list(images.items())
    font = ImageFont.load_default()
    label_width = 230
    image_width = max(image.width for _, image in rows)
    row_height = max(image.height for _, image in rows)
    grid = Image.new('RGB', (label_width + image_width, row_height * len(rows)), 'white')
    draw = ImageDraw.Draw(grid)
    for row, (label, image) in enumerate(rows):
        y = row * row_height
        draw.text((10, y + 10), label, fill='black', font=font)
        grid.paste(image.convert('RGB'), (label_width, y))
    grid.save(path)


def _save_horizontal_grid(images, path):
    if not images:
        return
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    grid = Image.new('RGB', (width * len(images), height), 'white')
    for index, image in enumerate(images):
        grid.paste(image.convert('RGB').resize((width, height)), (index * width, 0))
    grid.save(path)


def _slot_slices(array):
    height, width = array.shape[:2]
    row_edges = [round(height * idx / 3) for idx in range(4)]
    col_edges = [round(width * idx / 2) for idx in range(3)]
    slices = []
    for row in range(3):
        for col in range(2):
            slices.append((slice(row_edges[row], row_edges[row + 1]), slice(col_edges[col], col_edges[col + 1])))
    return slices


def _per_slot_abs_diff(left_array, right_array):
    if left_array.shape != right_array.shape:
        raise ValueError(f"Smoke-test image shapes differ: {left_array.shape} vs {right_array.shape}")
    diff = np.abs(left_array - right_array)
    values = {}
    for slot, (ys, xs) in enumerate(_slot_slices(diff)):
        values[f"slot_{slot}_az{ZERO123PLUS_TARGET_AZIMUTHS[slot]}"] = float(diff[ys, xs].mean())
    return values


def _save_diff_heatmap(left_array, right_array, path):
    diff = np.abs(left_array - right_array).mean(axis=2)
    max_value = float(diff.max())
    if max_value > 0:
        diff = diff / max_value
    heat = np.zeros((*diff.shape, 3), dtype=np.uint8)
    heat[..., 0] = np.rint(diff * 255).astype(np.uint8)
    heat[..., 1] = np.rint(np.sqrt(diff) * 180).astype(np.uint8)
    heat[..., 2] = np.rint((1.0 - diff) * 40).astype(np.uint8)
    Image.fromarray(heat, 'RGB').save(path)


def _save_per_slot_difference_grid(per_slot_report, path):
    comparisons = list(per_slot_report.items())
    font = ImageFont.load_default()
    cell_w = 140
    cell_h = 54
    label_w = 210
    grid = Image.new('RGB', (label_w + cell_w * 6, cell_h * len(comparisons)), 'white')
    draw = ImageDraw.Draw(grid)
    for row, (name, slot_values) in enumerate(comparisons):
        y = row * cell_h
        draw.text((10, y + 18), name, fill='black', font=font)
        max_value = max(slot_values.values()) if slot_values else 0.0
        for slot in range(6):
            key = f"slot_{slot}_az{ZERO123PLUS_TARGET_AZIMUTHS[slot]}"
            value = float(slot_values.get(key, 0.0))
            intensity = int(255 * value / max_value) if max_value > 0 else 0
            color = (255, 255 - intensity, 255 - intensity)
            x = label_w + slot * cell_w
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), fill=color, outline='black')
            draw.text((x + 8, y + 8), REFERENCE_SLOT_LABELS[slot], fill='black', font=font)
            draw.text((x + 8, y + 28), f"{value:.6f}", fill='black', font=font)
    grid.save(path)


def _getattr(model, name, default):
    return getattr(model, name, default)


def _routing_kwargs(model):
    return {
        'mode': _getattr(model, 'reference_slot_weight_mode', 'local'),
        'sigma_deg': _getattr(model, 'reference_slot_weight_sigma_deg', 80.0),
        'min_weight': _getattr(model, 'reference_slot_weight_min', 0.05),
        'normalize': _getattr(model, 'reference_slot_weight_normalize', True),
        'elevation_weight': _getattr(model, 'reference_slot_weight_elevation_weight', 0.25),
        'cross_view_propagation_enabled': _getattr(model, 'reference_cross_view_propagation_enabled', False),
        'cross_view_propagation_strength': _getattr(model, 'reference_cross_view_propagation_strength', 0.3),
        'cross_view_neighbor_degrees': _getattr(model, 'reference_cross_view_neighbor_degrees', 120.0),
    }


def _slot_weight_report(references, slot_weights, model):
    rows = []
    for reference, weights in zip(references, slot_weights):
        ranked = sorted(range(6), key=lambda slot: weights[slot], reverse=True)
        rows.append({
            'filename': os.path.basename(reference.path),
            'path': reference.path,
            'azimuth': reference.azimuth,
            'elevation': reference.elevation,
            'pose_source': reference.pose_source,
            'valid_mask': 1.0,
            'slot_weights': {
                f"slot_{slot}_az{ZERO123PLUS_TARGET_AZIMUTHS[slot]}": float(weights[slot])
                for slot in range(6)
            },
            'top_affected_slots': [
                {
                    'slot': int(slot),
                    'azimuth': ZERO123PLUS_TARGET_AZIMUTHS[slot],
                    'weight': float(weights[slot]),
                }
                for slot in ranked[:3]
            ],
        })
    return {
        'routing_mode': _getattr(model, 'reference_slot_weight_mode', 'local'),
        'sigma_deg': _getattr(model, 'reference_slot_weight_sigma_deg', 80.0),
        'min_weight': _getattr(model, 'reference_slot_weight_min', 0.05),
        'normalize': _getattr(model, 'reference_slot_weight_normalize', True),
        'elevation_weight': _getattr(model, 'reference_slot_weight_elevation_weight', 0.25),
        'global_token_enabled': _getattr(model, 'reference_global_token_enabled', True),
        'global_token_scale': _getattr(
            model,
            'reference_global_token_scale',
            _getattr(model, 'reference_global_scale', 0.05),
        ),
        'cross_view_propagation_enabled': _getattr(model, 'reference_cross_view_propagation_enabled', False),
        'cross_view_propagation_strength': _getattr(model, 'reference_cross_view_propagation_strength', 0.3),
        'cross_view_neighbor_degrees': _getattr(model, 'reference_cross_view_neighbor_degrees', 120.0),
        'references': rows,
    }


@torch.inference_mode()
def run_post_train_reference_smoke_test(model, checkpoint_path=None):
    """Run a small deterministic reference ablation using the model already resident on device."""
    output_dir = Path(model.logdir) / 'reference_smoke_test'
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'difference_report.json'

    if model.reference_adapter is None:
        raise RuntimeError("reference smoke test requires an initialized reference_adapter.")
    if not model.reference_smoke_test_cond_image or not model.reference_smoke_test_reference_dir:
        raise ValueError("reference smoke test requires reference_smoke_test_cond_image and reference_smoke_test_reference_dir.")

    checkpoint_label = 'in_memory_adapter'
    if checkpoint_path:
        checkpoint_path = os.path.abspath(checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"reference smoke-test checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        incompatible = model.reference_adapter.load_state_dict(_adapter_state_dict(checkpoint), strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "reference smoke-test checkpoint is incompatible: "
                f"missing={list(incompatible.missing_keys)}, unexpected={list(incompatible.unexpected_keys)}"
            )
        checkpoint_label = checkpoint_path

    condition_image = Image.open(model.reference_smoke_test_cond_image).convert('RGB')
    references = load_reference_images(
        model.reference_smoke_test_reference_dir,
        metadata_path=model.reference_smoke_test_metadata,
        pose_version='v1.2',
    )
    if not references:
        raise ValueError(f"No smoke-test references found: {model.reference_smoke_test_reference_dir}")
    reference_images = references_to_pil(references)
    view_ids = references_to_view_ids(references)
    routing_kwargs = _routing_kwargs(model)
    slot_weights = references_to_slot_weights(references, pose_version='v1.2', **routing_kwargs)
    slot_report = _slot_weight_report(references, slot_weights, model)
    (output_dir / 'slot_weight_report.json').write_text(
        json.dumps(slot_report, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    condition_image.save(output_dir / 'input_image.png')
    _save_horizontal_grid(reference_images, output_dir / 'reference_images.png')

    wrong_images = list(reversed(reference_images))
    if len(wrong_images) == 1:
        wrong_images = [ImageOps.mirror(wrong_images[0])]
    # Lightning may move registered submodules (notably the UNet) back to CPU
    # during fit teardown while DiffusionPipeline-owned modules remain on CUDA.
    # Reassemble the complete inference stack on one device before the smoke run.
    if torch.cuda.is_available():
        device = torch.device('cuda', torch.cuda.current_device())
    else:
        device = torch.device(model.device)
    pipeline = model.pipeline
    adapter = model.reference_adapter
    if hasattr(pipeline, 'to'):
        pipeline.to(device)
    adapter.to(device)
    print(f"[REFERENCE-SMOKE] inference device: {device}")
    pipeline_unet_was_training = pipeline.unet.training
    adapter_was_training = adapter.training
    pipeline.unet.eval()
    adapter.eval()

    common_reference = {
        'reference_adapter': adapter,
        'reference_token_scale': model.reference_token_scale,
        'reference_global_scale': _getattr(
            model,
            'reference_global_token_scale',
            _getattr(model, 'reference_global_scale', 0.05),
        ),
        'reference_global_token_enabled': _getattr(model, 'reference_global_token_enabled', True),
        'reference_match_scale': model.reference_match_scale,
        'reference_near_scale': model.reference_near_scale,
        'reference_nonmatch_scale': model.reference_nonmatch_scale,
        'reference_unknown_scale': model.reference_unknown_scale,
        'reference_spatial_gating': model.reference_spatial_gating,
        'reference_spatial_gate_scale': model.reference_spatial_gate_scale,
        'reference_debug_dump': False,
    }

    conditions = {
        'baseline_no_adapter': {},
        'adapter_no_refs': {'reference_adapter': adapter},
        'adapter_correct_refs': {
            **common_reference,
            'reference_images': reference_images,
            'reference_view_ids': view_ids,
            'reference_slot_weights': slot_weights,
        },
        'adapter_wrong_refs': {
            **common_reference,
            'reference_images': wrong_images,
            'reference_view_ids': view_ids,
            'reference_slot_weights': slot_weights,
        },
    }

    images = {}
    float_outputs = {}
    try:
        for name, kwargs in conditions.items():
            torch.manual_seed(model.reference_smoke_test_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(model.reference_smoke_test_seed)
            generator = torch.Generator(device=device).manual_seed(model.reference_smoke_test_seed)
            print(f"[REFERENCE-SMOKE] generating {name} ({model.reference_smoke_test_num_inference_steps} steps)")
            output = pipeline(
                condition_image,
                num_inference_steps=model.reference_smoke_test_num_inference_steps,
                generator=generator,
                output_type='pt',
                **kwargs,
            ).images[0]
            image, float_output = _output_to_image_and_array(output)
            image.save(output_dir / f'{name}.png')
            images[name] = image
            float_outputs[name] = float_output
    finally:
        pipeline.unet.train(pipeline_unet_was_training)
        adapter.train(adapter_was_training)

    report = {
        'adapter_checkpoint_used': checkpoint_label,
        'num_inference_steps': model.reference_smoke_test_num_inference_steps,
        'seed': model.reference_smoke_test_seed,
        'correct_vs_no_refs_mean_abs_diff': _mean_abs_array_diff(
            float_outputs['adapter_correct_refs'], float_outputs['adapter_no_refs']
        ),
        'wrong_vs_correct_mean_abs_diff': _mean_abs_array_diff(
            float_outputs['adapter_wrong_refs'], float_outputs['adapter_correct_refs']
        ),
    }

    per_slot_report = {
        'correct_vs_no_refs': _per_slot_abs_diff(
            float_outputs['adapter_correct_refs'], float_outputs['adapter_no_refs']
        ),
        'wrong_vs_correct': _per_slot_abs_diff(
            float_outputs['adapter_wrong_refs'], float_outputs['adapter_correct_refs']
        ),
    }
    (output_dir / 'per_slot_difference_report.json').write_text(
        json.dumps(per_slot_report, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    _save_per_slot_difference_grid(per_slot_report, output_dir / 'per_slot_difference_grid.png')
    _save_diff_heatmap(
        float_outputs['adapter_correct_refs'],
        float_outputs['adapter_no_refs'],
        output_dir / 'correct_vs_no_refs_diff_heatmap.png',
    )
    _save_diff_heatmap(
        float_outputs['adapter_wrong_refs'],
        float_outputs['adapter_correct_refs'],
        output_dir / 'wrong_vs_correct_diff_heatmap.png',
    )

    threshold = model.reference_smoke_test_difference_threshold
    if report['correct_vs_no_refs_mean_abs_diff'] <= threshold:
        verdict = 'FAIL'
    elif report['wrong_vs_correct_mean_abs_diff'] <= threshold:
        verdict = 'WARNING'
    else:
        verdict = 'PASS'
    report['difference_threshold'] = threshold
    report['verdict'] = verdict
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    _save_comparison_grid(images, output_dir / 'comparison_grid.png')
    object_summary = {
        'objects': [
            {
                'object_id': 'smoke_test_object',
                'input_image': str(output_dir / 'input_image.png'),
                'reference_images': str(output_dir / 'reference_images.png'),
                'no_ref_output': str(output_dir / 'adapter_no_refs.png'),
                'correct_ref_output': str(output_dir / 'adapter_correct_refs.png'),
                'wrong_ref_output': str(output_dir / 'adapter_wrong_refs.png'),
                'per_slot_difference': per_slot_report,
                'slot_weight_report': slot_report,
                'mean_abs_difference': {
                    key: report[key]
                    for key in (
                        'correct_vs_no_refs_mean_abs_diff',
                        'wrong_vs_correct_mean_abs_diff',
                    )
                },
            }
        ],
        'aggregate': {
            'num_objects': 1,
            'verdict': verdict,
            'routing_mode': slot_report['routing_mode'],
        },
    }
    (output_dir / 'object_summary.json').write_text(
        json.dumps(object_summary, indent=2, sort_keys=True),
        encoding='utf-8',
    )

    print("reference smoke test summary:")
    print(f"- adapter checkpoint used: {checkpoint_label}")
    print(f"- correct_vs_no_refs_mean_abs_diff: {report['correct_vs_no_refs_mean_abs_diff']:.8f}")
    print(f"- wrong_vs_correct_mean_abs_diff: {report['wrong_vs_correct_mean_abs_diff']:.8f}")
    print(f"- verdict: {verdict}")
    return report
