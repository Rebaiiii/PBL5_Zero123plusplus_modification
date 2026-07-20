import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from PIL import Image

from zero123plus.reference_adapter import (
    ReferenceAdapter,
    append_reference_tokens_to_prompt,
    expand_spatial_mask_batch,
    make_zero123plus_tile_masks,
    map_coarse_label_to_slot_weights,
    maybe_unfreeze_cross_attention,
    ref_slot_weights_to_spatial_mask,
)
from zero123plus.model import MVDiffusion
from zero123plus.pipeline import DepthControlUNet
from zero123plus.reference_smoke import _adapter_state_dict, run_post_train_reference_smoke_test
from zero123plus.reference_utils import (
    infer_pose_from_filename,
    label_to_slot_weights,
    load_reference_images,
    propagate_slot_weights,
    references_to_slot_weights,
    routed_pose_to_slot_weights,
)


class ReferenceAdapterTest(unittest.TestCase):
    def test_legacy_checkpoint_prefix_is_still_accepted(self):
        weight = torch.randn(2, 2)
        state = _adapter_state_dict({"state_dict": {"rag_adapter.ref_proj.0.weight": weight}})

        self.assertEqual(list(state), ["ref_proj.0.weight"])
        self.assertIs(state["ref_proj.0.weight"], weight)

    def test_coarse_labels_have_expected_slot_weights(self):
        self.assertEqual(label_to_slot_weights("front"), [1.0, 0.05, 0.05, 0.05, 0.05, 1.0])
        self.assertEqual(label_to_slot_weights("side"), [0.05, 1.0, 0.35, 0.35, 1.0, 0.05])
        self.assertEqual(label_to_slot_weights("back"), [0.05, 0.05, 1.0, 1.0, 0.05, 0.05])
        self.assertEqual(label_to_slot_weights("unknown"), [0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    def test_route_weights_prefer_matching_and_near_slots(self):
        view_ids = torch.tensor([[0, 2, 6]])
        weights = map_coarse_label_to_slot_weights(view_ids)

        self.assertGreater(weights[0, 0, 0], weights[0, 0, 2])
        self.assertGreater(weights[0, 0, 5], weights[0, 0, 2])
        self.assertGreater(weights[0, 1, 2], weights[0, 1, 0])
        self.assertTrue(torch.allclose(weights[0, 2], torch.full((6,), 0.1)))

    def test_adapter_appends_seven_tokens(self):
        adapter = ReferenceAdapter(embed_dim=8)
        ref_embeds = torch.randn(2, 3, 8)
        view_ids = torch.tensor([[0, 1, 2], [6, 6, 6]])
        slot_weights = torch.tensor([
            [
                label_to_slot_weights("front"),
                label_to_slot_weights("side"),
                label_to_slot_weights("back"),
            ],
            [
                label_to_slot_weights("unknown"),
                label_to_slot_weights("unknown"),
                label_to_slot_weights("unknown"),
            ],
        ])
        prompt = torch.randn(2, 77, 8)

        reference_tokens = adapter(ref_embeds, view_ids, ref_slot_weights=slot_weights)
        extended = append_reference_tokens_to_prompt(prompt, reference_tokens)

        self.assertEqual(reference_tokens.shape, (2, 7, 8))
        self.assertEqual(extended.shape, (2, 84, 8))

    def test_padded_references_do_not_affect_token_aggregation(self):
        torch.manual_seed(0)
        adapter = ReferenceAdapter(embed_dim=8)
        valid_embed = torch.randn(1, 1, 8)
        padded_a = torch.zeros(1, 1, 8)
        padded_b = torch.full((1, 1, 8), 1000.0)
        view_ids = torch.tensor([[0, 6]])
        slot_weights = torch.ones(1, 2, 6)
        valid_mask = torch.tensor([[1.0, 0.0]])

        output_a = adapter(
            torch.cat([valid_embed, padded_a], dim=1),
            view_ids,
            ref_slot_weights=slot_weights,
            ref_valid_mask=valid_mask,
        )
        output_b = adapter(
            torch.cat([valid_embed, padded_b], dim=1),
            view_ids,
            ref_slot_weights=slot_weights,
            ref_valid_mask=valid_mask,
        )

        self.assertTrue(torch.allclose(output_a, output_b, atol=1e-6))

    def test_fixed_zero123plus_tile_masks_are_row_major(self):
        masks = make_zero123plus_tile_masks(12, 8, device=torch.device("cpu"), dtype=torch.float32)

        self.assertEqual(masks.shape, (6, 1, 12, 8))
        self.assertEqual(float(masks[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(masks[1, 0, 0, 7]), 1.0)
        self.assertEqual(float(masks[2, 0, 4, 0]), 1.0)
        self.assertEqual(float(masks[3, 0, 4, 7]), 1.0)
        self.assertEqual(float(masks[4, 0, 11, 0]), 1.0)
        self.assertEqual(float(masks[5, 0, 11, 7]), 1.0)
        self.assertEqual(float(masks.sum()), 12 * 8)

    def test_ref_slot_weights_create_spatial_gate(self):
        ref_slot_weights = torch.tensor([[[1.0, 0.05, 0.05, 0.05, 0.05, 0.05]]])
        spatial_mask = ref_slot_weights_to_spatial_mask(ref_slot_weights, (1, 4, 12, 8))

        self.assertEqual(spatial_mask.shape, (1, 1, 12, 8))
        self.assertEqual(float(spatial_mask[0, 0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(spatial_mask[0, 0, 0, 7]), 0.05)
        self.assertAlmostEqual(float(spatial_mask[0, 0, 11, 7]), 0.05)

    def test_spatial_mask_batch_expansion_rejects_invalid_shapes(self):
        mask = torch.ones(2, 1, 12, 8)
        prediction = torch.ones(3, 4, 12, 8)
        with self.assertRaisesRegex(ValueError, "prediction shape=.*spatial mask shape"):
            expand_spatial_mask_batch(mask, prediction)

    def test_spatial_training_reuses_condition_noise_and_zero_reference_is_identity(self):
        calls = []

        def fake_forward_unet(
            latents,
            timestep,
            prompt_embeds,
            cond_latents,
            reference_tokens=None,
            condition_noise=None,
        ):
            calls.append((condition_noise, reference_tokens))
            return latents + condition_noise.mean() * 0.0

        model = SimpleNamespace(
            reference_spatial_gating=True,
            reference_spatial_gate_scale=1.0,
            reference_token_scale=0.0,
            reference_global_scale=0.0,
            reference_debug_dump=False,
            global_rank=0,
            _reference_spatial_train_debug_printed=False,
            forward_unet=fake_forward_unet,
            _capture_reference_prediction_metrics=lambda *args: None,
        )
        latents = torch.randn(1, 4, 12, 8)
        cond_latents = torch.randn(1, 4, 4, 4)
        result = MVDiffusion.forward_unet_spatial_gated(
            model,
            latents,
            torch.tensor([1]),
            torch.randn(1, 77, 8),
            cond_latents,
            reference_tokens=torch.randn(1, 7, 8),
            ref_slot_weights=torch.ones(1, 1, 6),
        )

        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0][0], calls[1][0])
        self.assertIsNone(calls[0][1])
        self.assertIsNone(calls[1][1])
        self.assertTrue(torch.allclose(result, latents))

    def test_validation_reference_status_is_false_for_only_padded_refs(self):
        model = SimpleNamespace(enable_reference_adapter=True, reference_adapter=object())
        batch = {
            "ref_imgs": torch.zeros(1, 2, 3, 8, 8),
            "ref_valid_mask": torch.zeros(1, 2),
        }
        kwargs, uses_reference = MVDiffusion._build_validation_reference_kwargs(model, batch, 0)
        self.assertFalse(uses_reference)
        self.assertEqual(kwargs, {})

    def test_validation_reference_kwargs_include_references_and_slot_weights(self):
        model = SimpleNamespace(
            enable_reference_adapter=True,
            reference_adapter=object(),
            reference_token_scale=0.1,
            reference_global_token_scale=0.1,
            reference_global_token_enabled=True,
            reference_match_scale=1.0,
            reference_near_scale=0.35,
            reference_nonmatch_scale=0.05,
            reference_unknown_scale=0.1,
            reference_spatial_gating=True,
            reference_spatial_gate_scale=1.0,
            reference_debug_dump=False,
        )
        batch = {
            "ref_imgs": torch.ones(1, 2, 3, 8, 8),
            "ref_valid_mask": torch.tensor([[1.0, 0.0]]),
            "ref_view_labels": torch.tensor([[1, 6]]),
            "ref_slot_weights": torch.tensor([[
                [0.05, 1.0, 0.35, 0.05, 0.35, 0.05],
                [1.0, 0.05, 0.05, 0.05, 0.05, 1.0],
            ]]),
        }

        kwargs, uses_reference = MVDiffusion._build_validation_reference_kwargs(model, batch, 0)

        self.assertTrue(uses_reference)
        self.assertEqual(len(kwargs["reference_images"]), 1)
        self.assertEqual(kwargs["reference_view_ids"], [1])
        self.assertEqual(kwargs["reference_valid_mask"], [1.0])
        self.assertTrue(torch.allclose(
            torch.tensor(kwargs["reference_slot_weights"]),
            torch.tensor([[0.05, 1.0, 0.35, 0.05, 0.35, 0.05]]),
        ))
        self.assertTrue(kwargs["reference_spatial_gating"])

    def test_unfreeze_cross_attention_excludes_attn1(self):
        module = nn.Module()
        module.attn1 = nn.Module()
        module.attn1.to_k = nn.Linear(2, 2, bias=False)
        module.attn1.to_v = nn.Linear(2, 2, bias=False)
        module.attn2 = nn.Module()
        module.attn2.to_k = nn.Linear(2, 2, bias=False)
        module.attn2.to_v = nn.Linear(2, 2, bias=False)
        for parameter in module.parameters():
            parameter.requires_grad = False

        count = maybe_unfreeze_cross_attention(module, limit=4)

        self.assertEqual(count, 2)
        self.assertFalse(module.attn1.to_k.weight.requires_grad)
        self.assertFalse(module.attn1.to_v.weight.requires_grad)
        self.assertTrue(module.attn2.to_k.weight.requires_grad)
        self.assertTrue(module.attn2.to_v.weight.requires_grad)

    def test_controlnet_uses_separate_base_and_ref_conditioning(self):
        class FakeControlNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def set_attn_processor(self, processor):
                pass

            def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
                self.calls.append(encoder_hidden_states.clone())
                value = encoder_hidden_states.mean()
                return [torch.full_like(sample, value)], torch.full_like(sample, value)

        class FakeWrappedUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.received = None

            def forward(self, sample, timestep, **kwargs):
                self.received = kwargs
                return (sample,)

        wrapped = FakeWrappedUNet()
        controlnet = FakeControlNet()
        module = DepthControlUNet(wrapped, controlnet=controlnet)
        ref_states = torch.ones(1, 4, 2)
        base_states = torch.zeros(1, 3, 2)
        sample = torch.zeros(1, 4, 2, 2)

        module(
            sample,
            torch.tensor([1]),
            ref_states,
            cross_attention_kwargs={
                "control_depth": torch.zeros_like(sample),
                "reference_spatial_mask": torch.ones(1, 1, 2, 2),
                "reference_base_encoder_hidden_states": base_states,
            },
        )

        self.assertEqual(len(controlnet.calls), 2)
        self.assertTrue(torch.equal(controlnet.calls[0], ref_states))
        self.assertTrue(torch.equal(controlnet.calls[1], base_states))
        self.assertEqual(float(wrapped.received["mid_block_res_sample"].mean()), 1.0)
        self.assertEqual(float(wrapped.received["base_mid_block_res_sample"].mean()), 0.0)

    def test_periodic_reference_metrics_are_written_as_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleNamespace(
                global_rank=0,
                logdir=tmpdir,
                reference_debug_metrics_filename="reference_debug_metrics.jsonl",
            )
            MVDiffusion._write_reference_debug_metrics(model, {
                "step": 20,
                "reference_delta_mean": 0.125,
                "ref_proj_grad_norm": 0.5,
            })
            rows = [
                json.loads(line)
                for line in (Path(tmpdir) / "reference_debug_metrics.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows[0]["step"], 20)
            self.assertEqual(rows[0]["reference_delta_mean"], 0.125)

    def test_tensor_metrics_capture_deltas_and_adapter_grad_norms(self):
        model = SimpleNamespace(
            _should_collect_reference_metrics=lambda: True,
            _reference_forward_debug_metrics=None,
        )
        base = torch.zeros(1, 2, 2, 2)
        ref = torch.ones_like(base)
        final = torch.full_like(base, 0.25)
        MVDiffusion._capture_reference_prediction_metrics(model, base, ref, final)
        self.assertEqual(model._reference_forward_debug_metrics["reference_delta_mean"], 1.0)
        self.assertEqual(model._reference_forward_debug_metrics["final_delta_mean"], 0.25)

        adapter = ReferenceAdapter(embed_dim=4)
        output = adapter(
            torch.randn(1, 1, 4),
            torch.zeros(1, 1, dtype=torch.long),
        )
        output.sum().backward()
        self.assertGreater(MVDiffusion._module_grad_norm(adapter, "ref_proj"), 0.0)
        self.assertGreater(MVDiffusion._module_grad_norm(adapter, "view_embed"), 0.0)

    def test_post_train_smoke_writes_ablation_outputs_and_report(self):
        class FakePipeline:
            def __init__(self):
                self.unet = nn.Identity()
                self.device = None

            def to(self, device):
                self.device = torch.device(device)
                self.unet.to(device)
                return self

            def __call__(self, image, **kwargs):
                refs = kwargs.get("reference_images") or []
                slot_weights = kwargs.get("reference_slot_weights") or []
                value = 0
                for index, ref in enumerate(refs):
                    value += (index + 1) * int(torch.tensor(list(ref.convert("RGB").getdata())[0]).sum().item())
                value += int(10 * sum(sum(weights) for weights in slot_weights))
                value = value % 256
                return SimpleNamespace(images=[Image.new("RGB", (8, 12), (value, value, value))])

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            refs = root / "refs"
            refs.mkdir()
            Image.new("RGB", (8, 8), (10, 0, 0)).save(refs / "ref_a.png")
            Image.new("RGB", (8, 8), (0, 0, 200)).save(refs / "ref_b.png")
            cond_path = root / "cond.png"
            Image.new("RGB", (8, 8), "white").save(cond_path)
            metadata_path = root / "metadata.json"
            metadata_path.write_text(json.dumps({
                "ref_a.png": {"azimuth": 30, "elevation": 20},
                "ref_b.png": {"azimuth": 210, "elevation": -10},
            }), encoding="utf-8")
            adapter = ReferenceAdapter(embed_dim=8)
            checkpoint_path = root / "adapter_last.pt"
            torch.save({
                "state_dict": {
                    f"reference_adapter.{name}": tensor.detach().clone()
                    for name, tensor in adapter.state_dict().items()
                }
            }, checkpoint_path)
            pipeline = FakePipeline()
            model = SimpleNamespace(
                logdir=str(root / "logs"),
                reference_adapter=adapter,
                reference_smoke_test_cond_image=str(cond_path),
                reference_smoke_test_reference_dir=str(refs),
                reference_smoke_test_metadata=str(metadata_path),
                reference_smoke_test_num_inference_steps=2,
                reference_smoke_test_seed=1,
                reference_smoke_test_difference_threshold=1e-6,
                reference_token_scale=0.1,
                reference_global_scale=0.05,
                reference_match_scale=1.0,
                reference_near_scale=0.35,
                reference_nonmatch_scale=0.05,
                reference_unknown_scale=0.1,
                reference_spatial_gating=True,
                reference_spatial_gate_scale=1.0,
                device=torch.device("cpu"),
                pipeline=pipeline,
            )

            report = run_post_train_reference_smoke_test(model, str(checkpoint_path))

            smoke_dir = Path(model.logdir) / "reference_smoke_test"
            self.assertTrue((smoke_dir / "comparison_grid.png").exists())
            self.assertTrue((smoke_dir / "difference_report.json").exists())
            self.assertTrue((smoke_dir / "slot_weight_report.json").exists())
            self.assertTrue((smoke_dir / "per_slot_difference_report.json").exists())
            self.assertTrue((smoke_dir / "per_slot_difference_grid.png").exists())
            self.assertTrue((smoke_dir / "correct_vs_no_refs_diff_heatmap.png").exists())
            self.assertTrue((smoke_dir / "wrong_vs_correct_diff_heatmap.png").exists())
            self.assertTrue((smoke_dir / "object_summary.json").exists())
            per_slot = json.loads((smoke_dir / "per_slot_difference_report.json").read_text(encoding="utf-8"))
            self.assertEqual(len(per_slot["correct_vs_no_refs"]), 6)
            self.assertIn(report["verdict"], {"PASS", "WARNING", "FAIL"})
            self.assertGreater(report["correct_vs_no_refs_mean_abs_diff"], 0.0)
            self.assertEqual(pipeline.device, next(adapter.parameters()).device)

    def test_manual_pose_metadata_produces_pose_aware_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            Image.new("RGB", (8, 8), "red").save(root / "side.jpg")
            metadata_path = root / "ref_metadata.json"
            metadata_path.write_text(json.dumps({
                "side.jpg": {"azimuth": 90, "elevation": -10},
            }), encoding="utf-8")

            references = load_reference_images(
                str(root / "side.jpg"),
                metadata_path=str(metadata_path),
                pose_version="v1.2",
            )
            weights = references_to_slot_weights(references, pose_version="v1.2")[0]

            self.assertEqual(references[0].pose_source, "metadata")
            self.assertEqual(references[0].azimuth, 90)
            self.assertEqual(references[0].elevation, -10)
            self.assertEqual(max(range(6), key=lambda slot: weights[slot]), 1)
            self.assertGreater(max(weights) - min(weights), 0.5)

    def test_wide_routing_spreads_210_degree_reference_to_neighbors(self):
        weights = routed_pose_to_slot_weights(
            210,
            -10,
            mode="wide",
            sigma_deg=80,
            min_weight=0.05,
            normalize=True,
            elevation_weight=0.25,
        )

        self.assertEqual(max(range(6), key=lambda slot: weights[slot]), 3)
        self.assertGreater(weights[2], 0.3)
        self.assertGreater(weights[3], 0.9)
        self.assertGreater(weights[4], 0.3)
        self.assertLess(weights[0], weights[2])

    def test_wide_routing_is_broader_than_local_routing(self):
        local = routed_pose_to_slot_weights(210, -10, mode="local")
        wide = routed_pose_to_slot_weights(210, -10, mode="wide", sigma_deg=80)

        self.assertGreater(sum(weight > 0.3 for weight in wide), sum(weight > 0.3 for weight in local))

    def test_wide_routing_handles_circular_wraparound(self):
        weights = routed_pose_to_slot_weights(330, -10, mode="wide", sigma_deg=80)

        self.assertGreater(weights[0], 0.3)
        self.assertEqual(max(range(6), key=lambda slot: weights[slot]), 5)

    def test_cross_view_propagation_increases_neighbor_slots(self):
        base = [0.05, 0.05, 0.05, 1.0, 0.05, 0.05]
        propagated = propagate_slot_weights(base, strength=0.3, neighbor_degrees=120)

        self.assertGreater(propagated[2], base[2])
        self.assertGreater(propagated[4], base[4])
        self.assertGreater(propagated[3], propagated[2])

    def test_adapter_global_token_uses_only_valid_references(self):
        torch.manual_seed(0)
        adapter = ReferenceAdapter(embed_dim=8)
        valid_embed = torch.randn(1, 1, 8)
        padded_a = torch.zeros(1, 1, 8)
        padded_b = torch.full((1, 1, 8), -1000.0)
        view_ids = torch.tensor([[0, 6]])
        valid_mask = torch.tensor([[1.0, 0.0]])

        output_a = adapter(
            torch.cat([valid_embed, padded_a], dim=1),
            view_ids,
            ref_valid_mask=valid_mask,
            token_scale=0.0,
            global_scale=0.1,
        )[:, 0]
        output_b = adapter(
            torch.cat([valid_embed, padded_b], dim=1),
            view_ids,
            ref_valid_mask=valid_mask,
            token_scale=0.0,
            global_scale=0.1,
        )[:, 0]

        self.assertTrue(torch.allclose(output_a, output_b, atol=1e-6))

    def test_valid_mask_zeroes_spatial_weights_in_wide_mode(self):
        refs = [
            SimpleNamespace(azimuth=210, elevation=-10, view_label="view_210"),
            SimpleNamespace(azimuth=30, elevation=20, view_label="view_30"),
        ]
        weights = torch.tensor(references_to_slot_weights(refs, mode="wide"))
        valid_mask = torch.tensor([1.0, 0.0])
        masked = weights * valid_mask.unsqueeze(-1)

        self.assertGreater(float(masked[0].sum()), 0.0)
        self.assertEqual(float(masked[1].sum()), 0.0)

    def test_filename_pose_parsing(self):
        self.assertEqual(infer_pose_from_filename("ref_090_el-10.jpg"), (90.0, -10.0))
        self.assertEqual(infer_pose_from_filename("ref_180_el20.jpg"), (180.0, 20.0))
        self.assertEqual(infer_pose_from_filename("ref_240_el-10.jpg"), (240.0, -10.0))


if __name__ == "__main__":
    unittest.main()
