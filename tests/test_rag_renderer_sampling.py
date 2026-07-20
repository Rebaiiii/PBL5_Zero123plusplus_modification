import random
import unittest
from types import SimpleNamespace

from scripts.render_rag_zero123plus_tiny import (
    build_rendered_views,
    ref_filename,
    resolve_model_render_options,
    resolve_yaw_offset,
    sample_reference_azimuths,
    sample_reference_elevations,
    target_pose_items,
)
from zero123plus.rag_utils import get_zero123plus_target_poses, pose_to_slot_weights


def in_ranges(angle, ranges):
    return any(start <= angle <= end for start, end in ranges)


class RagRendererSamplingTest(unittest.TestCase):
    def test_biased_sampling_prefers_side_and_back(self):
        args = SimpleNamespace(
            ref_sampling_mode="biased_side_back",
            ref_num=3,
            ref_front_prob=0.2,
            ref_side_prob=0.4,
            ref_back_prob=0.4,
            ref_min_target_gap=0.0,
        )
        random.seed(7)

        counts = {"front": 0, "side": 0, "back": 0, "other": 0}
        for _ in range(300):
            for angle in sample_reference_azimuths(args):
                if in_ranges(angle, [(330, 359), (0, 30)]):
                    counts["front"] += 1
                elif in_ranges(angle, [(60, 120), (240, 300)]):
                    counts["side"] += 1
                elif in_ranges(angle, [(135, 225)]):
                    counts["back"] += 1
                else:
                    counts["other"] += 1

        total = sum(counts.values())
        side_back_ratio = (counts["side"] + counts["back"]) / total
        self.assertGreater(side_back_ratio, 0.70)
        self.assertLess(counts["front"] / total, 0.30)
        self.assertEqual(counts["other"], 0)

    def test_reference_filenames_use_sampled_azimuth(self):
        self.assertEqual(ref_filename(7), "ref_007.png")
        self.assertEqual(ref_filename(359), "ref_359.png")

    def test_v12_target_pose_list(self):
        azimuths, elevations = get_zero123plus_target_poses("v1.2")
        self.assertEqual(azimuths, [30, 90, 150, 210, 270, 330])
        self.assertEqual(elevations, [20, -10, 20, -10, 20, -10])
        self.assertEqual(
            target_pose_items("v1.2"),
            [
                ("target_030.png", 30, 20),
                ("target_090.png", 90, -10),
                ("target_150.png", 150, 20),
                ("target_210.png", 210, -10),
                ("target_270.png", 270, 20),
                ("target_330.png", 330, -10),
            ],
        )

    def test_v11_target_pose_list(self):
        azimuths, elevations = get_zero123plus_target_poses("v1.1")
        self.assertEqual(azimuths, [30, 90, 150, 210, 270, 330])
        self.assertEqual(elevations, [30, -20, 30, -20, 30, -20])

    def test_ref_elevations_match_pose_version_range(self):
        args = SimpleNamespace(zero123plus_pose_version="v1.2", ref_num=20)
        random.seed(9)
        elevations = sample_reference_elevations(args)
        self.assertTrue(all(-10 <= elevation <= 20 for elevation in elevations))

    def test_pose_slot_weights_prefer_close_azimuth_and_elevation(self):
        weights = pose_to_slot_weights(
            92,
            -8,
            target_azimuths=[30, 90, 150, 210, 270, 330],
            target_elevations=[20, -10, 20, -10, 20, -10],
            azimuth_sigma=45,
            elevation_sigma=25,
        )
        self.assertEqual(max(range(6), key=lambda idx: weights[idx]), 1)
        self.assertGreater(weights[1], weights[2])
        self.assertGreaterEqual(min(weights), 0.05)

    def test_rendered_view_metadata_records_azimuth_and_elevation(self):
        views = build_rendered_views(
            target_pose_items("v1.2"),
            ["ref_184.png"],
            [184],
            [-3],
            0,
        )
        self.assertEqual(views["cond.png"], {"azimuth": 0, "elevation": 0})
        self.assertEqual(views["target_030.png"], {"azimuth": 30, "elevation": 20})
        self.assertEqual(views["target_090.png"], {"azimuth": 90, "elevation": -10})
        self.assertEqual(views["ref_184.png"], {"azimuth": 184, "elevation": -3})

    def test_orientation_metadata_resolves_filename_and_object_form(self):
        metadata = {
            "sideways.glb": 90,
            "backwards.glb": {"yaw_degrees": 180},
        }
        self.assertEqual(resolve_yaw_offset("models/sideways.glb", metadata), 90)
        self.assertEqual(resolve_yaw_offset("models/backwards.glb", metadata), 180)
        self.assertEqual(resolve_yaw_offset("models/already_front.glb", metadata), 0)

    def test_orientation_metadata_uses_default_yaw(self):
        self.assertEqual(resolve_yaw_offset("model.glb", {}, default_yaw_offset=270), 270)

    def test_model_render_options_allow_non_orientation_fixes(self):
        metadata = {
            "model.glb": {
                "force_opaque": True,
                "remove_meshes": ["Floor*"],
            }
        }
        self.assertEqual(
            resolve_model_render_options("model.glb", metadata),
            {
                "yaw_degrees": 0.0,
                "force_opaque": True,
                "remove_meshes": ["Floor*"],
            },
        )


if __name__ == "__main__":
    unittest.main()
