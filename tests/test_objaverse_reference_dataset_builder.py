import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.build_objaverse_reference_dataset import (
    REFERENCE_POSES,
    TARGET_ORDER,
    TARGET_POSES,
    build_dataset,
    load_metadata_records,
    parse_args,
    readable_objaverse_id,
    split_records,
)
from src.data.objaverse_zero123plus import ManifestReferenceZero123PlusData
from zero123plus.reference_utils import load_reference_images, references_to_slot_weights


def make_args(root, metadata_path, **overrides):
    values = dict(
        output_root=str(root),
        source_dir=None,
        metadata_path=str(metadata_path),
        seed_dataset_root=None,
        category_keywords=["plushie", "toy"],
        max_objects=10,
        train_ratio=0.34,
        val_ratio=0.33,
        test_ratio=0.33,
        seed=7,
        image_size=64,
        render_size=64,
        render_white_background=True,
        blender_path="blender",
        postprocess_python="python",
        radius=3.0,
        fov=30.0,
        target_size=1.6,
        min_occupancy=0.03,
        max_occupancy=0.92,
        max_center_offset=0.2,
        preview_count=8,
        skip_render=True,
        allow_empty=False,
        clip_filter_text=None,
        clip_filter_min_score=None,
    )
    values.update(overrides)
    return Namespace(**values)


def write_good_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((16, 12, 48, 52), fill=(220, 80, 120))
    image.save(path, quality=95)


def write_bad_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), "black").save(path)


def write_rendered_object(root, object_id, good=True):
    obj = root / "objects" / object_id
    filenames = ["cond.jpg", *[filename for _, filename, _, _ in TARGET_POSES], *[filename for filename, _, _ in REFERENCE_POSES]]
    for filename in filenames:
        if good:
            write_good_image(obj / filename)
        else:
            write_bad_image(obj / filename)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ObjaverseReferenceDatasetBuilderTest(unittest.TestCase):
    def test_portable_command_defaults(self):
        args = parse_args([])

        self.assertEqual(args.objaverse_cache_dir, "data/objaverse_cache")
        self.assertTrue(Path(args.postprocess_python).name.startswith("python"))

    def test_metadata_parsing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metadata.jsonl"
            path.write_text(
                json.dumps({"object_id": "toy_a", "tags": ["plushie"]}) + "\n"
                + json.dumps({"object_id": "toy_b", "category": "toy"}) + "\n",
                encoding="utf-8",
            )
            rows = load_metadata_records(str(path))

            self.assertEqual([row["object_id"] for row in rows], ["toy_a", "toy_b"])

    def test_objaverse_download_names_are_readable(self):
        object_id = readable_objaverse_id(
            "1234567890abcdef",
            {"name": "Cute Plush Bat Toy"},
            1,
        )

        self.assertTrue(object_id.startswith("cute_plush_bat_toy"))
        self.assertIn("1234567890ab", object_id)

    def test_builder_writes_object_level_splits_and_compatible_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata_path = root / "metadata.json"
            metadata = [
                {"object_id": "toy_a", "name": "plushie bear", "tags": ["plushie"]},
                {"object_id": "toy_b", "name": "toy rabbit", "tags": ["toy"]},
                {"object_id": "toy_c", "name": "stuffed toy", "categories": ["toy"]},
                {"object_id": "bad_d", "name": "toy broken", "tags": ["toy"]},
                {"object_id": "chair_e", "name": "wood chair", "tags": ["furniture"]},
            ]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            for object_id in ["toy_a", "toy_b", "toy_c"]:
                write_rendered_object(root, object_id, good=True)
            write_rendered_object(root, "bad_d", good=False)

            report = build_dataset(make_args(root, metadata_path))

            self.assertEqual(report["accepted_object_count"], 3)
            self.assertGreaterEqual(report["rejected_object_count"], 2)
            self.assertTrue((root / "split_report.json").exists())
            self.assertTrue((root / "dataset_preview_grid.png").exists())
            self.assertTrue((root / "external_test" / "README.md").exists())

            train = read_jsonl(root / "train.jsonl")
            val = read_jsonl(root / "val.jsonl")
            test = read_jsonl(root / "test_objaverse_heldout.jsonl")
            split_ids = [set(record["object_id"] for record in split) for split in (train, val, test)]
            self.assertEqual(sum(len(ids) for ids in split_ids), 3)
            self.assertFalse(split_ids[0] & split_ids[1])
            self.assertFalse(split_ids[0] & split_ids[2])
            self.assertFalse(split_ids[1] & split_ids[2])
            self.assertNotIn("bad_d", set().union(*split_ids))
            self.assertNotIn("chair_e", set().union(*split_ids))

            all_records = train + val + test
            for record in all_records:
                self.assertEqual(list(record["target_imgs"].keys()), TARGET_ORDER)
                self.assertEqual(record["target_azimuths"], [30, 90, 150, 210, 270, 330])
                self.assertEqual(record["target_elevations"], [20, -10, 20, -10, 20, -10])
                self.assertEqual(len(record["ref_imgs"]), 3)
                self.assertEqual(len(record["ref_azimuths"]), 3)
                self.assertTrue((root / "objects" / record["object_id"] / "ref_metadata.json").exists())

            dataset = ManifestReferenceZero123PlusData(
                root_dir=str(root),
                manifest_fname="train.jsonl",
                validation=False,
                validation_reserve=0,
                num_refs=3,
            )
            sample = dataset[0]
            self.assertEqual(sample["target_imgs"].shape, (6, 3, 64, 64))
            self.assertEqual(sample["ref_valid_mask"].tolist(), [1.0, 1.0, 1.0])

    def test_seed_dataset_is_copied_into_objects_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed_root = root / "seed"
            seed_obj = seed_root / "object_000001"
            seed_obj.mkdir(parents=True)
            for filename in [
                "cond.png",
                "target_030.png",
                "target_090.png",
                "target_150.png",
                "target_210.png",
                "target_270.png",
                "target_330.png",
                "ref_090.png",
                "ref_180.png",
                "ref_240.png",
            ]:
                write_good_image(seed_obj / filename)
            (seed_obj / "meta.json").write_text(
                json.dumps({
                    "object_id": "object_000001",
                    "source_model": str(root / "source_models" / "toy.glb"),
                    "validation_passed": True,
                }),
                encoding="utf-8",
            )
            seed_record = {
                "object_id": "object_000001",
                "cond_img": "object_000001/cond.png",
                "target_azimuths": [30, 90, 150, 210, 270, 330],
                "target_elevations": [20, -10, 20, -10, 20, -10],
                "target_imgs": {
                    "30": "object_000001/target_030.png",
                    "90": "object_000001/target_090.png",
                    "150": "object_000001/target_150.png",
                    "210": "object_000001/target_210.png",
                    "270": "object_000001/target_270.png",
                    "330": "object_000001/target_330.png",
                },
                "ref_imgs": [
                    "object_000001/ref_090.png",
                    "object_000001/ref_180.png",
                    "object_000001/ref_240.png",
                ],
                "ref_view_labels": ["unknown", "unknown", "unknown"],
                "ref_azimuths": [90, 180, 240],
                "ref_elevations": [-10, 0, -10],
            }
            (seed_root / "train.jsonl").write_text(json.dumps(seed_record) + "\n", encoding="utf-8")
            metadata_path = root / "metadata.json"
            metadata_path.write_text("[]", encoding="utf-8")

            report = build_dataset(make_args(
                root / "out",
                metadata_path,
                seed_dataset_root=str(seed_root),
                category_keywords=[],
                max_objects=1,
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            ))

            records = read_jsonl(root / "out" / "train.jsonl")
            self.assertEqual(report["seed_object_count"], 1)
            self.assertEqual(records[0]["object_id"], "seed_object_000001")
            self.assertTrue((root / "out" / "objects" / "seed_object_000001" / "cond.png").exists())
            self.assertEqual(records[0]["target_imgs"]["30"], "objects/seed_object_000001/target_030.png")

    def test_reference_metadata_takes_priority_over_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            refs = root / "refs"
            refs.mkdir()
            image_path = refs / "ref_240_el-10.jpg"
            write_good_image(image_path)
            metadata_path = refs / "ref_metadata.json"
            metadata_path.write_text(
                json.dumps({"ref_240_el-10.jpg": {"azimuth": 90, "elevation": -10}}),
                encoding="utf-8",
            )

            references = load_reference_images(str(refs), metadata_path=str(metadata_path), pose_version="v1.2")
            weights = references_to_slot_weights(references, pose_version="v1.2")

            self.assertEqual(references[0].pose_source, "metadata")
            self.assertEqual(references[0].azimuth, 90)
            self.assertEqual(max(range(6), key=lambda slot: weights[0][slot]), 1)

    def test_config_points_to_objaverse_dataset_root(self):
        config_path = Path("configs/zero123plus-reference-adapter-objaverse-wide-1000.yaml")
        text = config_path.read_text(encoding="utf-8")

        self.assertIn("root_dir: data/reference_zero123plus_objaverse_toys", text)
        self.assertIn("manifest_fname: train.jsonl", text)
        self.assertIn("manifest_fname: val.jsonl", text)
        self.assertIn('reference_slot_weight_mode: "wide"', text)

    def test_500_object_split_has_no_train_val_overlap(self):
        records = [{"object_id": f"toy_{index:03d}"} for index in range(500)]
        splits = split_records(records, train_ratio=0.9, val_ratio=0.1, test_ratio=0.0, seed=1)
        train_ids = {record["object_id"] for record in splits["train"]}
        val_ids = {record["object_id"] for record in splits["val"]}

        self.assertEqual(len(train_ids), 450)
        self.assertEqual(len(val_ids), 50)
        self.assertFalse(train_ids & val_ids)

    def test_medium_500_config_settings(self):
        from omegaconf import OmegaConf

        config = OmegaConf.load("configs/zero123plus-reference-adapter-objaverse-wide-500obj-500steps-val.yaml")

        self.assertEqual(config.data.params.train.params.root_dir, "data/reference_zero123plus_objaverse_toys_500")
        self.assertEqual(config.data.params.validation.params.root_dir, "data/reference_zero123plus_objaverse_toys_500")
        self.assertEqual(config.data.params.train.params.manifest_fname, "train.jsonl")
        self.assertEqual(config.data.params.validation.params.manifest_fname, "val.jsonl")
        self.assertFalse(bool(config.data.params.train.params.validation))
        self.assertTrue(bool(config.data.params.validation.params.validation))
        self.assertEqual(config.data.params.train.params.max_objects, 450)
        self.assertEqual(config.data.params.validation.params.max_objects, 50)
        self.assertEqual(config.lightning.trainer.max_steps, 500)
        self.assertEqual(config.lightning.trainer.limit_val_batches, 4)
        self.assertEqual(config.lightning.trainer.val_check_interval, 100)
        self.assertEqual(config.lightning.trainer.accelerator, "gpu")
        self.assertEqual(config.lightning.trainer.devices, 1)
        self.assertEqual(list(config.lightning.adapter_only_checkpoint.save_steps), [100, 250, 500])
        self.assertTrue(bool(config.lightning.adapter_only_checkpoint.save_last))
        self.assertEqual(config.model.params.reference_validation_num_inference_steps, 15)
        self.assertEqual(config.model.params.reference_debug_metrics_interval, 25)
        self.assertEqual(config.model.params.reference_slot_weight_mode, "wide")

    def test_medium_500_config_2500_steps_with_light_validation_logging(self):
        from omegaconf import OmegaConf

        config = OmegaConf.load("configs/zero123plus-reference-adapter-objaverse-wide-500obj-2500steps-val-earlystop.yaml")

        self.assertEqual(config.data.params.train.params.root_dir, "data/reference_zero123plus_objaverse_toys_500")
        self.assertEqual(config.data.params.validation.params.root_dir, "data/reference_zero123plus_objaverse_toys_500")
        self.assertEqual(config.data.params.train.params.max_objects, 450)
        self.assertEqual(config.data.params.validation.params.max_objects, 50)
        self.assertEqual(config.lightning.trainer.max_steps, 2500)
        self.assertEqual(config.lightning.trainer.limit_val_batches, 4)
        self.assertEqual(config.lightning.trainer.val_check_interval, 250)
        self.assertEqual(list(config.lightning.adapter_only_checkpoint.save_steps), [892, 1338, 1784, 2230, 2500])
        self.assertTrue(bool(config.lightning.adapter_only_checkpoint.save_last))
        self.assertEqual(dict(config.lightning.callbacks), {})
        self.assertEqual(config.model.params.reference_validation_num_inference_steps, 15)
        self.assertEqual(config.model.params.reference_debug_metrics_interval, 25)

    def test_existing_configs_do_not_point_to_medium_dataset(self):
        for path in [
            Path("configs/zero123plus-reference-adapter-100.yaml"),
            Path("configs/zero123plus-reference-adapter-tiny.yaml"),
            Path("configs/zero123plus-reference-adapter-objaverse-wide-1000.yaml"),
        ]:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("data/reference_zero123plus_objaverse_toys_500", text)
            self.assertNotIn("zero123plus-reference-adapter-objaverse-wide-500obj-500steps-val", text)


if __name__ == "__main__":
    unittest.main()
