import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.data.objaverse_zero123plus import ManifestRAGZero123PlusData


class RagManifestDatasetTest(unittest.TestCase):
    def test_manifest_dataset_returns_rag_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            obj = root / "object_000001"
            obj.mkdir()

            image_names = [
                "cond.png",
                "target_030.png",
                "target_090.png",
                "target_150.png",
                "target_210.png",
                "target_270.png",
                "target_330.png",
                "ref_front.png",
                "ref_side.png",
                "ref_back.png",
            ]
            for name in image_names:
                Image.new("RGBA", (16, 16), (255, 0, 0, 255)).save(obj / name)

            manifest = {
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
                    "object_000001/ref_front.png",
                    "object_000001/ref_side.png",
                    "object_000001/ref_back.png",
                ],
                "ref_view_labels": ["front", "side", "back"],
                "ref_azimuths": [30, 90, 210],
                "ref_elevations": [20, -10, -10],
            }
            with open(root / "train.jsonl", "w", encoding="utf-8") as handle:
                handle.write(json.dumps(manifest) + "\n")

            dataset = ManifestRAGZero123PlusData(
                root_dir=str(root), manifest_fname="train.jsonl", num_refs=4
            )
            sample = dataset[0]

            self.assertEqual(sample["cond_imgs"].shape, (3, 16, 16))
            self.assertEqual(sample["target_imgs"].shape, (6, 3, 16, 16))
            self.assertEqual(sample["ref_imgs"].shape, (4, 3, 16, 16))
            self.assertEqual(sample["ref_view_labels"].tolist(), [0, 1, 2, 6])
            self.assertEqual(sample["ref_slot_weights"].shape, (4, 6))
            self.assertEqual(sample["ref_valid_mask"].tolist(), [1.0, 1.0, 1.0, 0.0])
            self.assertEqual(torch.argmax(sample["ref_slot_weights"], dim=1).tolist(), [0, 1, 3, 0])
            self.assertTrue(torch.all(sample["ref_slot_weights"] >= 0.05))
            with open(root / "train.jsonl", "r", encoding="utf-8") as handle:
                saved = json.loads(handle.readline())
            self.assertIn("target_elevations", saved)
            self.assertIn("ref_elevations", saved)


if __name__ == "__main__":
    unittest.main()
