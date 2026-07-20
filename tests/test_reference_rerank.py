import unittest

from PIL import Image

from src.utils.reference_rerank import generate_zero123plus_candidate, split_zero123plus_sheet


class FakePipelineOutput:
    def __init__(self, image):
        self.images = [image]


class FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return FakePipelineOutput(Image.new("RGB", (640, 960), "white"))


class ReferenceRerankTest(unittest.TestCase):
    def test_split_zero123plus_sheet_returns_six_tiles(self):
        sheet = Image.new("RGB", (640, 960), "white")
        tiles = split_zero123plus_sheet(sheet)

        self.assertEqual(len(tiles), 6)
        self.assertEqual(tiles[0].size, (320, 320))

    def test_candidate_generation_does_not_pass_references_to_zero123plus(self):
        pipeline = FakePipeline()
        input_image = Image.new("RGB", (320, 320), "white")

        generate_zero123plus_candidate(
            pipeline,
            input_image,
            num_inference_steps=4,
            device="cpu",
            seed=0,
        )

        args, kwargs = pipeline.calls[0]
        self.assertEqual(args[0], input_image)
        self.assertIn("num_inference_steps", kwargs)
        self.assertIn("generator", kwargs)
        self.assertNotIn("retrieved_images", kwargs)
        self.assertNotIn("reference_images", kwargs)
        self.assertNotIn("reference_mode", kwargs)
        self.assertNotIn("reference_images", kwargs)


if __name__ == "__main__":
    unittest.main()
