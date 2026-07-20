import unittest

from PIL import Image, ImageDraw

from scripts.postprocess_rag_render import adaptive_foreground_resize, image_report


def make_circle_image(size, bbox):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse(bbox, fill=(220, 80, 120))
    return image


class RAGRenderPostprocessTest(unittest.TestCase):
    def test_adaptive_rescale_enlarges_only_small_foregrounds(self):
        small = make_circle_image(256, (108, 108, 148, 148))
        small_out, small_meta = adaptive_foreground_resize(
            small,
            64,
            min_fill_ratio=0.70,
            target_fill_ratio=0.78,
        )
        small_report = image_report(small_out)

        good = make_circle_image(256, (32, 32, 224, 224))
        good_out, good_meta = adaptive_foreground_resize(
            good,
            64,
            min_fill_ratio=0.70,
            target_fill_ratio=0.78,
        )
        good_report = image_report(good_out)

        self.assertTrue(small_meta["adaptive_rescale_applied"])
        self.assertGreater(small_report["bbox_width_ratio"], 0.65)
        self.assertFalse(good_meta["adaptive_rescale_applied"])
        self.assertGreater(good_report["bbox_width_ratio"], 0.65)


if __name__ == "__main__":
    unittest.main()
