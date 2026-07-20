import unittest

import torch
from einops import rearrange


class Zero123TargetLayoutTest(unittest.TestCase):
    def test_six_views_convert_to_original_zero123plus_sheet_layout(self):
        # Slot values encode [30, 90, 150, 210, 270, 330].
        target_imgs = torch.zeros(1, 6, 1, 2, 2)
        for slot in range(6):
            target_imgs[:, slot] = slot

        sheet = rearrange(target_imgs, 'b (x y) c h w -> b c (x h) (y w)', x=3, y=2)

        self.assertEqual(sheet.shape, (1, 1, 6, 4))
        self.assertTrue(torch.all(sheet[0, 0, 0:2, 0:2] == 0))  # 30 deg
        self.assertTrue(torch.all(sheet[0, 0, 0:2, 2:4] == 1))  # 90 deg
        self.assertTrue(torch.all(sheet[0, 0, 2:4, 0:2] == 2))  # 150 deg
        self.assertTrue(torch.all(sheet[0, 0, 2:4, 2:4] == 3))  # 210 deg
        self.assertTrue(torch.all(sheet[0, 0, 4:6, 0:2] == 4))  # 270 deg
        self.assertTrue(torch.all(sheet[0, 0, 4:6, 2:4] == 5))  # 330 deg


if __name__ == "__main__":
    unittest.main()
