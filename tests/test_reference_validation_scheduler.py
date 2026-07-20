import unittest

import torch
from diffusers.schedulers import DDPMScheduler, EulerAncestralDiscreteScheduler

from zero123plus.pipeline import RefOnlyNoisedUNet


class ReferenceValidationSchedulerTest(unittest.TestCase):
    def test_eval_condition_noise_uses_train_scheduler_for_non_inference_timestep(self):
        train_sched = DDPMScheduler(num_train_timesteps=1000)
        val_sched = EulerAncestralDiscreteScheduler(num_train_timesteps=1000)
        val_sched.set_timesteps(15, device="cpu")

        wrapper = RefOnlyNoisedUNet.__new__(RefOnlyNoisedUNet)
        torch.nn.Module.__init__(wrapper)
        wrapper.train_sched = train_sched
        wrapper.val_sched = val_sched
        wrapper._reference_val_scheduler_debug_printed = True
        wrapper.eval()

        timestep = torch.tensor([123], dtype=torch.long)
        self.assertFalse(
            bool(RefOnlyNoisedUNet._scheduler_timestep_membership(timestep, val_sched).all().item())
        )

        cond_lat = torch.randn(1, 4, 8, 8)
        noise = torch.randn_like(cond_lat)

        actual = wrapper._noise_condition_latents(cond_lat, noise, timestep)
        expected = train_sched.scale_model_input(
            train_sched.add_noise(cond_lat, noise, timestep),
            timestep,
        )

        self.assertTrue(torch.allclose(actual, expected))

    def test_eval_condition_noise_uses_train_scheduler_for_long_validation_timestep(self):
        train_sched = DDPMScheduler(num_train_timesteps=1000)
        val_sched = EulerAncestralDiscreteScheduler(num_train_timesteps=1000)
        val_sched.set_timesteps(1000, device="cpu")

        wrapper = RefOnlyNoisedUNet.__new__(RefOnlyNoisedUNet)
        torch.nn.Module.__init__(wrapper)
        wrapper.train_sched = train_sched
        wrapper.val_sched = val_sched
        wrapper._reference_val_scheduler_debug_printed = True
        wrapper.eval()

        timestep = torch.tensor([123], dtype=torch.long)
        self.assertTrue(
            bool(RefOnlyNoisedUNet._scheduler_timestep_membership(timestep, val_sched).all().item())
        )

        scheduler, reason, _ = wrapper._condition_noise_scheduler(timestep)

        self.assertIs(scheduler, train_sched)
        self.assertEqual(reason, "validation_supervised_noise")


if __name__ == "__main__":
    unittest.main()
