import unittest

import torch

from model.mdm_dit import MDM_DiT


class MDMDitRot2xyzTest(unittest.TestCase):
    def _make_model(self):
        return MDM_DiT(
            modeltype="",
            njoints=263,
            nfeats=1,
            num_actions=1,
            translation=True,
            pose_rep="xyz",
            glob=True,
            glob_rot=None,
            latent_dim=32,
            ff_size=64,
            num_layers=1,
            num_heads=4,
            dropout=0.1,
            activation="gelu",
            data_rep="hml_vec",
            dataset="humanml",
            cond_mode="no_cond",
            arch="dit_prenorm",
        )

    def test_to_allows_lazy_rot2xyz_smpl_init(self):
        model = self._make_model()
        model.to(torch.device("cpu"))
        self.assertIsNone(model.rot2xyz.smpl_model)

    def test_train_allows_lazy_rot2xyz_smpl_init(self):
        model = self._make_model()
        model.train()
        self.assertIsNone(model.rot2xyz.smpl_model)

    def test_exposes_keyframe_training_compat_flags(self):
        model = self._make_model()
        self.assertFalse(model.keyframe_conditioned)
        self.assertFalse(model.zero_keyframe_loss)
        self.assertEqual(model.keyframe_selection_scheme, "random_frames")


if __name__ == "__main__":
    unittest.main()
