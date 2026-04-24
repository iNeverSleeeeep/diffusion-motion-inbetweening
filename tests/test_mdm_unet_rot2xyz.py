import unittest

import torch

from model.mdm_unet import MDM_UNET


class MDMUnetRot2xyzTest(unittest.TestCase):
    def _make_model(self):
        return MDM_UNET(
            modeltype="",
            njoints=263,
            nfeats=1,
            num_actions=1,
            translation=True,
            pose_rep="xyz",
            glob=True,
            glob_rot=None,
            latent_dim=32,
            dim_mults=(1, 1),
            attention=False,
            data_rep="hml_vec",
            dataset="humanml",
            cond_mode="no_cond",
            arch="unet",
        )

    def test_to_allows_lazy_rot2xyz_smpl_init(self):
        model = self._make_model()
        model.to(torch.device("cpu"))
        self.assertIsNone(model.rot2xyz.smpl_model)

    def test_train_allows_lazy_rot2xyz_smpl_init(self):
        model = self._make_model()
        model.train()
        self.assertIsNone(model.rot2xyz.smpl_model)


if __name__ == "__main__":
    unittest.main()
