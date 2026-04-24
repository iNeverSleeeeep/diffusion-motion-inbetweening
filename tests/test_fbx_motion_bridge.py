import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from utils.fbx_motion_bridge import (
    build_blender_background_command,
    build_observed_motion_from_poses,
    build_two_keyframe_feature_mask_from_joint_mask,
    build_two_keyframe_joint_mask,
    interpolate_pose_sequence,
    pack_generated_motion,
)


class FbxMotionBridgeTests(unittest.TestCase):
    @staticmethod
    def _load_example_poses():
        example = np.load("dataset/000021.npy")[:, :22].astype(np.float32)
        return example[0], example[1]

    def test_build_two_keyframe_joint_mask_marks_only_boundaries(self):
        lengths = torch.tensor([6, 4], dtype=torch.long)
        mask = build_two_keyframe_joint_mask(lengths=lengths, n_joints=22, n_features=1, n_frames=6)

        self.assertEqual(mask.shape, (2, 22, 1, 6))
        self.assertTrue(mask[0, :, :, 0].all())
        self.assertTrue(mask[0, :, :, 5].all())
        self.assertFalse(mask[0, :, :, 1:5].any())

        self.assertTrue(mask[1, :, :, 0].all())
        self.assertTrue(mask[1, :, :, 3].all())
        self.assertFalse(mask[1, :, :, 1:3].any())
        self.assertFalse(mask[1, :, :, 4:].any())

    def test_feature_mask_matches_root_and_joint_position_layout(self):
        lengths = torch.tensor([5], dtype=torch.long)
        joint_mask = build_two_keyframe_joint_mask(lengths=lengths, n_joints=22, n_features=1, n_frames=5)
        feature_mask = build_two_keyframe_feature_mask_from_joint_mask(joint_mask, feature_mode="pos")

        self.assertEqual(feature_mask.shape, (1, 263, 1, 5))
        self.assertTrue(feature_mask[0, 1:4, 0, 0].all())
        self.assertTrue(feature_mask[0, 1:4, 0, 4].all())
        self.assertFalse(feature_mask[0, :, 0, 1:4].any())

    def test_interpolate_pose_sequence_keeps_endpoints(self):
        start_pose = np.zeros((22, 3), dtype=np.float32)
        end_pose = np.ones((22, 3), dtype=np.float32) * 10

        seq = interpolate_pose_sequence(start_pose, end_pose, num_frames=6)

        self.assertEqual(seq.shape, (7, 22, 3))
        np.testing.assert_allclose(seq[0], start_pose)
        np.testing.assert_allclose(seq[-1], end_pose)
        np.testing.assert_allclose(seq[3], np.ones((22, 3), dtype=np.float32) * 5)

    def test_build_observed_motion_from_poses_returns_expected_shapes(self):
        start_pose, end_pose = self._load_example_poses()

        observed_motion, feature_mask, joint_mask, positions = build_observed_motion_from_poses(
            start_pose=start_pose,
            end_pose=end_pose,
            num_output_frames=5,
            feature_mode="pos",
        )

        self.assertEqual(observed_motion.shape, (1, 263, 1, 5))
        self.assertEqual(feature_mask.shape, (1, 263, 1, 5))
        self.assertEqual(joint_mask.shape, (1, 22, 1, 5))
        self.assertEqual(positions.shape, (6, 22, 3))
        self.assertTrue(joint_mask[0, :, :, 0].all())
        self.assertTrue(joint_mask[0, :, :, 4].all())
        self.assertFalse(joint_mask[0, :, :, 1:4].any())

    def test_pack_generated_motion_writes_npz_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "generated_motion.npz"
            motion = np.zeros((5, 22, 3), dtype=np.float32)

            pack_generated_motion(
                output_path=output_path,
                joint_positions=motion,
                fps=20,
                source_fbx_a="a.fbx",
                source_fbx_b="b.fbx",
                source_frame_a=1,
                source_frame_b=2,
            )

            with np.load(output_path) as payload:
                self.assertEqual(payload["joint_positions"].shape, (5, 22, 3))
                self.assertEqual(int(payload["fps"]), 20)
                self.assertEqual(str(payload["source_fbx_a"]), "a.fbx")

    def test_build_blender_background_command_contains_script_and_args(self):
        cmd = build_blender_background_command(
            blender_path="C:/Blender/blender.exe",
            script_path="tools/blender/extract_fbx_keyframe.py",
            script_args={"--fbx": "a.fbx", "--frame": 12, "--output": "pose_a.npz"},
        )

        self.assertEqual(
            cmd[:4],
            [
                "C:/Blender/blender.exe",
                "--background",
                "--python",
                "tools/blender/extract_fbx_keyframe.py",
            ],
        )
        self.assertIn("--", cmd)
        joined = " ".join(cmd)
        self.assertIn("--fbx a.fbx", joined)
        self.assertIn("--frame 12", joined)


if __name__ == "__main__":
    unittest.main()
