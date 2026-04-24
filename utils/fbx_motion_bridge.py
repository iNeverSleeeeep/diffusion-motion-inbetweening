from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Tuple

import numpy as np
import torch

from data_loaders import humanml_utils
from data_loaders.humanml.scripts.motion_process import process_file
from utils.editing_util import joint_to_full_mask


def validate_pose_array(pose: np.ndarray, expected_joints: int = 22) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape != (expected_joints, 3):
        raise ValueError(f"Expected pose shape {(expected_joints, 3)}, got {pose.shape}")
    return pose


def interpolate_pose_sequence(start_pose: np.ndarray, end_pose: np.ndarray, num_frames: int) -> np.ndarray:
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")
    start_pose = validate_pose_array(start_pose)
    end_pose = validate_pose_array(end_pose)
    alphas = np.linspace(0.0, 1.0, num_frames + 1, dtype=np.float32)
    return ((1.0 - alphas)[:, None, None] * start_pose[None, ...]
            + alphas[:, None, None] * end_pose[None, ...]).astype(np.float32)


def build_two_keyframe_joint_mask(
    lengths: torch.Tensor,
    n_joints: int = 22,
    n_features: int = 1,
    n_frames: int | None = None,
) -> torch.Tensor:
    if n_frames is None:
        n_frames = int(lengths.max().item())
    mask = torch.zeros((lengths.shape[0], n_joints, n_features, n_frames), dtype=torch.bool)
    for idx, length in enumerate(lengths.tolist()):
        if length < 2:
            raise ValueError("Each sequence length must be at least 2")
        mask[idx, :, :, 0] = True
        mask[idx, :, :, length - 1] = True
    return mask


def build_two_keyframe_feature_mask_from_joint_mask(
    joint_mask: torch.Tensor,
    feature_mode: str = "pos",
) -> torch.Tensor:
    return joint_to_full_mask(joint_mask, mode=feature_mode)


def positions_to_hml_features(positions: np.ndarray, feet_threshold: float = 0.002) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[1:] != (22, 3):
        raise ValueError(f"Expected positions shape [T, 22, 3], got {positions.shape}")
    data, _, _, _ = process_file(torch.from_numpy(positions), feet_threshold)
    return np.asarray(data, dtype=np.float32)


def build_observed_motion_from_poses(
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    num_output_frames: int,
    feature_mode: str = "pos",
    feet_threshold: float = 0.002,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    if num_output_frames < 2:
        raise ValueError("num_output_frames must be at least 2")
    interpolated_positions = interpolate_pose_sequence(start_pose, end_pose, num_output_frames)
    hml_features = positions_to_hml_features(interpolated_positions, feet_threshold=feet_threshold)
    if hml_features.shape[0] != num_output_frames:
        raise ValueError(
            f"Expected {num_output_frames} motion frames after conversion, got {hml_features.shape[0]}"
        )

    observed_motion = torch.from_numpy(hml_features.T).float().unsqueeze(0).unsqueeze(2)
    lengths = torch.tensor([num_output_frames], dtype=torch.long)
    joint_mask = build_two_keyframe_joint_mask(lengths=lengths, n_frames=num_output_frames)
    feature_mask = build_two_keyframe_feature_mask_from_joint_mask(joint_mask, feature_mode=feature_mode)
    return observed_motion, feature_mask, joint_mask, interpolated_positions


def apply_dataset_normalization(observed_motion: torch.Tensor, dataset) -> torch.Tensor:
    motion = observed_motion.permute(0, 2, 3, 1)
    motion = dataset.t2m_dataset.transform_th(motion).float()
    return motion.permute(0, 3, 1, 2)


def pack_generated_motion(
    output_path: str | Path,
    joint_positions: np.ndarray,
    fps: int,
    source_fbx_a: str,
    source_fbx_b: str,
    source_frame_a: int,
    source_frame_b: int,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joint_positions = np.asarray(joint_positions, dtype=np.float32)
    root_positions = joint_positions[:, 0, :]
    np.savez(
        output_path,
        joint_positions=joint_positions,
        root_positions=root_positions,
        fps=np.int32(fps),
        source_fbx_a=np.array(source_fbx_a),
        source_fbx_b=np.array(source_fbx_b),
        source_frame_a=np.int32(source_frame_a),
        source_frame_b=np.int32(source_frame_b),
        num_frames=np.int32(joint_positions.shape[0]),
    )


def build_blender_background_command(
    blender_path: str,
    script_path: str,
    script_args: Mapping[str, object],
) -> List[str]:
    command = [blender_path, "--background", "--python", script_path, "--"]
    for key, value in script_args.items():
        command.extend([key, str(value)])
    return command


def load_extracted_pose(npz_path: str | Path) -> np.ndarray:
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=True) as payload:
        if "ordered_world_positions" in payload:
            return validate_pose_array(payload["ordered_world_positions"])

        if "joint_names" not in payload or "world_positions" not in payload:
            raise ValueError(f"{npz_path} 缺少 joint_names/world_positions 或 ordered_world_positions")

        joint_names = [str(name) for name in payload["joint_names"].tolist()]
        world_positions = np.asarray(payload["world_positions"], dtype=np.float32)

    name_to_index = {name.lower(): idx for idx, name in enumerate(joint_names)}
    ordered = []
    missing = []
    for joint_name in humanml_utils.HML_JOINT_NAMES:
        idx = name_to_index.get(joint_name.lower())
        if idx is None:
            missing.append(joint_name)
            continue
        ordered.append(world_positions[idx])

    if missing:
        raise ValueError(f"提取结果缺少标准关节: {missing}")

    return validate_pose_array(np.asarray(ordered, dtype=np.float32))
