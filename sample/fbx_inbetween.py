import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.tensors import lengths_to_mask
from model.cfg_sampler import ClassifierFreeSampleModel
from utils import dist_util
from utils.fixseed import fixseed
from utils.fbx_motion_bridge import (
    apply_dataset_normalization,
    build_blender_background_command,
    build_observed_motion_from_poses,
    load_extracted_pose,
    pack_generated_motion,
)
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.parser_util import CondSyntArgs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--fbx_a", required=True)
    parser.add_argument("--frame_a", required=True, type=int)
    parser.add_argument("--fbx_b", required=True)
    parser.add_argument("--frame_b", required=True, type=int)
    parser.add_argument("--reference_fbx", default="")
    parser.add_argument("--blender_path", required=True)
    parser.add_argument("--armature_name", default="")
    parser.add_argument("--num_output_frames", type=int, default=60)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--feature_mode", choices=["pos", "pos_rot", "pos_rot_vel"], default="pos")
    parser.add_argument("--text_prompt", default="")
    parser.add_argument("--export_bvh", action="store_true")
    parser.add_argument("--export_fbx", action="store_true")
    parser.add_argument("--keep_intermediate", action="store_true")
    return parser.parse_args()


def load_dataset(args):
    conf = DatasetConfig(
        name="humanml",
        batch_size=1,
        num_frames=196,
        split="test",
        hml_mode="text_only",
        use_abs3d=True,
        traject_only=False,
        use_random_projection=False,
        augment_type="none",
    )
    return get_dataset_loader(conf, shuffle=False, num_workers=0, drop_last=False)


def run_blender_script(blender_path: str, script_path: str, script_args: dict):
    command = build_blender_background_command(blender_path, script_path, script_args)
    subprocess.run(command, check=True)


def ensure_output_dirs(output_dir: Path):
    extracted_dir = output_dir / "extracted"
    intermediate_dir = output_dir / "intermediate"
    preview_dir = output_dir / "preview"
    exports_dir = output_dir / "exports"
    for path in [extracted_dir, intermediate_dir, preview_dir, exports_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return extracted_dir, intermediate_dir, preview_dir, exports_dir


def create_model_runtime(cli_args):
    model_args = CondSyntArgs()
    args_path = Path(cli_args.model_path).resolve().parent / "args.json"
    with open(args_path, "r", encoding="utf-8") as fr:
        saved_args = json.load(fr)
    for key, value in saved_args.items():
        if hasattr(model_args, key):
            setattr(model_args, key, value)
    model_args.device = cli_args.device
    model_args.seed = cli_args.seed
    model_args.batch_size = 1
    model_args.num_samples = 1
    model_args.num_repetitions = 1
    model_args.dataset = "humanml"
    model_args.abs_3d = True
    model_args.keyframe_conditioned = True
    model_args.guidance_param = 0.0 if not cli_args.text_prompt else model_args.guidance_param
    model_args.editable_features = cli_args.feature_mode
    return model_args


def sample_motion(model_args, dataset_loader, observed_motion, observed_mask, lengths, text_prompt):
    dist_util.setup_dist(model_args.device)
    model, diffusion = create_model_and_diffusion(model_args, dataset_loader)
    load_saved_model(model, model_args.model_path)
    if model_args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()

    model_kwargs = {
        "y": {
            "mask": lengths_to_mask(lengths, observed_motion.shape[-1]).unsqueeze(1).unsqueeze(1).to(dist_util.dev()),
            "lengths": lengths.to(dist_util.dev()),
            "text": [text_prompt],
        },
        "obs_x0": observed_motion.to(dist_util.dev()),
        "obs_mask": observed_mask.to(dist_util.dev()),
    }

    if model_args.keyframe_guidance_param != 1:
        model_kwargs["y"]["keyframe_scale"] = torch.ones(1, device=dist_util.dev()) * model_args.keyframe_guidance_param

    sample = diffusion.p_sample_loop(
        model,
        (1, model.njoints, model.nfeats, observed_motion.shape[-1]),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )
    return sample, model_kwargs, model


def decode_sample(sample, dataset_loader, model_args):
    n_joints = 22 if sample.shape[1] in [263, 264] else 21
    sample = dataset_loader.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
    sample = recover_from_ric(sample, n_joints, abs_3d=model_args.abs_3d)
    sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
    return sample.cpu().numpy()


def save_results(output_dir: Path, generated_motion: np.ndarray, lengths: np.ndarray, text_prompt: str):
    np.save(
        output_dir / "results.npy",
        {
            "motion": generated_motion[None, ...],
            "text": np.asarray([[text_prompt]]),
            "lengths": lengths[None, ...],
            "num_samples": 1,
            "num_repetitions": 1,
        },
    )


def render_preview(preview_dir: Path, generated_motion: np.ndarray, text_prompt: str, fps: int):
    skeleton = paramUtil.t2m_kinematic_chain
    motion = generated_motion[0].transpose(2, 0, 1)
    plot_3d_motion(
        str(preview_dir / "sample00_rep00.mp4"),
        skeleton,
        motion,
        dataset="humanml",
        title=text_prompt or "fbx_inbetween",
        fps=fps,
    )


def main():
    cli_args = parse_args()
    fixseed(cli_args.seed)
    output_dir = Path(cli_args.output_dir or Path("save/results") / f"fbx_inbetween_seed{cli_args.seed}")
    extracted_dir, intermediate_dir, preview_dir, exports_dir = ensure_output_dirs(output_dir)

    extract_script = str(Path("tools/blender/extract_fbx_keyframe.py"))
    run_blender_script(
        cli_args.blender_path,
        extract_script,
        {
            "--fbx": cli_args.fbx_a,
            "--frame": cli_args.frame_a,
            "--output": extracted_dir / "pose_a.npz",
            "--armature-name": cli_args.armature_name,
        },
    )
    run_blender_script(
        cli_args.blender_path,
        extract_script,
        {
            "--fbx": cli_args.fbx_b,
            "--frame": cli_args.frame_b,
            "--output": extracted_dir / "pose_b.npz",
            "--armature-name": cli_args.armature_name,
        },
    )

    start_pose = load_extracted_pose(extracted_dir / "pose_a.npz")
    end_pose = load_extracted_pose(extracted_dir / "pose_b.npz")

    raw_observed_motion, observed_mask, joint_mask, interpolated_positions = build_observed_motion_from_poses(
        start_pose=start_pose,
        end_pose=end_pose,
        num_output_frames=cli_args.num_output_frames,
        feature_mode=cli_args.feature_mode,
    )

    dataset_loader = load_dataset(cli_args)
    observed_motion = apply_dataset_normalization(raw_observed_motion, dataset_loader.dataset)
    np.savez(
        intermediate_dir / "conditioned_input.npz",
        observed_motion=observed_motion.cpu().numpy(),
        observed_mask=observed_mask.cpu().numpy(),
        joint_mask=joint_mask.cpu().numpy(),
        interpolated_positions=interpolated_positions,
    )

    model_args = create_model_runtime(cli_args)
    sample, model_kwargs, model = sample_motion(
        model_args=model_args,
        dataset_loader=dataset_loader,
        observed_motion=observed_motion,
        observed_mask=observed_mask,
        lengths=torch.tensor([cli_args.num_output_frames], dtype=torch.long),
        text_prompt=cli_args.text_prompt,
    )

    generated_motion = decode_sample(sample, dataset_loader, model_args)
    save_results(output_dir, generated_motion, np.asarray([cli_args.num_output_frames]), cli_args.text_prompt)
    render_preview(preview_dir, generated_motion, cli_args.text_prompt, cli_args.fps)
    pack_generated_motion(
        intermediate_dir / "generated_motion.npz",
        joint_positions=generated_motion[0].transpose(2, 0, 1),
        fps=cli_args.fps,
        source_fbx_a=cli_args.fbx_a,
        source_fbx_b=cli_args.fbx_b,
        source_frame_a=cli_args.frame_a,
        source_frame_b=cli_args.frame_b,
    )

    reference_fbx = cli_args.reference_fbx or cli_args.fbx_a
    if cli_args.export_bvh or cli_args.export_fbx:
        export_script = str(Path("tools/blender/export_motion_to_fbx.py"))
        export_args = {
            "--reference-fbx": reference_fbx,
            "--motion-npz": intermediate_dir / "generated_motion.npz",
            "--armature-name": cli_args.armature_name,
            "--fps": cli_args.fps,
        }
        if cli_args.export_bvh:
            export_args["--output-bvh"] = exports_dir / "inbetween.bvh"
        if cli_args.export_fbx:
            export_args["--output-fbx"] = exports_dir / "inbetween.fbx"
        run_blender_script(cli_args.blender_path, export_script, export_args)

    with open(output_dir / "args.json", "w", encoding="utf-8") as fw:
        json.dump(vars(cli_args), fw, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
