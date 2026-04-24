import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import bpy
    from mathutils import Vector
except ImportError as exc:  # pragma: no cover - only available inside Blender
    raise SystemExit("This script must run inside Blender.") from exc


CANONICAL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


CHILD_HINTS = {
    "pelvis": "spine1",
    "left_hip": "left_knee",
    "right_hip": "right_knee",
    "spine1": "spine2",
    "left_knee": "left_ankle",
    "right_knee": "right_ankle",
    "spine2": "spine3",
    "left_ankle": "left_foot",
    "right_ankle": "right_foot",
    "spine3": "neck",
    "neck": "head",
    "left_collar": "left_shoulder",
    "right_collar": "right_shoulder",
    "left_shoulder": "left_elbow",
    "right_shoulder": "right_elbow",
    "left_elbow": "left_wrist",
    "right_elbow": "right_wrist",
}


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-fbx", required=True)
    parser.add_argument("--motion-npz", required=True)
    parser.add_argument("--output-fbx", default="")
    parser.add_argument("--output-bvh", default="")
    parser.add_argument("--armature-name", default="")
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_fbx(path: str):
    bpy.ops.import_scene.fbx(filepath=path)


def find_armature(armature_name: str):
    if armature_name:
        obj = bpy.data.objects.get(armature_name)
        if obj is None or obj.type != "ARMATURE":
            raise RuntimeError(f"Armature '{armature_name}' not found.")
        return obj
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("No armature found in reference FBX.")
    return armatures[0]


def build_joint_index():
    return {name: idx for idx, name in enumerate(CANONICAL_JOINT_NAMES)}


def apply_frame(armature, pose_bones, frame_positions, joint_index):
    bpy.context.scene.frame_set(bpy.context.scene.frame_current)
    pelvis = pose_bones.get("pelvis")
    if pelvis is not None:
        pelvis.location = Vector(frame_positions[joint_index["pelvis"]].tolist())
        pelvis.keyframe_insert(data_path="location")

    for joint_name, child_name in CHILD_HINTS.items():
        pose_bone = pose_bones.get(joint_name)
        if pose_bone is None:
            continue
        child_idx = joint_index.get(child_name)
        joint_idx = joint_index.get(joint_name)
        if child_idx is None or joint_idx is None:
            continue

        start = Vector(frame_positions[joint_idx].tolist())
        end = Vector(frame_positions[child_idx].tolist())
        direction = end - start
        if direction.length < 1e-6:
            continue
        quat = direction.to_track_quat("Y", "Z")
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.rotation_quaternion = quat
        pose_bone.keyframe_insert(data_path="rotation_quaternion")


def main():
    args = parse_args()
    clear_scene()
    import_fbx(args.reference_fbx)
    armature = find_armature(args.armature_name)

    payload = np.load(args.motion_npz)
    joint_positions = np.asarray(payload["joint_positions"], dtype=np.float32)
    fps = int(payload["fps"]) if "fps" in payload else args.fps

    scene = bpy.context.scene
    scene.render.fps = fps
    scene.frame_start = 1
    scene.frame_end = int(joint_positions.shape[0])

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")
    pose_bones = armature.pose.bones
    joint_index = build_joint_index()

    for frame_idx, frame_positions in enumerate(joint_positions, start=1):
        scene.frame_set(frame_idx)
        apply_frame(armature, pose_bones, frame_positions, joint_index)

    bpy.ops.object.mode_set(mode="OBJECT")

    if args.output_bvh:
        Path(args.output_bvh).parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.export_anim.bvh(filepath=args.output_bvh, frame_start=scene.frame_start, frame_end=scene.frame_end, root_transform_only=False)

    if args.output_fbx:
        Path(args.output_fbx).parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.export_scene.fbx(filepath=args.output_fbx, bake_anim=True, use_selection=False)


if __name__ == "__main__":
    main()
