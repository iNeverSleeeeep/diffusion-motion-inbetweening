import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import bpy
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


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx", required=True)
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--armature-name", default="")
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
        names = [obj.name for obj in bpy.data.objects]
        raise RuntimeError(f"No armature found. Imported objects: {names}")
    if len(armatures) > 1:
        return armatures[0]
    return armatures[0]


def bone_world_matrix(armature_obj, pose_bone):
    return armature_obj.matrix_world @ pose_bone.matrix


def build_parent_indices(bones):
    bone_index = {bone.name: idx for idx, bone in enumerate(bones)}
    parent_indices = []
    for bone in bones:
        if bone.parent is None:
            parent_indices.append(-1)
        else:
            parent_indices.append(bone_index[bone.parent.name])
    return np.asarray(parent_indices, dtype=np.int32)


def canonical_positions(joint_names, world_positions):
    name_to_index = {name.lower(): idx for idx, name in enumerate(joint_names)}
    ordered = []
    for joint_name in CANONICAL_JOINT_NAMES:
        idx = name_to_index.get(joint_name.lower())
        if idx is None:
            return np.empty((0, 3), dtype=np.float32)
        ordered.append(world_positions[idx])
    return np.asarray(ordered, dtype=np.float32)


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clear_scene()
    import_fbx(args.fbx)
    armature = find_armature(args.armature_name)

    scene = bpy.context.scene
    start, end = int(scene.frame_start), int(scene.frame_end)
    if not (start <= args.frame <= end):
        raise RuntimeError(f"Requested frame {args.frame} out of range [{start}, {end}]")
    scene.frame_set(args.frame)
    bpy.context.view_layer.update()

    bones = list(armature.pose.bones)
    joint_names = np.asarray([bone.name for bone in bones])
    parent_indices = build_parent_indices(bones)
    local_matrices = np.asarray([np.array(bone.matrix, dtype=np.float32) for bone in bones], dtype=np.float32)
    world_matrices = np.asarray([np.array(bone_world_matrix(armature, bone), dtype=np.float32) for bone in bones], dtype=np.float32)
    world_positions = np.asarray([bone_world_matrix(armature, bone).translation[:] for bone in bones], dtype=np.float32)
    ordered_world_positions = canonical_positions(joint_names, world_positions)

    np.savez(
        output_path,
        joint_names=joint_names,
        parent_indices=parent_indices,
        local_matrices=local_matrices,
        world_matrices=world_matrices,
        world_positions=world_positions,
        ordered_joint_names=np.asarray(CANONICAL_JOINT_NAMES),
        ordered_world_positions=ordered_world_positions,
        frame_index=np.int32(args.frame),
        source_fbx=np.array(args.fbx),
    )


if __name__ == "__main__":
    main()
