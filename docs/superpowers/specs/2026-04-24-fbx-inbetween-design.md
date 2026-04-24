# FBX 双关键帧补间设计文档

**目标**

在当前仓库基础上增加一条端到端流程，使用户可以输入两段 FBX 动画、分别指定一个关键帧，并生成两者之间的补间动作；最终同时输出仓库原生结果 `results.npy`、预览视频 `mp4`、以及可回到 DCC 软件的 `bvh` 和 `fbx`。

**背景**

当前仓库已经具备“基于关键帧条件的动作生成”能力，但缺少一层外部资产接入能力，无法直接读取 FBX、提取指定姿态、转换成模型可消费的表示，再把生成结果导回原骨架动画。

本设计面向的第一版场景有以下约束：

- 两个输入 FBX 使用同一个角色
- 两者骨架层级和骨骼命名一致
- 允许使用 Blender 作为 FBX 读写和回写工具
- 第一版优先解决“同骨架稳定可用”，不做通用自动重定向

## 仓库现有可复用能力

仓库内部中段能力已经比较完整，主要可复用以下模块：

- [sample/conditional_synthesis.py](E:/Projects/diffusion-motion-inbetweening/sample/conditional_synthesis.py:1)：加载条件模型并执行关键帧条件采样
- [sample/edit.py](E:/Projects/diffusion-motion-inbetweening/sample/edit.py:1)：展示了基于观测动作和 mask 的编辑式采样流程
- [utils/editing_util.py](E:/Projects/diffusion-motion-inbetweening/utils/editing_util.py:56)：生成关键帧 mask
- [data_loaders/humanml/scripts/motion_process.py](E:/Projects/diffusion-motion-inbetweening/data_loaders/humanml/scripts/motion_process.py:190)：在关节轨迹与 HumanML 风格向量表示之间转换

因此新增工作应尽量放在“外部资产桥接层”，而不是重写现有模型采样逻辑。

## 推荐整体架构

新增一条独立但复用现有模型的流程：

`FBX A/B -> Blender 抽取指定帧姿态 -> 标准化 22 关节表示 -> 显式首尾关键帧条件 -> CondMDI 采样 -> 生成完整关节动画 -> Blender 回写为 BVH/FBX`

这样可以把模型相关逻辑保持在仓库原有风格内，同时把 FBX 相关复杂度隔离在 Blender 辅助脚本里。

## 组件拆分

### 1. Blender 姿态提取脚本

新增脚本：

- `tools/blender/extract_fbx_keyframe.py`

职责：

- 在 Blender 后台模式下导入单个 FBX
- 定位 Armature
- 切换到指定帧
- 导出骨架元数据和该帧骨骼变换
- 生成固定目标关节集合的 3D 关节位置
- 保存 `.npz` 中间结果供 Python 主流程消费

输入：

- `--fbx`
- `--frame`
- `--output`
- 可选 `--armature-name`

输出字段建议包括：

- `joint_names`
- `parent_indices`
- `world_positions`
- `local_matrices`
- `world_matrices`
- `frame_index`
- `source_fbx`

### 2. FBX / HumanML 桥接模块

新增模块：

- `utils/fbx_motion_bridge.py`

职责：

- 把 Blender 提取的骨架关节映射到仓库使用的 22 关节标准布局
- 进行 Blender 坐标系到仓库坐标系的转换
- 执行尺度、root 位置、朝向等归一化
- 构造“只观察首尾两个关键帧”的输入 motion 张量
- 把模型输出的关节动画重新打包成 Blender 可回写的数据格式

该文件是整个方案的核心兼容层，应尽量写成不依赖 Blender 运行时的纯 Python 逻辑，方便单元测试。

### 3. 双关键帧采样入口

新增脚本：

- `sample/fbx_inbetween.py`

职责：

- 调用 Blender 提取 `fbx_a` 与 `fbx_b` 的指定帧
- 根据用户输入构造目标输出帧数
- 组装只在首帧和末帧有观测值的 motion 输入
- 构造显式关键帧 mask
- 加载 CondMDI 模型和 diffusion
- 执行条件采样
- 保存 `results.npy`
- 生成骨架预览 `mp4`
- 调 Blender 导出 `bvh` 与 `fbx`

该脚本应尽量复用 [sample/conditional_synthesis.py](E:/Projects/diffusion-motion-inbetweening/sample/conditional_synthesis.py:1) 中的模型加载和采样结构，避免复制粘贴过多内部逻辑。

### 4. Blender 动画回写脚本

新增脚本：

- `tools/blender/export_motion_to_fbx.py`

职责：

- 导入一个参考 FBX 以恢复原始骨架
- 读取生成的完整关节动画数据
- 将每帧姿态应用到原骨架
- bake 成动画 Action
- 导出 `.bvh`
- 导出 `.fbx`

第一版只针对“同角色、同骨架”场景，不做自动 retarget。

## 数据格式设计

为了便于排查问题，流程中间建议保留两类中间文件。

### A. 提取后的关键帧数据

建议格式：`.npz`

字段：

- `joint_names`: 字符串数组
- `parent_indices`: `int` 数组
- `world_positions`: `float32 [J, 3]`
- `local_matrices`: `float32 [J, 4, 4]`
- `world_matrices`: `float32 [J, 4, 4]`
- `frame_index`: `int`
- `source_fbx`: 字符串

这一层主要服务 Blender 提取和调试。

### B. 生成后的完整动作数据

建议格式：`.npz`

字段：

- `joint_positions`: `float32 [T, 22, 3]`
- `root_positions`: `float32 [T, 3]`
- `fps`: `int`
- `source_fbx_a`: 字符串
- `source_fbx_b`: 字符串
- `source_frame_a`: `int`
- `source_frame_b`: `int`
- `num_frames`: `int`

这一层是 `sample/fbx_inbetween.py` 与 Blender 回写脚本之间的契约。

## 条件构造策略

现有仓库里的 mask 生成更偏向 benchmark sparse / clip 等模式。对于本功能，最干净的方案是增加一种“显式首尾关键帧条件”。

建议新增一个 helper，位置可选：

- `utils/fbx_motion_bridge.py`
- 或 `utils/editing_util.py`

该 helper 生成的 mask 规则为：

- 第 `0` 帧完全观测
- 第 `T-1` 帧完全观测
- 其余所有帧不观测

输入 motion 张量的构造规则：

- 第 `0` 帧填入起始关键帧姿态
- 第 `T-1` 帧填入结束关键帧姿态
- 中间帧填零或中性占位值

第一版优先使用 `pos` 条件，必要时再扩展到 `pos_rot_vel`。这样可以降低从 FBX 恢复旋转特征带来的不稳定性。

## 坐标系与骨架约定

这是第一版实现的最高风险点，必须显式约定。

第一版假设：

- 两个输入 FBX 使用同一套 Armature
- 关节命名稳定可映射
- 能够稳定映射到仓库内部 22 关节表示
- 生成结果最终回写到同一参考骨架上

必须完成的归一化步骤：

1. Blender 坐标系转换到仓库坐标系
2. 选定 canonical root joint
3. 将输入姿态重心或 root 的 XZ 平移归一到起点
4. 必要时统一初始朝向
5. 保存可逆的变换参数，供导出时还原

桥接模块必须保存这些逆变换信息，否则无法稳定回写到原始资产空间。

## CLI 设计

主命令示例：

```powershell
python -m sample.fbx_inbetween `
  --model_path save/condmdi_randomframes/model000750000.pt `
  --fbx_a path/to/start_anim.fbx `
  --frame_a 12 `
  --fbx_b path/to/end_anim.fbx `
  --frame_b 48 `
  --num_output_frames 60 `
  --reference_fbx path/to/start_anim.fbx `
  --output_dir save/results/fbx_inbetween_demo `
  --blender_path "C:\Program Files\Blender Foundation\Blender 4.0\blender.exe"
```

建议参数：

- `--fbx_a`
- `--frame_a`
- `--fbx_b`
- `--frame_b`
- `--num_output_frames`
- `--fps`
- `--reference_fbx`
- `--blender_path`
- `--output_dir`
- 现有条件采样参数中可复用的模型参数
- `--export_bvh`
- `--export_fbx`
- `--keep_intermediate`

## 输出目录结构

建议输出为：

```text
<output_dir>/
  args.json
  extracted/
    pose_a.npz
    pose_b.npz
  intermediate/
    conditioned_input.npz
    generated_motion.npz
  results.npy
  preview/
    sample00_rep00.mp4
  exports/
    inbetween.bvh
    inbetween.fbx
```

这样每一层失败时都能快速定位问题。

## 错误处理

预期错误场景及处理方式：

1. Blender 可执行文件不存在
   - 在流程开始前直接失败，并打印当前使用的绝对路径

2. FBX 导入成功但未找到 Armature
   - 报告导入对象列表，并提示可使用 `--armature-name`

3. 所选帧超出范围
   - 明确返回该动画允许的帧范围

4. 关节映射不完整
   - 列出缺失的骨骼名并在采样前终止

5. 模型输入输出形状不一致
   - 报告预期 shape 与实际 shape，同时保留中间文件

6. 导出 BVH/FBX 失败
   - 保留 `results.npy`、预览视频与 `.npz` 中间文件，并将导出阶段标记为部分失败

## 测试策略

新增逻辑的主要风险在桥接层，因此测试重点也应放在桥接层。

### 单元测试

建议覆盖：

- 双关键帧 mask 构造
- 坐标系转换 helper
- 关节映射校验
- synthetic motion 张量构造
- 逆变换参数保存与恢复

建议位置：

- `tests/test_fbx_motion_bridge.py`

### 集成测试

建议覆盖：

- 用离线 mock 提取 payload 测桥接层，不依赖 Blender
- 验证 `sample/fbx_inbetween.py` 能正确构造模型输入并输出关节结果
- 验证 Blender 导出命令拼装逻辑

如果 CI 环境没有 Blender，则 Blender 端到端验证放到手工测试。

### 手工验证流程

第一轮端到端验证建议按以下顺序：

1. 分别提取两个 FBX 指定帧，检查中间 `.npz`
2. 用骨架可视化确认提取姿态是否正确
3. 对较短时长执行条件采样
4. 检查生成序列第 `0` 帧和第 `T-1` 帧是否接近输入关键帧
5. 导出 BVH/FBX，并在 Blender 中回放检查骨骼方向、尺度和动画连续性

## 分阶段交付

为了降低风险，按阶段推进。

### Phase 1

- 增加 Blender 姿态提取脚本
- 增加桥接模块
- 增加显式双关键帧条件采样
- 输出 `results.npy` 和预览 `mp4`

### Phase 2

- 增加导出 `bvh`
- 增加导出 `fbx`

### Phase 3

- 如果位置条件不足，再增强旋转恢复和动作平滑
- 视效果决定是否支持更丰富的条件方式

## 第一版默认取舍

第一版明确采用以下默认策略：

- 只支持同骨架
- FBX 读写依赖 Blender
- 条件方式为显式首尾关键帧
- 结果同时输出研究资产与 DCC 资产
- 优先保证稳健、可调试，而不是一开始就追求大而全

## 结论

推荐按“仓库融合型”方案推进：

- 新增一层桥接工具围绕现有 CondMDI 采样能力
- 尽量少改已有模型内部逻辑
- 使用 Blender 负责 FBX 输入输出
- 保留分层中间结果，便于定位问题
- 测试重点放在桥接和条件构造模块

这样可以最快落地一个对你当前资产场景可用、并且后续还能扩展到更复杂动画工作流的版本。
