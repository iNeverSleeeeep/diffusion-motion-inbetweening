# FBX Keyframe In-Betweening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为当前仓库增加“输入两段 FBX 动画并各取一帧，生成中间补间动画，同时导出预览与 DCC 资产”的完整流程。

**Architecture:** 保持 CondMDI 现有采样逻辑不变，在其外围增加 Blender 提取脚本、Python 桥接层和导出脚本。第一阶段先打通“姿态提取 -> 双关键帧条件构造 -> 采样预览”，第二阶段补全 BVH/FBX 回写与更完整验证。

**Tech Stack:** Python, PyTorch, NumPy, Blender background Python, pytest

---

## 文件结构

**Create**

- `tools/blender/extract_fbx_keyframe.py`: Blender 后台提取单帧关节数据
- `tools/blender/export_motion_to_fbx.py`: Blender 后台回写生成动画并导出
- `utils/fbx_motion_bridge.py`: FBX 中间数据、关节映射、双关键帧条件构造
- `sample/fbx_inbetween.py`: 主入口，串联提取、采样、导出
- `tests/test_fbx_motion_bridge.py`: 桥接层单元测试

**Modify**

- `utils/editing_util.py`: 如有必要，复用或保持不改；双关键帧 mask 优先放在新桥接模块
- `docs/superpowers/progress/2026-04-24-fbx-inbetween-progress.md`: 按阶段记录进度

### Task 1: 计划与文档落地

**Files:**
- Modify: `docs/superpowers/specs/2026-04-24-fbx-inbetween-design.md`
- Create: `docs/superpowers/plans/2026-04-24-fbx-inbetween.md`
- Create: `docs/superpowers/progress/2026-04-24-fbx-inbetween-progress.md`

- [ ] **Step 1: 将设计文档改为中文 UTF-8 并补进度日志**

Run: `Get-Content docs/superpowers/specs/2026-04-24-fbx-inbetween-design.md -Encoding UTF8`
Expected: 文档内容可按 UTF-8 读取

- [ ] **Step 2: 写入实施计划文档**

Run: `Get-Content docs/superpowers/plans/2026-04-24-fbx-inbetween.md -Encoding UTF8`
Expected: 计划文档存在且可读取

### Task 2: 桥接层与双关键帧条件

**Files:**
- Create: `utils/fbx_motion_bridge.py`
- Test: `tests/test_fbx_motion_bridge.py`

- [ ] **Step 1: 先写失败测试，覆盖双关键帧 mask 与条件张量构造**

Run: `pytest tests/test_fbx_motion_bridge.py -q`
Expected: FAIL，提示桥接模块或函数不存在

- [ ] **Step 2: 实现最小桥接层**

实现内容：
- `build_two_keyframe_joint_mask`
- `build_observed_motion_from_poses`
- `pack_generated_motion`
- 基础校验 helper

- [ ] **Step 3: 重新运行测试直到通过**

Run: `pytest tests/test_fbx_motion_bridge.py -q`
Expected: PASS

- [ ] **Step 4: 记录阶段进度**

在 `docs/superpowers/progress/2026-04-24-fbx-inbetween-progress.md` 追加“桥接层已完成”

### Task 3: Blender 提取脚本

**Files:**
- Create: `tools/blender/extract_fbx_keyframe.py`
- Test: `tests/test_fbx_motion_bridge.py`

- [ ] **Step 1: 先补失败测试，至少覆盖命令参数和输出 payload 打包函数**

Run: `pytest tests/test_fbx_motion_bridge.py -q`
Expected: FAIL，提示缺少 Blender payload 相关函数

- [ ] **Step 2: 实现 Blender 提取脚本**

实现内容：
- 参数解析
- FBX 导入
- Armature 查找
- 指定帧采样
- `.npz` 输出

- [ ] **Step 3: 做语法验证**

Run: `python -m py_compile tools/blender/extract_fbx_keyframe.py utils/fbx_motion_bridge.py`
Expected: 无输出，退出码 0

### Task 4: Phase 1 主入口

**Files:**
- Create: `sample/fbx_inbetween.py`
- Modify: `docs/superpowers/progress/2026-04-24-fbx-inbetween-progress.md`

- [ ] **Step 1: 先写失败测试，至少覆盖参数解析与主流程中不依赖模型的部分**

Run: `pytest tests/test_fbx_motion_bridge.py -q`
Expected: FAIL，提示主入口依赖函数不存在

- [ ] **Step 2: 实现 Phase 1 主入口**

实现内容：
- 调 Blender 提取两个 pose
- 基于提取结果构造双关键帧条件
- 复用现有模型加载与采样逻辑
- 输出 `results.npy` 和中间 `.npz`
- 生成基础预览

- [ ] **Step 3: 做语法验证**

Run: `python -m py_compile sample/fbx_inbetween.py`
Expected: 无输出，退出码 0

- [ ] **Step 4: 运行 Phase 1 测试集合**

Run: `pytest tests/test_fbx_motion_bridge.py -q`
Expected: PASS

- [ ] **Step 5: 记录 Phase 1 进度**

在进度日志中追加 Phase 1 完成情况与未完成项

### Task 5: Phase 2 导出脚本

**Files:**
- Create: `tools/blender/export_motion_to_fbx.py`
- Modify: `sample/fbx_inbetween.py`
- Modify: `docs/superpowers/progress/2026-04-24-fbx-inbetween-progress.md`

- [ ] **Step 1: 先写失败测试，覆盖导出命令拼装与 payload 读取**

Run: `pytest tests/test_fbx_motion_bridge.py -q`
Expected: FAIL，提示导出相关函数不存在

- [ ] **Step 2: 实现导出脚本与主入口对接**

实现内容：
- 读取 `generated_motion.npz`
- 导入参考 FBX
- 应用帧动画
- 导出 `bvh`
- 导出 `fbx`

- [ ] **Step 3: 做语法验证**

Run: `python -m py_compile tools/blender/export_motion_to_fbx.py sample/fbx_inbetween.py`
Expected: 无输出，退出码 0

- [ ] **Step 4: 更新进度日志**

记录 Phase 2 完成情况、导出限制与后续改进方向

### Task 6: 整体验证

**Files:**
- Modify: `docs/superpowers/progress/2026-04-24-fbx-inbetween-progress.md`

- [ ] **Step 1: 运行当前单元测试**

Run: `pytest tests/test_fbx_motion_bridge.py -q`
Expected: PASS

- [ ] **Step 2: 运行语法校验**

Run: `python -m py_compile utils/fbx_motion_bridge.py tools/blender/extract_fbx_keyframe.py tools/blender/export_motion_to_fbx.py sample/fbx_inbetween.py`
Expected: 无输出，退出码 0

- [ ] **Step 3: 汇总当前状态**

在进度日志中记录：
- 已完成阶段
- 已验证项
- 尚未做的真实 FBX 端到端手工验证
