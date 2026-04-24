# FBX 双关键帧补间进度记录

## 2026-04-24

### 阶段 0：需求澄清与设计

- 已确认输入为两段 FBX 动画，分别指定一帧作为关键帧
- 已确认输出既要 `results.npy/mp4`，也要 `bvh/fbx`
- 已确认允许 Blender 参与流程
- 已确认当前只需要支持同角色、同骨架、同层级
- 已完成中文设计文档：
  - `docs/superpowers/specs/2026-04-24-fbx-inbetween-design.md`
- 已完成实施计划：
  - `docs/superpowers/plans/2026-04-24-fbx-inbetween.md`

### 阶段 1：桥接层与双关键帧采样入口

- 已新增桥接模块：
  - `utils/fbx_motion_bridge.py`
- 已实现能力：
  - 双关键帧 joint mask 构造
  - 双关键帧 feature mask 构造
  - 起止 pose 插值为中间关节轨迹
  - 关节轨迹转 HumanML 风格特征
  - 中间结果 `generated_motion.npz` 打包
  - Blender 后台命令拼装
- 已新增主入口：
  - `sample/fbx_inbetween.py`
- 当前主入口已串联：
  - Blender 抽帧
  - pose 读取
  - 双关键帧条件构造
  - 条件采样入口
  - `results.npy` / `mp4` / 中间 `.npz` 输出

### 阶段 1：验证记录

- 已通过单元测试：
  - `.\.env_condmdi\Scripts\python.exe -m unittest tests.test_fbx_motion_bridge -v`
- 已通过语法校验：
  - `.\.env_condmdi\Scripts\python.exe -m py_compile utils/fbx_motion_bridge.py tools/blender/extract_fbx_keyframe.py tools/blender/export_motion_to_fbx.py sample/fbx_inbetween.py`
- 当前仍存在的非阻塞问题：
  - `motion_process.py` 内部仍有 `np.float` 旧写法，运行时会出现 DeprecationWarning
  - 尚未对真实 FBX 资产执行端到端手工验证

### 阶段 2：导出侧实现状态

- 已新增 Blender 导出脚本：
  - `tools/blender/export_motion_to_fbx.py`
- 当前实现为同骨架场景的 best-effort 版本：
  - 写入 root 位移
  - 根据父子关节方向近似恢复部分骨骼旋转
  - 支持导出 `bvh` 与 `fbx`
- 当前限制：
  - 还没有在真实角色资产上验证骨骼朝向是否完全一致
  - 若原骨架轴向定义与 HumanML 22 关节差异较大，仍可能需要额外校正

### 今日收尾记录

- 今天先暂停，不继续推进真实资产验证
- 当前代码和文档状态已经适合下次直接接着做端到端验证
- 下次恢复时，优先不要再改架构，先用真实资产跑通并记录问题

### 当前已完成项汇总

- 中文设计文档已完成
- 实施计划已完成
- 阶段进度日志已建立
- 桥接层与双关键帧条件构造已实现
- Blender 单帧提取脚本已实现
- 主入口 `sample/fbx_inbetween.py` 已实现
- Blender 导出脚本已实现 first-pass 版本
- 单元测试与语法校验已通过

### 当前暂停点

- 还未使用真实 FBX 资产执行端到端验证
- 还未确认你的角色骨架命名是否能和 HumanML 22 关节直接匹配
- 还未验证导出的 BVH/FBX 在 Blender 中的朝向、层级和动画连续性

### 下次继续时要做的事

1. 准备两段真实 FBX 动画资产
2. 确认各自要使用的关键帧编号
3. 提供 Blender 可执行文件路径
4. 先单独运行提取脚本，检查 `pose_a.npz` 和 `pose_b.npz`
5. 再运行 `sample/fbx_inbetween.py`，生成 `results.npy`、`mp4`、`generated_motion.npz`
6. 最后验证 `bvh/fbx` 导出效果

### 下次继续时我需要你提供的信息

- `fbx_a` 路径
- `frame_a`
- `fbx_b` 路径
- `frame_b`
- `blender.exe` 路径
- 如果 Armature 名字不是默认可识别的，还需要提供 `armature_name`

### 建议的下次执行顺序

```text
真实资产抽帧验证
-> 关键帧映射检查
-> 生成补间结果
-> 预览视频检查
-> Blender 回写导出检查
-> 根据问题修正关节映射或坐标系
```
