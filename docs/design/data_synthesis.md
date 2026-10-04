# 数据合成与增强（第 4 节）

## 1. 事实来源

- `HumanML3D_272d`：26846 个 `(T,272)` 片段 @30 FPS，train/val/test=23384/1337/4041；
  272 维布局用官方 `Li-xingXiao/272-dim-Motion-Representation` 处理脚本核对：
  `0:2` root xz 速度、`2:8` heading 增量 6D、`8:74` 局部关节位置 22×3、`74:140` 局部速度、
  `140:272` 局部 6D 旋转。
- `HumanML3D`（263d 目录）本机仅解出 1 个样本；`KIT-ML` 为 `.rar`；`AMASS` 为空。
  三者默认关闭，启用而缺文件会显式报错（禁止静默跳过导致统计口径不一致）。
- 只读解压目录，`motion_data.zip` 不参与统计；manifest 记录每个产物 sha256。

## 2. 8 步管线

| 步骤 | 模块 | 输出 | 实测 |
| --- | --- | --- | --- |
| S1 ingest | `cb_data/ingest.py` | MotionClip + 文本 | 96 片段（smoke profile） |
| S2 canonicalize | `canonicalize.py` | 根对齐/30FPS/相位 | 固定 seed 可复现 |
| S3 retarget | `retarget.py` | G1 29DOF qpos + IK 误差 | 实测 IK 均值 ~0.05 m |
| S4 segment | `segment.py` | SkillSegment | 308 accepted（96 片段输入） |
| S5 augment | `synthesize*.py` | 镜像/时间/幅度/相位/根噪声/过渡 | profile hash 入 manifest |
| S6 physics | `physics_filter.py` | accepted/repaired/rejected | 26 rejected，原因分布见报告 |
| S7 split | `split.py` | 按 source 聚簇 | 同源不跨 split（断言） |
| S8 manifest | `manifest.py` | manifest.json/report.md/stats.csv | `runs/data_quality/<v>/` |

## 3. 规范动作空间（ANCSH 方法论迁移）

根轨迹（朝向系）+ 相位时钟 + 幅度缩放 + 结构分解（root/left_leg/right_leg/torso/
left_arm/right_arm/head）。结构分组同时用于组合调度的组仲裁。

## 4. 重定向

主算法是对真实 G1 MJCF 的阻尼最小二乘 IK：SMPL-22 位置 → G1 body keypoints
（pelvis/hip/knee/ankle/shoulder/elbow/wrist），逐帧记录 IK 误差与限位触碰；
`ik_reject_m=0.16` 以上整段拒绝。该实现与 GMR 的完整 SMPL-X 管线不同，已在 TASK/decisions 登记。

## 5. 领域 gap（强制声明）

- `1901.02970v2.pdf`（NOCS）是物体类别级 6D 位姿与 context-aware 混合现实数据生成；
- `1912.11913v2.pdf`（ANCSH）是铰接物体类别级位姿与规范空间层级；
- 二者均属物体位姿领域，本项目只迁移「合成覆盖 + 真实校准」「规范空间 + 结构分解」
  方法论，**不声称它们直接支持人形动作增强**。

## 6. 增强参数（configs/data.yaml）

镜像、时间 0.85–1.15、幅度 0.9–1.1、相位 ±0.05、根 xy 噪声 1–3 cm、yaw ±2°、
观测噪声、0–2 控制步延迟、过渡窗口 0.3–0.6 s。所有参数可开关、固定 seed 复现，
并写入 `aug_profile`。
