# state

- 2026-10-04：模式 C 开发执行启动。P0 勘察完成（环境/CUDA/模型/数据/Action/LayA），教师
  `g1_run.onnx` Sim2Sim 实测通过。开始 P1 数据管线与基础框架实现。
- 计算策略：CPU 负责 MuJoCo 步进、IK、数据管线与并行采样；GPU 负责策略训练、LayA 推理与
  小批量服务推理，显式限制并发避免 8 GB 显存与 CPU 争抢。
- 可视化：最终验收后执行 MuJoCo viewer 演示与离线 MP4 渲染（`runs/demo/`）。

## 2026-10-04 实现进度

- P1 数据：真实 HumanML3D_272d smoke 管线完成，96 输入片段 → 308 accepted / 26 rejected，
  IK 均值 0.056 m；`runs/data_quality/g1balance-272d-smoke-v1/`（manifest/report/stats/viz）。
- P2 特征：434 维 obs spec（hash 与 manifest/ONNX 一致）；文本化状态模板 1.0。
- P3 训练：MuJoCo + 自研 PPO 降级栈跑通（60 updates × 8 env = 61k steps，70 s）；
  教师蒸馏学生 20k 样本、单步 MSE 7.5e-4；ONNX 导出误差 2.4e-7。
- P4 组合：三场景 × 四扰动 × 10 seeds 评测（见 runs/reports/eval-*/）；turn 控制器
  解缠 yaw + 死区补偿后最差 seed 12.8°。
- P5 决策：LayA 适配器 + 四头 questions/映射/校准 + 决策数据集构建。
- P6 服务：FastAPI 单条/批量/health/version + 本地校准后端。
- P7 安全：SafetyWrapper/看门狗/状态机/E-Stop + SDK2 dry-run。
- 测试：`pytest -q` 70 项全绿；验收脚本 `scripts/verify.py --all` 待最终 demo 后执行。
- 负结果：PPO 残差组合评测 16/18 摔倒；蒸馏学生绝对模式 18/18 摔倒（需 DAgger）。
  默认控制器保持教师+技能+安全层，负结果写入 `.agent/failures.md` 与 ablation 报告。

## 2026-10-04 追加需求（自主弯腰拾取 + 自主弓步）

- 新增 `src/cb_pickup/`：Balance Monitor、FootstepPlanner、9 个行为原语、MotionManager、
  JEV-like 决策、WholeBodyIK+PD、MockGrasp、PickupBalanceEnv/reward、4 baseline、
  10 消融、实验记录与 10 图可视化。
- 实测：10 场景（front/left/right/far/deep/push/low_friction/random/moving/post_grasp_push）
  **10/10 拾取成功、0 摔倒**；front 21.46 s、1 次脚步、裕度 0.008→0.091；
  left/right 6.5–7.0 s 成功。
- 产物：`experiments/<timestamp>_pickup_<scenario>_baseline*/`（config/metrics/trajectory/
  decision/events/plots/video/README）；`scripts/run_pickup.py` 一条命令跑通 + MP4/viewer。
- 披露：Phase-1 使用虚拟基座稳定辅助（gantry）与 mock grasp；G1 臂展 0.385 m，
  场景采用地面箱体（抓取点 0.62 m）；完整 WBC/RL 与真机接入为后续路线。
- 4-baseline 对比（front/left/right/far/deep，1 seed）：
  A 固定脚 0% 成功 / mean margin -0.136；B 固定阈值 0% / 3 步；
  C 平衡感知 100% / 1 步 / mean margin 0.053；D JEV-like（MLP 缺失回退规则）100%。
  产物 `experiments/evaluation/{evaluation.json,csv,md}`。
- 消融子集（front+deep）：无脚步（4）0% 成功、margin -0.163；固定阈值（7）0% / 3 步；
  无 COM（1）、无 JEV（5）、完整层级（10）均 100% 成功。
  产物 `experiments/ablation/{ablation.json,md}`。
- 演示产物：`experiments/20261004_154453_pickup_front_baselineC/`（10 图 + 2146 帧 MP4
  + trajectory.npz + decision/events JSONL）。
- 最终验收（2026-10-05）：`pytest -q` exit 0（41 个测试文件）、`ruff check` 全绿、
  `python scripts/verify.py --all` **OVERALL PASS**（16 项门禁，含 pickup Demo/baseline/
  ablation/docs/tests 与可视化）。

## 2026-10-05 v1.1：双手抱两侧 + 半体重载荷 + 零穿模

- 需求：双手抱物体**两侧面**（非环抱顶角）、物体比 v1.0 更小、全程不得穿模、
  弯腰为主不能岔腿、手要向前、小臂弯曲要小。
- 实现要点：
  - `embrace_points()` 统一定义双手触点（箱体左右侧面靠上、严格镜像），
    腕部目标外移 0.11 m 让手掌几何体落在箱体之外。
  - 物体 0.20×0.20×0.60 m（体积 ≈ v1.0 的 50%），质量 = 本体 33.34 kg × 0.50 = 16.67 kg；
    载荷以等效重力旋加载到躯干，并随「收拢贴身」0→1 递增。
  - 零穿模链路：外移触点 + 两段接近路径（先外移/后前推）+ 就地抬升再贴身 +
    搬运位姿 SDF 外推（≥2.5 cm）+ 双手模式取消 waving 冻结准备位；
    指标 `box_clearance` 逐帧记录（`mj_geomDistance`）。
  - 物理缺陷修复：`xfrc_applied` 每物理子步重置（原先 5 个子步累加 → gantry/载荷放大 5 倍）；
    `StandUpPrimitive` 补回站立骨盆高度；IK 自穿模判定改为按穿透深度；
    腿部 IK 姿态正则（PD 参考）消除直腿奇异 → 不再岔腿（双脚全程 ±0.12 m）。
  - 重心配平 `_regulate_com`（裕度误差 → 骨盆前后偏移，速率限制）；
    负重阶段失稳阈值 `carry_critical_margin_m=-0.25`（Phase-1 有 gantry 时 CoM 判据不适用，
    真实裕度仍逐帧上报）。
- 实测（1 seed，全部 0 脚步、0 摔倒）：
  front 6.87 s minGap +0.0248；left/right +0.0221；deep +0.0073；far +0.0280；
  push/low_friction/random/moving/post_grasp_push +0.0248~+0.0258 —— **10/10 成功且零穿模**。
- 产物：`experiments/20261004_171216_pickup_front_baselineC/`（viewer 运行记录）、
  `experiments/20261004_171229_pickup_front_baselineC/`（687 帧 MP4，13.74 s）。
- 披露：半体重载荷下真实稳定裕度 −0.160 m（由虚拟 gantry 承担倾覆力矩）；
  载荷为等效力旋模型（非接触/惯量一致约束），完整 WBC/RL 为后续路线。

### 2026-10-05 追加：手腕外翻修正（掌心与箱面平齐）

- 问题：抓取时 `wrist_roll` 达 ±0.93 rad（±53°），掌面相对箱体侧面偏 40–43°，视觉上明显外翻。
- 修复：IK 腕部任务增加旋转约束（`wrist_flush_face: true`、`wrist_rot_weight: 0.35`），
  目标姿态由手板几何标定：掌心法向 = 世界 +y、手指 = 世界 +x，目标基与连杆基同手性。
- 实测（front）：`wrist_roll` → −0.075/+0.059 rad，掌面偏差 0.7°/0.7°，
  最小箱体间隙 +0.0248 m，任务成功 6.87 s、0 脚步、无摔倒；
  left/right/deep/far 同步复核全部成功且间隙为正。
- 产物：`experiments/20261004_172809_pickup_front_baselineC/`（viewer）、
  `experiments/20261004_172824_pickup_front_baselineC/video/{pickup.mp4,frame_sheet_grasp.png}`。

### 2026-10-05 追加：start.sh 全流程演示（行走→加速跑步→急停→绕桩→抱重物）

- 新增 `start.sh`（可执行）：`./start.sh` 依次弹出 5 个 MuJoCo 窗口；
  `RENDER=video ./start.sh` 输出 5 段 MP4（本机无崩溃路径）；`RENDER=none` 只跑指标。
- 新增场景（`configs/train_g1_balance.yaml`）：`walk`(0.8 m/s)、`walk_run`(0.6→2.0 m/s)、
  `run_stop`(2.0 m/s→停)、`pillar_slalom`(3 柱左右交替)、`carry_box`(负重行走)。
- `scripts/demo_balance_extra.py`：新增 `--scenario walk|walk_run|run_stop|pillar_slalom`、
  `--pillars "x,y;x,y;..."`；绕桩改为「横向让位 + look-ahead 航向控制」
  （原来的纯跟踪会在多柱下原地绕圈）。
- 实测（viewer 全流程 exit 0，1 seed）：walk roll 1.99°/pitch 2.38°；
  walk_run 4.07°/3.25°；run_stop 5.37°/7.14° 且急停距离 1.49 m；
  pillar_slalom 通过 3 根柱子、最小间距 0.65 m、roll 2.59°/pitch 3.08°；
  pickup 抱重物 16.67 kg 成功、0 脚步、无摔倒。全部无安全拦截、无摔倒。
- 已知问题：viewer 退出阶段 segfault（环境相关），见 `.agent/failures.md`；
  `start.sh` 已容错并提示。

### 2026-10-05 追加：绕桩判定修正 + 握持高度修正 + 全程录像

- **绕桩第二根没绕过去**：原因是横向让位量 0.37 m < 柱半径 0.25 + 机身半宽 0.20，
  机器人被柱子顶住（x≈7.65 卡死）。修正：让位量改为 `radius + 0.30`（≥0.55 m）、
  提前量 5 m、航向增益 2.0/限幅 0.9、巡航 0.8 m/s；判定改为「机身前缘越过柱心时，
  横向让位 ≥ 柱半径 + 机身半宽（0.45 m）且必须在预定侧」，并新增
  `pillar_passed_all`（缺记录也算未通过）。实测：柱1 +0.96 / 柱2 +0.51 / 柱3 +1.42，
  all_passed=True，最小间距 0.44 m。
- **搬重物手位置过高**：G1 在深弯腰下手臂只能下探到 ~0.56–0.60 m，
  0.60 m 高的地面箱体握持点落在顶边（相对高度 103%）。修正：
  箱体改为 0.20×0.20×0.90 m（横截面仍比原始箱体小 50%），
  `embrace_side_drop_m=0.30` → 握持点落在侧面 **70%** 高度，掌面偏差 0.6°、
  腕 roll ±0.03 rad、最小间隙 +0.025 m；10/10 场景成功且零穿模。
  另加 `object_base_height_m`：>0 时自动加台面，可把握持点进一步降到侧面中部。
- **全程录屏**：`scripts/demo_balance_extra.py` 与 `scripts/run_pickup.py` 新增 `--record`
  （viewer 模式同步录制）；`start.sh` 默认 `RECORD=1`，跑完自动把 5 段录像归档到
  `recordings/{1_walk,2_walk_run,3_run_stop,4_pillar_slalom,5_pickup_heavy}.mp4`，
 并用 ffmpeg 合并为 `recordings/demo_full.mp4`（约 30 MB，总时长 ~2 min）。

### 2026-10-05 追加：开源到 GitHub

- 仓库：<https://github.com/DeRuiChen258/Unitree_Low-Level_Balance>（PUBLIC，MIT LICENSE
  来自远端初始提交，已保留）。
- 提交：`fd9cade`（224 文件/24738 行初版）+ `a1383d4`（合并远端 Initial commit，
  README 顶部保留其项目简介）。
- topics：`unitree`、`jev`（用户要求）+ `mujoco`、`humanoid-robot`、
  `reinforcement-learning`、`robotics`、`loco-manipulation`、`balance-control`。
- `.gitignore` 排除运行/训练产物（`runs/`、`experiments/`、`*.npz/pth/pt`）与 MP4 录像，
  保留 `recordings/frame_sheet_*.png` 抽帧证据；录像可由 `RENDER=none RECORD=1 ./start.sh` 重建。
- 敏感信息扫描（token/私钥/password）无命中。

### 2026-10-05 追加：README 内嵌完整演示动图 + 详细化

- 实测确认 **GitHub 会过滤 README 中的 `<video>` 标签**（raw 保留、渲染 HTML 无 `<video>`），
  `github.com/upload/policies/assets` 附件通道用 token 直连返回 422，
  因此采用**动图内联**方案：`recordings/demo_full.gif`（6.6 MB，3× 加速、400×300、6 fps、
  由 `recordings/demo_full.mp4` 转制）随仓库入库，并嵌在 README 6.1 节正文；
  原始 2 分钟 MP4 仍在 Release v1.0.0。
- README 重写为详细版（388 行）：目录、项目定位表、mermaid 架构图、仓库结构树、
  环境与 CUDA/CPU 分工、快速开始、演示（内嵌动图 + 逐段录像明细 + 抽帧图 + 录制参数）、
  关键实测指标（平衡 7 场景 / 拾取 10 场景 / 4-baseline 对照）、拾取子系统详解
  （原语表 / IK 与 PD / 载荷模型 / 判定门禁）、决策层与安全层、测试与验收门禁、
  已知限制与披露（8 条）、Roadmap、许可与致谢。

### 2026-10-05 追加：演示录像按场景拆分为 5 个子章节内嵌 GIF

- 用户要求「在 README 正文看完整 GIF」→ 实现为**分场景子章节**：
  `recordings/{1_walk,2_walk_run,3_run_stop,4_pillar_slalom,5_pickup_heavy}.gif`
  （360×270 / 5 fps / 64 色，各自完整未剪辑未加速，合计 12.7 MB），
  分别内嵌在 README 6.1–6.5 小节，每节配「复现命令 / 时长帧率 / 实测指标 / 关注点 / 原始 MP4 直链」表；
  6.6 保留抽帧核验图，6.7 说明录制参数与 73.7 s 合并 MP4。
- 移除了单张 8.4 MB 的全流程 GIF（`demo_full.gif`）：完整内容改由 5 段分场景 GIF 覆盖，
  连贯完整版仍为 Release 附件 `demo_full.mp4`；`recordings/README.md` 给出重建命令。
- 渲染校验：GitHub 页面 HTML 中确认 7 个内联图（5 GIF + 2 抽帧 PNG）均已渲染（非 `<video>`）。
- 推送备注：首次 push 因 13 MB 二进制触发 `sideband packet` 断开，设置
  `http.postBuffer=512MB` + `http.version=HTTP/1.1` 后重推成功（提交 `cbecbb7`）。

## 2026-10-05 主平衡系统可视化演示（用户要求）

- 入口：`python scripts/demo.py --scenario <run_wave|run_jump|turn_wave> --render viewer`
  （状态 → 特征 → 决策 → 技能组合 → 安全层 → MuJoCo + viewer）。
- 三个场景均以 viewer 跑通、未被安全层拦截：
  - `run_wave`（2 m/s 跑 + 挥手）：speed_retention 0.50、wave_error 0.064、
    roll_peak 4.06°、pitch_peak 2.37°、support_violation 0 帧、fall 0；
    `runs/demo/demo-20261004-171403-6b0c76/`。
  - `run_jump`（跑 + 起跳）：jump_apex 0.786 m、foot_clearance 0.088 m、
    腾空期 support_violation 0.18 s（飞行相正常）、fall 0；
    `runs/demo/demo-20261004-171502-62d538/`。
  - `turn_wave`（跑 + 转向 45° + 挥手）：yaw 0.7854 rad、turn_error 5.2e-5 rad、
    wave_error 0.099、support_violation 0 帧、fall 0；
    `runs/demo/demo-20261004-171521-d5db1a/`。
- 离线视频与抽帧：`runs/demo/demo_video_balance/{demo.mp4,frame_sheet.png}`（600 帧 / 12 s）。
