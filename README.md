# G1 小脑平衡 + 多动作自适应组合系统

> 基于 MuJoCo 的 Unitree G1 小脑平衡与多动作自适应系统：动作数据管线与 IK 重定向、教师 ONNX 复用加 PPO 残差、技能原语组合调度、LayA 结构化决策、安全门禁与 SDK2 预留。在此之上实现自主弯腰拾取：双手抱两侧、可搬运约半体重重物、绕柱避障，全程零穿模，并提供一键可视化演示与 MP4 录像

以小脑平衡策略为底座、技能原语为积木、LayA 结构化决策为路由的 Unitree G1 低层行为系统。
完整规格见 `Prompt/小脑平衡多动作自适应系统_完整实现提示词.md`，实现台账见 `TASK.md`。

## 快速开始

```bash
conda activate unitree_rt
pip install -e ".[dev,export,laya]"

# 0 环境自检
python -c "import sys,torch,mujoco; print(sys.version.split()[0], torch.__version__, torch.cuda.is_available(), mujoco.__version__)"

# 1 数据管线（smoke profile 默认，full profile 见 configs/data.yaml）
python scripts/data_build.py --config configs/data.yaml

# 2 训练 + 导出
python scripts/train.py --config configs/train_g1_balance.yaml --seed 1
python scripts/export_onnx.py --checkpoint runs/exp/latest/checkpoints/latest.pt --out runs/exp/latest/policy.onnx

# 3 评测
python scripts/eval.py --policy runs/exp/latest/policy.onnx --scenarios run_jump run_wave turn_wave --seeds 10

# 4 端到端 demo 与验收
python scripts/demo.py --config configs/system.yaml --scenario run_wave --render viewer
python scripts/verify.py --all
pytest -q
```

## 全流程演示（一条命令）

`start.sh` 依次演示：**行走 → 加速跑步 → 急停 → 绕多个柱子 → 停下来抱重物**，
覆盖平衡系统（教师策略 + 技能 + 安全层 + 平衡指标）与拾取系统（双手抱两侧 + 半体重载荷）。

```bash
./start.sh                    # 依次弹出 5 个 MuJoCo 窗口
RENDER=video ./start.sh       # 输出 5 段 MP4（无窗口）
RENDER=none  ./start.sh       # 只跑指标（最快，约 13 s）
RENDER=none RECORD=1 ./start.sh   # 无窗口但仍录制 MP4（最快出片）
STEPS=600 SEED=2 CARRY_MASS=16.67 ./start.sh   # 可调步数/种子/抱持质量
```

演示结束后 5 段录像自动归档到 `recordings/`：
`1_walk.mp4`、`2_walk_run.mp4`、`3_run_stop.mp4`、`4_pillar_slalom.mp4`、`5_pickup_heavy.mp4`，
并用 ffmpeg 合并为 `recordings/demo_full.mp4`；抽帧核验图见
`recordings/frame_sheet_slalom.png` 与 `recordings/frame_sheet_pickup.png`。
- 绕桩判定：机身前缘越过柱心时，横向让位必须 ≥ 柱半径 + 机身半宽（0.45 m）且位于预定侧，
  报告 `pillar_passed_all` 与逐柱明细（缺记录视为未通过）。
- 抱物握持：G1 深弯腰下手臂只能下探到约 0.56–0.60 m，因此箱体取 0.20×0.20×0.90 m，
  握持点落在侧面约 70% 高度（`embrace_side_drop_m=0.30`）；
  需要握到侧面中部时用 `object_base_height_m>0`（自动加台面）。

单场景入口：

```bash
python scripts/demo_balance_extra.py --scenario walk|walk_run|run_stop|pillar_slalom|carry_box \
    --render viewer --pillars "4.5,0.45;8.0,-0.45;11.5,0.45"
```

> 注：本机 Wayland/EGL 下 MuJoCo 交互窗口在**进程退出阶段**可能 segfault（发生在结果落盘之后），
> `start.sh` 已容错；需要无崩溃输出时用 `RENDER=video`。详见 `.agent/failures.md`。

## G1 自主弯腰拾取 + 自主弓步（新增需求 v1.0）

在原有小脑平衡系统之上新增 `src/cb_pickup/`：Balance Monitor（COM/ZMP/CP/支撑域）→
JEV-like 行为决策 → Motion Manager（状态机 + 稳定门禁）→ 9 个行为原语 →
全身 IK + 关节 PD → MuJoCo（500 Hz 物理 / 100 Hz 控制 / 20 Hz 决策 / 10 Hz 脚步）。

```bash
# 一条命令运行完整 Demo（弯腰 → 自主脚步/弓步 → 伸手 → 抓取 → 起身 → 恢复）
python scripts/run_pickup.py --scenario front --baseline C --render video

# 交互可视化（MuJoCo window）
python scripts/run_pickup.py --scenario left --render viewer

# 4 baseline 对比 / 消融 / 总评测（论文式表格）
python scripts/evaluate.py --baselines A B C D
python scripts/run_ablation.py --ablations 1 3 4 5 7 10
```

实测：10 场景全部拾取成功、0 摔倒；产物在 `experiments/<timestamp>_<tag>/`
（config/metrics/trajectory/decision/events/plots/video）。设计文档：
[`docs/design/pickup_balance.md`](docs/design/pickup_balance.md)。

4-baseline 结果（front/left/right/far/deep）：固定站姿 0% 成功 / 稳定裕度 -0.136；
固定阈值脚步 0% / 3 步；平衡感知 100% / 1 步（裕度 0.008→0.053）；JEV-like 100%。
消融：去掉脚步或使用固定阈值均 0% 成功；完整层级与去掉 COM/JEV 变体 100%。

Phase-1 披露：使用虚拟基座稳定辅助与 mock grasp（mocap 附着），
完整 WBC/RL 平衡控制器与真实抓取为后续路线；严格限于 MuJoCo，无真机验证。

## 目录

| 目录 | 作用 |
| --- | --- |
| `src/cb_common` | 共享类型、配置、日志、seed、异常 |
| `src/cb_data` | 272 维动捕 ingest → 规范化 → IK 重定向 → 切片/增强/过滤/切分/manifest |
| `src/cb_features` | 观测 spec、本体/平衡/相位/指令/文本化状态/归一化 |
| `src/cb_train` | MuJoCo 环境、奖励、课程、域随机化、AMP、PPO/蒸馏、ONNX 导出 |
| `src/cb_skills` | stand/walk/run/jump/wave/turn/recover/safe_stop 与组合器 |
| `src/cb_policy` | 教师封装、学生策略、残差补偿、技能调度器 |
| `src/cb_decision` | LayA 适配、questions、映射、校准、数据集、微调、运行时策略 |
| `src/cb_eval` | 指标、批量 rollout、回放、报告 |
| `src/cb_serving` | FastAPI 服务、客户端抽象、vLLM 适配器占位 |
| `src/cb_safety` | 限幅、看门狗、状态机、E-Stop |
| `src/cb_robot` | mock / MuJoCo / SDK2 dry-run 统一接口 |
| `configs` | 所有参数唯一来源（含安全与 LayA 校准） |
| `docs/design` | 9 份设计文档 + sim2real |
| `runs` | 数据质量、训练、评测、决策、demo 产物（不入库） |

## 限制

- 本机无真机：硬件结论止于编译级 / 启动级 / 接口级 / dry-run（Gate 0–2）。
- IsaacLab 未安装：训练栈为 MuJoCo + 自研 PPO，接口对齐 rsl_rl，详见
  `docs/design/cerebellum_policy.md`。
- HumanML3D 学术许可；KIT-ML 为 rar、AMASS 为空，未参与训练且脚本显式报错。
- LayA 为结构化决策模型，不进入 50 Hz 控制回路，不输出关节目标。
