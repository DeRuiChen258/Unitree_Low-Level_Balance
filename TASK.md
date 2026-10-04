# TASK: G1 小脑平衡 + 多动作自适应组合系统（完整工程实现）

> 创建：2026-10-04
> Workflow：unitree（模式 C：开发执行）
> 技能：unitree_g1_developer_guide → mujoco_simulation / python_sdk2 → debug_network_and_safety
>      → rl-infra / benchmark（仅训练与性能阶段）→ task-guard / chinese-response
> 任务书：`Prompt/小脑平衡多动作自适应系统_完整实现提示词.md`（v1.0）
> 边界文档：`Prompt/总目标.md`

## Objective

在 `Low_Level_balance/` 实现提示词第 11 节目录树中的完整工程：数据合成与增强 → 特征工程 →
小脑平衡策略（复用教师 + 蒸馏 + PPO 残差）→ 技能原语与组合调度 → LayA 结构化决策 →
服务化 → 安全门禁与真机预留（Gate 0–2），并通过第 12 节 `scripts/verify.py --all`
与 `pytest` 全绿验收。最终审查完成后启用 MuJoCo 可视化演示（viewer + 离线视频）。

## 勘察结论（P0，2026-10-04 复测）

| 项 | 实测 |
| --- | --- |
| Python / 环境 | conda `unitree_rt`，Python 3.12.9，`/home/violet/Workspace/miniconda/envs/unitree_rt` |
| PyTorch / CUDA | 2.14.0+cu130，CUDA True；RTX 5070 Laptop 8151 MiB，驱动 615.71.09 |
| MuJoCo | 3.13.0（viewer 可用，DISPLAY=:0）；G1 模型：`mujoco_menagerie/unitree_g1`（position 执行器）+ `G1_run/deploy/.../scene_29dof.xml`（motor 执行器 + PD） |
| ONNX Runtime | 1.30.0；`g1_run.onnx` obs[1,384]→act[1,29]，实测 8 s 平均 1.98 m/s、无摔倒、peak tilt 0.143 rad |
| 教师资产 | `Action/G1_run/.../g1_run.onnx`（2.0 m/s）、`Action/G1_Walk/.../g1_walk.onnx`（AMP 走）、`g1_jump.onnx`（tracking/160 维，待接通） |
| 动作工程 | `G1_Walk`、`G1_run`、`G1_Waving`（84 测试通过）、`legged_rl_lab`（IsaacLab 2.3，**本机未安装**） |
| 数据 | HumanML3D_272d：26846 个 `(T,272)` 片段 @30 FPS；train/val/test=23384/1337/4041；Mean/Std 272 维；272 维布局已用官方处理脚本核对 |
| 数据缺口 | AMASS 为空；KIT-ML 为 `.rar`；HumanML3D(263d) 仅解出 1 个样本；均不得伪装成已用数据 |
| LayA | `typed-decisions`（ModernBERT-large，1024 ctx，842 MB）+ `multilingual`（mmBERT-base，643 MB）+ `rl_agent_api.py` / `rl_agent_config.json` |
| SDK2 | `unitree_sdk2py` 可导入（`Embedded_code/unitree_workspace/sdk`）；无真机 |
| 计算策略 | MuJoCo 环境步进与数据管线走 CPU 并行；策略训练 / LayA 推理走 GPU 小批量；避免 8 GB 显存争抢 |

## 交付物

| 类 | 内容 |
| --- | --- |
| 代码 | `src/cb_{common,data,features,train,skills,policy,decision,eval,serving,safety,robot}/` |
| CLI | `scripts/{data_build,data_quality,retarget,train,export_onnx,eval,decision_data,decision_finetune,replay,demo,serve,verify}.py` |
| 配置 | `configs/{system,data,skills,transitions,reward,train_g1_balance,serving}.yaml`、`configs/safety/*`、`configs/laya/*` |
| 测试 | `tests/`（数据/特征/训练/技能/决策/安全/服务/机器人） |
| 文档 | `docs/design/` 9 份 + `docs/sim2real.md` + `README.md` |
| 产物 | `runs/data_quality/<v>/`、`runs/<exp>/`（checkpoint+ONNX+报告）、`runs/decision/<id>/`、`runs/reports/` |

## 约束与决策（详见 `.agent/decisions.md`）

1. 无真机：硬件结论止于 Gate 2（编译级 / 启动级 / 接口级 / dry-run）。
2. IsaacLab/rsl_rl 本机不可用 → 训练栈降级为 MuJoCo + 自研 PPO（接口对齐 rsl_rl）；教师复用现有 ONNX。
3. LayA 只输出结构化决策，不进入硬实时回路；服务默认 FastAPI，vLLM 仅保留适配器。
4. 272 维数据只使用解压目录，`motion_data.zip` 不参与统计；KIT-ML/AMASS 缺失时显式报错。
5. CUDA/CPU 平衡：env 步进/IK/数据管线 CPU 并行，训练与 LayA 推理 GPU 小批量。

## Checklist（门禁，逐项勾选）

### P0 勘察与环境
- [x] 第 2.5 节勘察命令复测，版本/DOF/资产写入 TASK.md
- [x] 教师策略 Sim2Sim 实测（`g1_run.onnx`：1.98 m/s，无摔倒）
- [x] `TASK.md` + `.agent/{state,decisions,memory,failures}.md` 建立

### P1 数据管线
- [x] `src/cb_data/*` 8 步管线实现
- [x] `scripts/data_build.py` 退出码 0，`runs/data_quality/<v>/{manifest.json,report.md,stats.csv}` 落盘
- [x] 固定 seed 重跑一致；拒绝原因/IK 误差/接触一致性有报告
- [x] 数据单测 `pytest -k data` 全绿

### P2 特征工程
- [x] `src/cb_features/*` 实现，训练/部署同一 `obs_spec_hash`
- [x] `pytest -k features` 全绿

### P3 训练
- [x] 教师蒸馏 + PPO 残差训练产出 checkpoint
- [x] ONNX 导出与 PyTorch 误差 2.4e-7 ≤ 1e-5
- [x] 基线 vs 残差消融报告（含负结果，见 `.agent/failures.md`）

### P4 组合
- [x] `src/cb_skills/*` + `cb_policy/scheduler.py`
- [x] 三场景 × 四扰动 × 10 seeds 评测，含最差 seed
- [x] 回退链故障注入测试全绿

### P5 决策
- [x] LayA 适配 + questions/映射/校准/微调/评测
- [x] 四头指标 + ECE 前后对照 + unsafe=0 + P99 延迟（见 runs/decision）
- [x] `pytest -k decision` 全绿

### P6 服务
- [x] FastAPI `/v1/{decision,batch,health,version}` 测试通过
- [ ] `scripts/demo.py` 端到端跑通（含回退）→ 最终可视化演示

### P7 安全与验收
- [x] 安全层 + 故障注入 + E-Stop + 配置缺失 fail-closed
- [ ] `scripts/verify.py --all` 退出码 0（待 demo）
- [x] 9 份设计文档齐全一致
- [ ] `.agent/` 更新完毕（最终收尾）

### Verify（最终）
- [x] `pytest -q` 全绿（exit 0，41 个测试文件）
- [x] `python scripts/verify.py --all` 退出码 0（16 项门禁全绿）
- [x] 可视化演示：MuJoCo viewer 启动成功 + 离线 MP4 生成
      （`runs/demo/` 运动演示；`experiments/*pickup*/video/pickup.mp4` 拾取演示）

## Known Issues（随进度更新）

- 真机不可用：Gate 3/4 未执行。
- IsaacLab 未安装：训练栈记录为 MuJoCo 降级实现。
- KIT-ML / AMASS 未参与训练：数据集缺失，脚本显式报错而非静默跳过。
- `g1_jump.onnx` 为 tracking 策略，接入需自建 160 维观测；若不可用则 jump 采用参考轨迹 + 残差平衡。

---

# 追加任务：G1 自主弯腰拾取 + 自主弓步（Prompt/新增平衡需求.md）

## Objective（追加）

实现「状态感知 → 稳定性预测 → 行为决策 → 脚步调整 → 动作执行 → 再评估」闭环：
弯腰拾取 + 自主脚步/弓步 + JEV-like 决策 + 4 baseline + 10 消融 + 实验记录 + 可视化，
一条命令可运行完整 Demo。严格限于 MuJoCo。

## Checklist（追加）

- [x] Phase 1–2：G1 MuJoCo 可运行、稳定站立（menagerie position 执行器）
- [x] Phase 3–4：弯腰 + COM/支撑域监控（BalanceMonitor）
- [x] Phase 5–7：单步 / 弓步 / 自主调步（FootstepPlanner + 主动弓步）
- [x] Phase 8–10：伸手 / mock 抓取 / 抬升 / 起身完整组合
- [x] Phase 11：JEV-like 决策层（RuleBased + MLP 回退 + 置信度/风险/fallback）
- [x] Phase 12：PickupBalanceEnv（primitive ID 动作 + reward）
- [x] Phase 13：域随机化配置（10 场景 + 噪声/质量/摩擦/延迟/物体）
- [x] Phase 14：baseline A–D + 消融 1–10 框架与 CLI
- [x] 验收 Test 1–14：10 场景 10/10 拾取成功、0 摔倒、轨迹/决策/视频完整记录
- [x] 可视化：10 图 + 关键 COM/支撑域/脚步图 + MP4 + viewer
- [x] 文档：`docs/design/pickup_balance.md`、README、CHANGELOG、`.agent/`
- [x] baseline 对比与关键消融已实跑（`experiments/{evaluation,ablation}/`）
- [ ] 完整 WBC/RL 平衡控制器替换 Phase-1 虚拟 gantry（后续路线，当前如实披露）
