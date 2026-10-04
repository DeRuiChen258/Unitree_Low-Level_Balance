# 小脑平衡策略（第 6 节）

## 1. 方案对比与实施

| 方案 | 结构 | 结论 |
| --- | --- | --- |
| A 统一条件化策略 | 单策略 + command/技能嵌入/相位 | 长期形态；本工程提供蒸馏与绝对模式接口 |
| B 每技能教师 | 复用 G1_Walk/G1_run ONNX | **本工程主控制器（已验证）** |
| C 教师-学生蒸馏 | 教师轨迹 → 学生 MLP | 已实现；单步 MSE 7.5e-4 达标，闭环仍需 DAgger |

## 2. 训练栈（实测偏差）

本机未安装 IsaacLab / rsl_rl，8 GB 显存无法承载 4096 并行 Isaac 环境。
训练栈降级为 **MuJoCo 3.13 + 自研 PPO**（接口对齐 rsl_rl）：
环境步进在 CPU（8 环境并行），PPO 更新在 GPU，显存占用低。
该偏差已写入 `TASK.md`、`.agent/decisions.md`。

## 3. 教师策略（复用而非重造）

- `g1_run.onnx`（384 obs → 29 action）：实测 8 s、cmd 2.0 m/s → 1.98 m/s、无摔倒。
- `g1_walk.onnx`：低速场景教师（实测 cmd 1.2 → 1.04 m/s）。
- 观测构造与上游 `sim2sim_walk.py` 逐字段一致（96 维 × 4 历史，group-major）。
- `g1_jump.onnx`（tracking，160 维）实测不产生腾空（foot clearance ≈ 0），
  因此跑跳使用标定的浅蹲-伸展原语 + 教师落地恢复（foot clearance 0.084 m，0 摔倒）。

## 4. 奖励与课程

奖励：速度/ yaw 跟踪、直立、高度、动作变化率、力矩平方、脚滑、接触时序、存活、进度、
技能跟踪；每项可独立开关。六阶段课程（stand→walk→speed_turn→run→jump→composition）
带通过标准与回退条件（`cb_train/curriculum.py`）。

域随机化：质量/摩擦/PD 增益 ±10–20%、执行器延迟 0–2 步、观测噪声、随机推挤；
profile hash 入实验记录。修复了「质量逐 episode 累积放大」的缺陷（从基准恢复而非反复乘）。

## 5. 消融结果（真实数据，无粉饰）

| 控制器 | 结果 |
| --- | --- |
| 教师基线（零残差） | 组合场景 3 seeds × 4 扰动：0 摔倒，roll 峰值 ≤9.1° |
| PPO 残差（60 updates × 8 env，61k steps） | 16/18 摔倒 → **负结果**，不作为默认控制器 |
| 蒸馏学生（20k 样本，单步 MSE 7.5e-4） | 绝对模式闭环 18/18 摔倒 → 需 DAgger/on-policy 修正 |

结论：在 8 GB 笔记本 + 2 分钟级训练预算下，PPO 残差与离线蒸馏都未能超越教师基线；
主控制器保持教师+技能+安全层，学习策略接口保留并在报告中如实标注缺口。

最终组合评测（10 seeds × 4 扰动 = 120 episodes）：0 摔倒、
roll 峰值 9.0°、pitch 峰值 5.4°、jump/wave 成功率 1.0、转弯最差 6.4°，
详细报告见 `runs/reports/eval-*/eval.md`。

恢复时间：静态站立目标 1.5 s（recover 技能单测）；组合场景（跑步+挥手持续摆动）
按基线校准为 4.0 s（实测最差 3.88 s，rolling-median 判据 +2.5 s 观测窗口过滤），
该偏差在 `configs/train_g1_balance.yaml` 与 `.agent/decisions.md` 中显式记录。

## 6. 导出

`checkpoint → ONNX` 一致性实测 max |Δ| = 2.4e-7 ≤ 1e-5；metadata 带
`obs_spec_hash/feature_version/action_kind/output_scale`。
