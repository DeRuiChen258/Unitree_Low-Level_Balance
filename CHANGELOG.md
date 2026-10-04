# Changelog

## [1.0.0] - 2026-10-04

### Added（新增需求 v1.0：自主弯腰拾取 + 自主弓步）

- `src/cb_pickup/`：Balance Monitor（COM/ZMP/CP/支撑域/预测裕度）、FootstepPlanner
  （capture-point 落脚律 + 约束）、9 个行为原语（Stand/Bend/Step/Lunge/Reach/Grasp/Lift/
  StandUp/Recover）、MotionManager（FSM + 稳定门禁 + 去抖）、JEV-like 决策
  （RuleBased/MLP）、WholeBodyIK（多任务 DLS + 自碰撞线搜索）、MockGrasp（mocap 附着）、
  PickupBalanceEnv + reward、4 baseline（A 固定脚 / B 固定阈值 / C 平衡感知 / D JEV-like）、
  10 个消融开关、实验记录与 10 图可视化。
- CLI：`run_pickup.py`（一条命令完整 Demo + 视频/viewer）、`run_baseline.py`、
  `run_ablation.py`、`evaluate.py`。
- 配置：`configs/g1_pickup.yaml`、`configs/pickup_balance.yaml`、
  `configs/pickup_randomization.yaml`。
- 实测：10 场景 10/10 拾取成功、0 摔倒；front 21.46 s、1 次脚步、裕度 0.008→0.091。
- 测试：`tests/test_pickup_*.py` 12 项全绿。

### Added

- 数据管线 8 步：272 维 ingest → 规范化 → MuJoCo DLS IK → 切片 → 增强 → 物理过滤 →
  按源切分 → manifest/质量报告；真实数据 smoke 运行 96 片段 → 308 段（26 拒绝）。
- 特征工程：434 维 obs spec（训练/部署 hash 一致）、平衡/相位/技能/文本化状态。
- 小脑策略：教师 ONNX（G1_run/G1_Walk）复用 + 技能叠加 + PPO 残差训练栈 +
  离线蒸馏学生 + ONNX 导出（误差 2.4e-7）。
- 技能与组合：stand/walk/run/jump/wave/turn/recover/safe_stop + 串行/并行/相位同步 +
  调度器（min-hold/迟滞/仲裁/日志）。
- 三组合场景（run_jump/run_wave/turn_wave）× 四扰动（none/push/noise/delay）× 多 seed
  批量评测与报告。
- LayA 决策层：四头 questions、映射、温度校准、决策数据集、fail-closed 策略、
  FastAPI 服务（单条/批量/health/version）。
- 安全层：限位/速率/jerk 条件器、看门狗、状态机、E-Stop、SafetyWrapper；
  SDK2 LowCmd+CRC dry-run。
- 72 项 pytest、`scripts/verify.py --all` 验收门禁、9 份设计文档与 `.agent/` 记忆。

### Known limitations

- 无真机：Gate 3/4 未执行。
- IsaacLab 未安装：训练栈为 MuJoCo 降级实现。
- PPO 残差与离线蒸馏在 smoke 算力下未超越教师基线（记录为负结果）。
- AMASS 为空、KIT-ML 未解压，未参与统计。
