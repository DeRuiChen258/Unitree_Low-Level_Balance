# failures

- 2026-10-04：直接用 `G1_run` motor 模型 + 从零 PD 站立会失稳（膝盖弯曲默认姿态 + 未使用上游
  reset 流程）。改用它自己的 `Sim2SimController` 作为教师基线后通过；自研环境若做 torque PD
  必须复刻上游 reset/姿态，否则改用 menagerie position 执行器。
- 2026-10-04：`conda run -n unitree_rt python - <<'PY'` 不传递 stdin，改用当前已激活 shell 的
  `python - <<'PY'`。
- 2026-10-04：PPO 残差（60 updates × 8 env，init std 0.25）闭环评测 16/18 摔倒，
  显著劣于教师基线（0/18）。处置：记录为负结果；默认控制器不使用该残差；
  后续需要奖励重标定、残差幅值门控与 on-policy 评估后重训。
- 2026-10-04：离线蒸馏学生单步 MSE 7.5e-4 达标，但绝对模式闭环 18/18 摔倒
  （compounding error）。处置：保留为可选研究产物，需要 DAgger/on-policy 数据修正；
  报告中不得声称其可部署。
- 2026-10-04：`check_termination` 曾用 `2·acos(|w|)` 作为倾倒角，把 yaw 计入导致原地转弯
  被误判摔倒；已改为重力方向投影。
- 2026-10-04：域随机化曾在同一 MjModel 上逐 episode 乘质量缩放，导致后续 episode 动力学
  漂移与假性摔倒；已改为保存 base mass/friction 后恢复。
- 2026-10-04：LayA `typed-decisions` 单流单条决策 P99 = 323 ms，未达 150 ms 目标
  （P50 77 ms / P95 99 ms 达标）。处置：记录为已知缺口；服务侧缓解路径为 batch 窗口、
  ONNX Runtime、缩短选项文案（需升 questions_version 并重校准）；控制回路 5 Hz +
  hold-last-valid，不受该延迟阻塞。
- 2026-10-04：安全层曾把跑步/跑跳的正常姿态瞬态（8° tilt、2 rad/s 角速度、踝 11 rad/s）
  判为 ABORT，导致 demo 提前中断。处置：速度/角速度阈值改为「工程估算 + 动态技能实测」
  并显式标注真机首测需重新核定；倾斜阈值保持保守。
- 2026-10-04：拾取系统初版「全身 IK + 位置执行器 + 自由基座」在弯腰/弓步时向后滑倒
  （base 漂移 0.1–0.7 m，脚滑/接触约束不匹配）。处置：引入可开关的虚拟基座稳定辅助
  （Phase-1 gantry），并把完整 WBC/RL 列为后续替换项；脚滑与物体被踢动的混杂因素
  通过脚锚定（默认关闭）/物体抓取前固定来隔离。
- 2026-10-04：手臂 IK 解与躯干自碰撞，position 执行器输出 -676 Nm 仍无法到位；
  处置：IK 增加自碰撞线搜索 + 侧向目标预伸手 warm start（左侧用左臂）。
- 2026-10-04：抓取后 mock 附着因手-物瞬时距离超过 release 阈值而释放，LIFT 卡死；
  处置：抓取锁存直到显式 release。
- 2026-10-05：`scripts/demo.py` / `scripts/demo_balance_extra.py` 在 `--render viewer`
  下进程退出阶段 segfault（exit 139），发生在 summary.json 落盘之后；
  `--render video` 与 `--render none` 均正常（exit 0），最小复现脚本
  （MuJoCo + viewer）与 onnxruntime+viewer 均不崩，属本机 Wayland/EGL + GLFW
  上下文回收的环境问题（试过去掉 `viewer.close()`、强制 X11 均无效）。
  处置：`start.sh` 的 `run_balance()` 只把 139 记为 note 并继续（其它非零码仍然中止），
  需要无崩溃输出时用 `RENDER=video`。同类现象不影响仿真与指标。
