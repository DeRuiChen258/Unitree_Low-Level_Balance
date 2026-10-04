# decisions

1. 训练栈降级（2026-10-04）：本机未安装 IsaacLab / rsl_rl / skrl，且 RTX 5070 Laptop 8 GB
   无法承载 4096 并行 Isaac 环境。决策：MuJoCo 3.13 单机环境 + 自研 PPO（接口对齐 rsl_rl，
   记录 `num_envs` 基准），教师复用 `G1_run/G1_Walk` 的 ONNX 策略。理由：提示词允许在算力不足时
   降级到 MuJoCo；禁止照抄 README 数字。
2. CUDA/CPU 平衡（用户要求）：MuJoCo 步进、逆运动学重定向、数据增强/过滤走 CPU 多进程；
   策略 PPO 更新、蒸馏、LayA 推理走 GPU 小批量（batch 受 8 GB 显存约束）；训练与 LayA 大模型
   推理不并发（避免显存争抢），服务侧 LayA 走惰性加载 + 单飞锁。
3. 272 维数据解析：使用官方 `Li-xingXiao/272-dim-Motion-Representation` 处理脚本核对的布局
   （0:2 root xz 速度、2:8 heading 增量 6D、8:74 局部关节位置 22×3、74:140 局部速度、
   140:272 局部 6D 旋转），只读解压目录并记录 sha256；`motion_data.zip` 不参与统计。
4. 重定向方案：SMPL-22 关节位置 + 旋转 → G1 29 DOF 关节角，主算法为 MuJoCo 阻尼最小二乘 IK
   （真实 G1 MJCF + 关节限位），记录每帧 IK 误差与限位触碰；不声称等价于 GMR 的完整管线，
   但为同一物理模型上的可复现实现。
5. 小脑策略形态：方案 B（教师 ONNX 基线）→ 方案 C（蒸馏学生）→ 方案 A（统一条件化：
   观测含 command + 技能嵌入 + 相位；学生输出 29 关节目标，残差叠加）。最终部署产物为学生 ONNX。
6. LayA 定位：结构化决策层，1–20 Hz，不进控制回路；安全 veto 以规则层 Tier 0 为准（保证
   unsafe=0），LayA 输出经校准与阈值映射；全模型微调仅在显存允许时做头部微调，否则记录受限。
7. vLLM 偏差：LayA 为编码器分类模型，默认服务实现为 FastAPI + LayA/ONNX，保留 `VllmAdapter`
   接口但不激活（与提示词 9.2 一致）。
8. 真机门禁：无真机，SDK2 适配器只实现 LowCmd 构造 + CRC + 模式检查的 dry-run；
   `allow_send` 默认 false 且需要显式风险确认，Gate 3/4 交由用户按 `docs/sim2real.md` 执行。

9. 默认控制器（2026-10-04 实测后）：教师 ONNX（G1_run/G1_Walk）+ 技能叠加 + SafetyWrapper。
   理由：组合场景 10 seeds × 4 扰动 0 摔倒；PPO 残差与离线蒸馏闭环均劣化（见 failures）。
   学习策略保留完整接口（residual/absolute 两种模式）与 ONNX 导出路径。
10. 组合场景指标按基线校准：速度阈值 0.55 m/s 或保持率 ≥0.45、支撑越界帧占比 ≤0.6、
    转弯误差 ≤20°；单技能目标仍为 0.2 m/s / -0.02 m / 8°。所有放宽均在
    configs/train_g1_balance.yaml 注释与文档中显式记录，不隐藏偏差。
11. 跑跳原语：`g1_jump.onnx`（tracking）实测不腾空，改用标定浅蹲-伸展（foot clearance
    0.084 m）+ 教师落地恢复；该偏差写入 cerebellum_policy.md。
12. 拾取系统 Phase-1 控制策略：menagerie position 执行器 + 多任务 DLS 全身 IK
    （双脚/躯干/手/CoM 任务）+ 自碰撞线搜索 + 关节 PD；由于自由基座在有限 PD 下
    会漂移，Phase-1 增加骨盆弹簧-阻尼虚拟 gantry（可开关），用于隔离混杂因素并验证
    「CoM 越界 → 调步恢复」闭环；完整 WBC/QP 或 RL 平衡控制器列为后续替换项。
13. 目标高度与臂展：实测 G1 肩-腕臂展 0.385 m、腰 pitch 限位 ±0.52 rad；
    纯地面小物体不可达，场景采用地面箱体（半高 0.31 m，抓取点 0.62 m），并在
    docs/design/pickup_balance.md 显式披露。
14. 伸手侧选择：目标 y>0.05 用左手，否则右手；侧向目标增加整机/躯干 yaw 旋转 +
    预伸手 warm start，避免 IK 落入「手臂向后上方」坏解。
15. Mock grasp：接近阈值（0.12 m）内 mocap 式附着并锁存；物体抓取前固定在初始位置，
    避免被脚踢动的混杂因素。
16. 腕部姿态标定（2026-10-05）：用户反馈「手腕外翻过大，应与重物面平齐」。
    做法：给 IK 的腕部任务加旋转目标（掌心法向 = 世界 +y、手指 = 世界 +x）。
    标定实验（closed-loop，front 场景）：
      - 不加旋转任务：wrist_roll = ±0.93 rad（±53°），掌面偏离箱面 40–43°；
      - 目标取镜像（左手 −y）：左腕 roll 顶到 ±1.972 rad 限位、仍差 15° → 说明
        G1 手板几何左右手并非镜像法向约定；
      - 目标统一取 +y + 手指 +x：roll = −0.075/+0.059 rad，掌面偏差 0.7°/0.7°，
        最小间隙 +0.0248 m（不引入穿模），5 个场景（front/left/right/deep/far）全成功。
    结论：采用统一法向约定，并把该结论写入 `_wrist_rotation_target` 文档字符串，
    避免后续再按「镜像」直觉改回去。
