# memory

- 教师策略事实：`Action/G1_run/deploy/g1_deploy/exported_policy/g1_run.onnx` 输入 [1,384]，
  输出 [1,29]；配置 `g1_amp.yaml`：control_dt=0.02、sim_dt=0.005、decimation=4、history=4，
  PD 增益与关节序（isaac_to_mujoco_map）是该部署栈的事实标准。
- Sim2Sim 复测（2026-10-04）：G1_run 教师 8 s、cmd 2.0 m/s → mean 1.98 m/s、fall=False、
  peak tilt 0.143 rad、sim/wall ratio 0.015；可直接作为 run 原语教师。
- MuJoCo 事实：`mujoco_menagerie/unitree_g1/scene.xml` 为 position 执行器（kp=500, dampratio=1），
  以 keyframe `stand` 重置后 5 s 零控制稳定（height 0.792，roll/pitch≈0）；适合做站立/平衡环境。
  `G1_run/deploy/.../scene_29dof.xml` 为 motor 执行器 + 代码 PD，与教师策略配套。
- G1_Waving `common/safety.py` 的三级安全（WARN/ABORT/EMERGENCY）与命令条件器（位置→速度→
  加速度→jerk）是安全层阈值设计来源；`BalanceController` 协议是残差补偿接入点。
- HumanML3D_272d 是 MotionStreamer 272 维表示，不是 HumanML3D 263 维；解析必须用 272 布局，
  263 维脚本不可直接套用。
- G1 蹲起关节符号：hip_pitch 负 + knee 正 + ankle_pitch 负（如 -0.5/+1.0/-0.5）稳定；
  跳跃伸展用 knee→-0.05、ankle_pitch→+0.15~0.2，浅蹲 0.12 rad 可稳定离地并被教师恢复。
- G1_Walk 教师速度曲线（实测，6 s 稳态）：cmd 0.4→0.0、0.6→0.26、0.8→0.58、1.0→0.81、
  1.2→1.04 m/s；低速有死区，低速场景需选 walk 教师或提高命令。
- 挥手对速度耦合显著：手臂叠加会让 walk 教师掉速约 0.4 m/s；组合场景需记录速度保持率。
- 摔倒判定必须用重力方向（roll/pitch），不能把 yaw 计入 total rotation，否则原地转弯会被误判为摔倒。
- 域随机化的质量/摩擦必须从保存的基准数组恢复后缩放，不能每 episode 在原模型上重复相乘。
- 教师 tracking 策略 `g1_jump.onnx` 在 Sim2Sim 中不产生腾空（foot clearance≈0）。
- LayA `typed-decisions` 真机加载：代码 `rl_agent_api.py` 在 checkpoint 的上一级目录，
  需把该父目录加入 sys.path；加载 + 3 次预热约 12 s；单条决策 P50 77 ms/P99 323 ms
  （本机 8 GB，长选项文案）。
- 安全层命令阈值 vs 测量状态容差：命令按 position_margin 严格检查，测量状态允许 1e-2 rad
  数值越界；跑跳落地角速度峰值 196 deg/s、踝速度 11.3 rad/s，阈值必须覆盖动态技能。
- 可视化自检：`mujoco.Renderer.update_scene(data, camera=MjvCamera)` 必须每帧设置 lookat，
  否则机器人跑出视野只剩地面；视频 fps 应与控制频率一致（50 Hz）。
- G1 menagerie 模型：position 执行器（kp=500，dampratio=1），物理默认 dt=0.002（500 Hz），
  足端为每脚 4 个球接触（heel x=-0.05、toe x=+0.12、y=±0.03），支撑域 0.17×0.06 m/脚。
- G1 臂展实测：肩-腕 0.385 m；腰 pitch 限位 ±0.52 rad；直立站立时手腕最低约 0.7 m，
  深蹲+前倾（base 0.62、pitch 0.5）可到 ~0.62 m。
- 直腿站姿是 IK 奇异位形：必须以微屈膝（policy default: hip -0.1/knee 0.3/ankle -0.2）
  为初值，否则 DLS 会收敛到膝超伸的坏解。
- 非对称关节空间「弓步」会把一只脚抬离地面（FK 脚 z 0.06–0.11 m）导致侧翻；
  正确做法是用足端位置任务 + CoM 任务求解全身 IK。
- 位置执行器 + 自由基座时，IK 假设的 base 位姿与实际不一致会造成漂移；
  Phase-1 用骨盆弹簧-阻尼 gantry 稳定 base，同时保留真实 CoM/支撑域观测。
- 手臂 IK 会解出自碰撞构型（肩 pitch -84° 时力 -676 Nm 卡住）；
  WholeBodyIK 增加机器人体段接触线搜索 + 预伸手 warm start 后解决。
- Capture-point 落脚律（落脚 = cp + 0.06·away_dir）+ 主动弓步偏置（0.18 m）
  在 10 个场景中 0 摔倒。
