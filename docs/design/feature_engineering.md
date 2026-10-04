# 特征工程（第 5 节）

## 1. 观测 spec（训练/部署同源）

`src/cb_features/obs_spec.py` 定义分组、维度、单位、归一化与缺失策略：

| 组 | 维度 | 历史 | 单位 |
| --- | --- | --- | --- |
| ang_vel | 3 | ✓ | rad/s |
| gravity（projected） | 3 | ✓ | unit |
| lin_vel | 3 | ✓ | m/s |
| command | 5 | ✓ | m/s, rad/s, flag |
| joint_pos（相对默认） | 29 | ✓ | rad |
| joint_vel | 29 | ✓ | rad/s |
| last_action | 29 | ✓ | rad |
| contact | 2 | ✗ | bool |
| phase_clock | 4 | ✗ | unit |
| skill_embedding | 16 | ✗ | unit |
| balance_feats | 8 | ✗ | mixed |

历史组按 group-major 堆叠：`7×101` 单帧 → `404`；总观测 **434** 维。
`ObsSpec.hash()` 写入 checkpoint/ONNX metadata，部署端加载前校验，不匹配直接拒绝。

## 2. 平衡特征（8 维）

roll/pitch 误差、CoM 相对量（xy）、CP（近似）、支撑域余量、平衡综合分。
支撑域由左右踝 roll link 的四个足角点构成凸多边形；CoM 取 MuJoCo `subtree_com`。
动态动作允许 CoM 短时越界，评测统计越界帧占比与最长持续时间（见 cerebellum_policy.md）。

## 3. 相位

步态相位由接触切换对齐（sin/cos）；技能相位 `[0,1)`；跳跃五段
`approach/takeoff/flight/landing/recover`。转弯使用解缠 yaw（避免 ±π 回绕）。

## 4. LayA 文本化状态

`build_text_state()` 输出稳定 JSON：固定键序、固定小数位、候选动作由调度器注入：

```json
{"balance":{"com_margin_m":0.05,"contact":["left"],"cp_margin_m":0.03,
 "pitch_deg":-1.4,"roll_deg":2.1,"stability":"stable"},
 "candidates":["continue_current","run","recover","safe_stop"],
 "context":{"battery_ok":true,"external_push":false,"latency_ms":12.0,"terrain":"flat"},
 "history":{"falls_last_60s":0,"watchdog_trips_last_60s":0},
 "motion":{"phase":0.62,"skill":"run","time_in_skill_s":1.8,"vx":2.1,"vy":0.0,"yaw_rate":0.4},
 "robot":"unitree_g1_29dof","state_version":"1.0","task":"run_then_wave_and_turn"}
```

模板版本变更视为新特征版本，必须重新校准并跑决策回归。
