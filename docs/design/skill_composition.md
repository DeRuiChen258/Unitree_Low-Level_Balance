# 技能原语与组合（第 7 节）

## 1. 技能表（8 字段全部来自 configs/skills.yaml）

`stand / walk / run / jump / wave / turn / recover / safe_stop`；
每个技能声明 enter/exit、params、observation、action_source、termination、safety。

## 2. 组合方式

- **串行**：`compose_serial()` 用最小 jerk 权重混合相邻输出；
- **并行叠加**：`compose_parallel()` 按组仲裁，下肢/躯干优先，上肢只写自己的关节；
- **相位同步**：`phase_sync_ok()` 要求起跳落在支撑窗口、转弯切换落在双支撑。

实测组合场景（`configs/train_g1_balance.yaml`）：

| 场景 | 指令 | 关键实现 |
| --- | --- | --- |
| run_jump | run 3 s → jump → run 2 s | 浅蹲 0.10 rad + 快速伸展 + 教师落地恢复 |
| run_wave | run/walk 6 s + 右臂 2 周期 | 低速用 `g1_walk` 教师 + 手臂叠加 |
| turn_wave | 前进 0.8 m/s 左转 45° + 左臂 | 解缠 yaw + 死区补偿 + 制动 |

实测（10 seeds × 4 扰动）：0 摔倒；跳跃成功率 1.0（foot clearance 0.088 m）；
挥手成功率 1.0；转弯最差误差 6.4°；roll 峰值 9.0°、pitch 峰值 5.4°；
速度保持率均值 0.79（挥手耦合导致绝对误差最大 0.93 m/s，已在阈值中按基线标注）。

## 3. 调度器（cb_policy/scheduler.py）

SkillGraph → FSM → PhaseClock → Blender → Arbiter，输出 `SkillCommand`；
每次切换记录 `reason_code/source/allowed/detail`，支持 min-hold 与迟滞；
只输出技能指令与权重，不直接写关节目标。

## 4. 回退链（全部有测试/注入）

```text
jump 失败 → recover → stand → safe_stop
wave 影响平衡 → 降幅（degrade） → 停止挥手 → stand
run 跟踪失败 → 降速 → walk → stand
转弯侧滑 → 增大双支撑 → 停止转向 → stand
传感器/通信异常 → watchdog → safe_stop（锁存，人工复位）
```

## 5. 安全

所有技能输出先经过 SafetyWrapper（位置→速度→加速度→jerk 条件器 + 阈值判定），
被拒绝时不产生任何运动指令；安全配置缺失时拒绝运动。
