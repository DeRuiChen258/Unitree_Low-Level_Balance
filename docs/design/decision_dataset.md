# 决策数据集标签定义（第 8.5 节）

## 来源

1. **仿真 rollout**：三组合场景 + 扰动/噪声/延迟，每 4 个控制步（5 Hz）采样一次状态；
2. **规则专家标签**：`safety_level_from_state()` 按 roll/pitch/CoM 裕度/接触给出金标准 veto；
3. **人工标注**：边界与升级样本（接口保留；本机无人工标注流程，未伪造）。

## 标签

| 头 | 标签来源 |
| --- | --- |
| motion_primitive_head | 场景脚本技能 + 速度/yaw 状态映射 |
| safety_veto_head | Tier 0 规则（roll/pitch ≥8°、margin<-0.02、单脚超时、NaN、E-Stop） |
| recovery_head | 由安全等级映射：none / small_step / stand_recover / safe_stop |
| escalation_head | 摔倒、EMERGENCY、连续失败 |

## 泄漏控制

同一 rollout 的连续帧只能落在一个 split；场景模板同时出现在 train/val/test 但
种子不同；manifest 记录 rollout 列表与 split 映射。

## 对抗样本

包含「看似正常但物理危险」的样本：CP 裕度骤降、单脚支撑超时、推挤后未恢复。
