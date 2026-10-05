# LayA 决策层（第 8 节）

## 1. 定位与接口

LayA 是编码器结构化决策模型（非自回归），输入文本/JSON + typed questions，
输出 choice/score/noul 概率；决策频率 1–20 Hz，不进入 50 Hz 硬实时回路。
输出映射为 `DecisionOutput`：

```text
action_id / primitive / confidence / risk / need_human_review /
fallback_action / probabilities / reason_code / heads / meta
```

## 2. 四个头

`motion_primitive_head`（choice，候选项由调度器注入）、`safety_veto_head`（noul）、
`recovery_head`（choice）、`escalation_head`（noul）；定义与版本在
`configs/laya/questions.json`，hash 进入每次决策记录。

## 3. 校准与阈值

- 温度继承 `rl_agent_config.json`（typed-decisions: 1.0148/1.0374/1.0575；
  multilingual: 1.0/1.0/1.0），分桶按 `temperature_by_options`；
- 阈值：confidence 0.60、risk 0.35、veto fail-closed；
- 回退：低置信 → stand；高风险 → safe_stop；超时/异常 → hold_current_skill + 人工复核；
- 升级：连续失败 3 次或决策翻转超限。

## 4. 数据集（docs/design/decision_dataset.md）

来源：仿真 rollout + 规则专家标签（Tier 0 安全规则）+ 危险边界样本；
按 rollout 切分（同源不跨 split）；标签含 primitive/veto/recovery/escalation。

## 5. 微调与校准

`cb_decision/finetune.py` 在留出集上拟合分桶温度并输出校准参数；
8 GB 显存下不做全编码器微调（已在报告中标注），但保留完整 `adapter.py`
可直接加载 `typed-decisions`/`multilingual` checkpoint。

## 6. 指标

accuracy/P/R/F1（按头）、ECE 与可靠性曲线（校准前后）、P50/P95/P99 延迟、
fallback rate（按原因）、unsafe decision rate（Tier 0 规则保证 0）、
escalation rate、decision-flip rate。产物：`runs/decision/<id>/`。

## 7. 实测（2026-10-04，RTX 5070 Laptop 8 GB）

- `typed-decisions` checkpoint 真实加载成功（`model_ready=true`，加载 + 3 次预热 ~12 s）；
- 单条决策（含文本化 + 4 头前向 + 后处理）：P50 77 ms、P95 99 ms、P99 323 ms、mean 85 ms；
- **P99 目标 150 ms 未达成**：单流单条推理在长选项文本下出现偶发 300 ms 尾部。
  缓解路径：服务侧 batch 窗口累积（batch 8/16）、ONNX Runtime 导出、缩短 motion 头选项文案
  （需升 `questions_version` 并重新校准）、降频到 5 Hz + hold-last-valid（控制回路不等决策）。
  该偏差已记录于本文档与 CHANGELOG，不得声称 P99 达标。
