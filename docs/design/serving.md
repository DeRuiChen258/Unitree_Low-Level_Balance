# 服务化（第 9 节）

## 1. 接口

| 方法 | 路径 | 语义 |
| --- | --- | --- |
| POST | `/v1/decision` | 单条：`{state, questions?}` → `{answers, meta}` |
| POST | `/v1/decision:batch` | 批量：`{items:[...]}` → `{results, meta}` |
| GET | `/v1/health` | `{status, model_loaded, backend}` |
| GET | `/v1/version` | 模型/checkpoint/feature/questions 版本 |

请求校验失败返回 422；后端不可用 503；推理超时 504，**不返回默认动作**
（由调用方 fail-closed 处理）。响应 meta 带 `latency_ms/backend/questions_hash/state_hash`。

## 2. vLLM 偏差（第 9.2 节）

`总目标.md` 指定 vLLM；但 vLLM 面向自回归生成的调度/采样，LayA 是编码器分类模型，
直接套用会造成不必要复杂度与性能损失。本工程：

1. 默认实现 FastAPI + LayA（可选 ONNX Runtime）或本地校准后端；
2. 保留 `VllmAdapter` 接口，仅当后续接入生成式动作模型/统一 LLM 网关时启用；
3. 上层 `DecisionClient` 抽象不感知后端。

## 3. 性能目标与实测

- 目标：P50 ≤ 40 ms、P99 ≤ 150 ms（本机 GPU）；
- 决策层 5 Hz（`configs/system.yaml`），控制回路不等决策；
- 模型不可用时显式降级到本地规则后端并记录 `fallback_reason`。

## 4. 测试

`tests/test_serving_api.py` 覆盖 health/version/单条/批量/422/危险状态 veto。
