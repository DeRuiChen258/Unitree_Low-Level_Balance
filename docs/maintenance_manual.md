# 维护手册

## 日常检查

1. `pytest -q` 与 `python scripts/verify.py --all` 全绿；
2. 数据 manifest 的 sha256 无漂移（`scripts/data_quality.py`）；
3. 训练/评测报告四类版本号齐全；
4. `runs/` 体积增长时归档到外部存储，仓库只保留报告摘要。

## 故障排查

| 症状 | 优先检查 |
| --- | --- |
| 教师策略不加载 | ONNX 路径、`assets/` 相对路径、onnxruntime 版本 |
| 仿真立即摔倒 | 初始高度/姿态、PD 增益、质量随机化是否累积 |
| 观测 hash 不匹配 | feature_version / ObsSpec 变更后未重新导出 |
| 决策服务 503 | LayA checkpoint/依赖（transformers/safetensors） |
| 决策延迟高 | GPU 争抢（训练与推理串行化）、batch 窗口、CPU fallback |
| 安全误触发 | 阈值是否被放宽、支撑多边形/接触估计、单脚超时 |

## 升级流程

1. 更新配置/代码 → 2. 跑单测 → 3. 跑三组合评测 → 4. 更新 CHANGELOG/文档 → 5. 打 tag。
