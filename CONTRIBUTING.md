# 贡献指南

## 分支与提交

- 分支：`main`（可发布）/ `feature/*` / `fix/*` / `experiment/*`；
- 提交信息遵循 Conventional Commits（`feat:`, `fix:`, `docs:`, `test:`, `exp:`）；
- 安全层（`src/cb_safety`）、决策映射、数据 schema 变更需要 2 名 reviewer；
- 任何实验结论必须附带 `dataset_version/feature_version/model_version/questions_version`
  与可重跑命令，否则不允许合入报告目录。

## 开发门禁

```bash
pytest -q
python scripts/verify.py --all
ruff check src scripts tests
mypy src
```

## 变更类型

| 变更 | 必须动作 |
| --- | --- |
| 观测 spec 字段/顺序 | 升 `feature_version` + 重新训练/导出 + 回归 |
| questions/模板 | 升 `questions_version` + 重新校准 |
| 安全阈值 | 更新 `configs/safety/limits.yaml` + 故障注入测试 |
| 数据集 | 更新 manifest + 记录 sha256 + 重新训练 |

## 数据与许可

HumanML3D/AMASS 为学术许可，商用前必须重新评估；Model/LayA 按实际发布版本核对；
不得把 `Action/`、数据集、模型 checkpoint 提交进普通 Git。
