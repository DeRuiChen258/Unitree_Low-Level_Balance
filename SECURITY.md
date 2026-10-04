# 安全与保密

## 人员与设备安全

- 最高优先级：人员安全 > 设备安全 > 平衡稳定 > 动作表现 > 开发速度；
- 真机动作必须通过 Gate 0–2 并完成 `docs/sim2real.md` 首测清单；
- 软件 E-Stop 独立通道 + 硬件急停双保险；急停后只能人工复位。

## 代码安全

- 运动命令唯一出口是 `SafetyWrapper`；`SDK2Adapter(allow_send=True)` 直接拒绝构造；
- 安全配置缺失/非法时 `SafetyError` fail-closed；
- 决策服务不可用时保持当前技能 → 降级 stand → 必要时 safe_stop。

## 数据与密钥

- 不在日志中写入密钥；`.gitignore` 排除 `runs/`、模型权重与大文件；
- 对外发布前用 `research-integrity` 流程检查结论、基线与选择性报告。

## 报告漏洞

在 issue 中标注 `security`，附最小复现、影响范围与建议修复；不要在公开渠道披露未修复漏洞。
