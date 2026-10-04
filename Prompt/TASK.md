# TASK: G1 小脑平衡 + 多动作自适应组合系统（完整实现提示词）

> 创建时间：2026-10-04
> Workflow：unitree（模式 A 全量设计 → 提示词交付）
> 技能：unitree_g1_developer_guide → mujoco_simulation / python_sdk2 → task-guard → chinese-response

## Objective

基于 `Prompt/总目标.md`，把「LayA 结构化决策 + rl_infra 训练评测 + 小脑平衡策略 +
多动作组合（跑跳 / 挥手 / 转弯）」整合为一份可直接交给 AI 智能体执行的完整实现提示词；
提示词必须覆盖数据合成 → 特征工程 → 训练 → 组合调度 → 决策/服务 → 安全部署，
并详细到目标工程每个文件的作用。

## Inputs（已核验）

| 资产 | 路径 | 状态 |
| --- | --- | --- |
| 总目标 | `Prompt/总目标.md` | 483 行，三层原型（决策层 / rl_infra / vLLM） |
| 决策模型 | `/home/violet/Workspace/Model/laya` | LayA System-1 非自回归结构化决策模型（3 个 checkpoint + RL Agent 接口） |
| 动作工程 | `Action/{G1_Walk,G1_run,G1_Waving,legged_rl_lab}` | 走 / 跑 ONNX、挥手状态机、IsaacLab+rsl_rl 训练栈、GMR 重定向 |
| 动作数据 | `/home/violet/Workspace/Data/Embodied_AI/Unitree_G1` | HumanML3D_272d（21G）/ HumanML3D / KIT-ML / AMASS（空） |
| 数据增强参考 | `Prompt/data_enhance/*.pdf` | NOCS(1901.02970) + ANCSH(1912.11913)，方法论迁移并标注领域 gap |
| 本机环境 | conda `unitree_rt` | Python 3.12.9 / torch 2.14.0+cu130 / CUDA True / mujoco 3.13.0 |

## Deliverables

- `小脑平衡多动作自适应系统_完整实现提示词.md`（主交付物，分章节落盘）
- `TASK.md` 与本目录 `.agent/{state,decisions,memory,failures}.md`

## Constraints

- 只新增文档，不修改 `Action/`、数据集与 LayA 模型文件。
- 无真机：硬件结论只能标注编译级 / 启动级 / 接口级。
- 提示词中所有路径、模型事实必须来自本次勘察，不得凭记忆编造。

## Verification Criteria

- [x] 输入资产全部实读核验（模型卡、数据集结构、Action 四条工程链、PDF 主题）
- [x] 主提示词覆盖：数据合成 → 特征工程 → 小脑训练 → 组合调度 → LayA 决策 → 服务化 → 安全
- [x] 目标工程目录树逐文件给出职责 / 输入 / 输出 / 验收点
- [x] 与 `总目标.md` 边界一致：不直接控制电机力矩、不假设大模型做硬实时
- [x] 已标注技术偏差（vLLM 与编码器分类模型、data_enhance 论文领域、AMASS 为空）
- [x] 主文档自检：标题结构完整、代码块闭合（82 个围栏成对）、路径引用与实测一致

## Known Limitations

- 本机无真机；所有真机步骤只写流程与门禁，未执行。
- 23/29 DOF 关节数、G1 固件版本未在真机核验，提示词中以「先勘察再定参」处理。
