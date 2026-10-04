# decisions

1. 决策模型定位：LayA 是非自回归 System-1 结构化决策模型（输入文本/JSON，输出 choice/score/noul
   校准概率），因此系统提示词把它放在「决策层」，不承担动作生成与硬实时控制。
2. 服务层偏差记录：`总目标.md` 要求 vLLM；但 vLLM 面向自回归 LLM 生成，LayA 是编码器分类模型。
   提示词保留统一服务接口，默认实现改为 `laya[serve]` / ONNX Runtime / FastAPI，并把 vLLM 作为
   「仅当后续接入生成式动作模型时」的可选适配器。
3. data_enhance 两篇论文（NOCS / ANCSH）属于物体位姿领域，只迁移「混合现实合成数据」与
   「规范空间 + 类别级泛化」方法论，不做领域等同声明。
4. 小脑平衡策略优先复用 `legged_rl_lab`（IsaacLab 2.3 + rsl_rl，含 G1 velocity/AMP/parkour）与
   `G1_Walk`（MuJoCo Warp GRPO/PPO）两条已有训练栈，不新造训练框架。
