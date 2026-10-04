# memory

- `Action/G1_Waving/common/balance_controller.py` 已定义 BalanceController 协议
  （PASS_THROUGH / LOW_LEVEL_SIM_ONLY），是真机安全与仿真补偿的天然接入点。
- `Action/G1_run/deploy/g1_deploy/config/g1_amp.yaml`：control_dt=0.02、sim_dt=0.005、
  decimation=4、history_length=4，是 50 Hz 控制回路的现成事实标准。
- `legged_rl_lab/GMR` 支持 `smplx_to_robot.py` / `bvh_to_robot.py`，且自带
  `assets/unitree_g1/g1_mocap_29dof.xml`，HumanML3D → G1 重定向可直接落地。
- HumanML3D_272d：272 维、30 FPS、train/val/test = 23384/1337/4041，Mean/Std 各 272 维。
- 官方 MuJoCo 窗口入口为 `/home/violet/Workspace/Code/Embedded_code/unitree_workspace/scripts/mujoco_sim.py`。
