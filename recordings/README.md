# 演示录像（recordings/）

本目录保存 `./start.sh` 全流程演示的**抽帧核验图**；MP4 录像体积较大，未纳入版本库，
按下面命令可一键重新生成（本地实测：5 段合计约 30 MB）。

```bash
# 有窗口 + 同步录像（推荐，需要显示器）
./start.sh

# 无窗口、最快出片（约 2 分钟）
RENDER=none RECORD=1 ./start.sh
```

生成的录像（自动归档到本目录）：

| 文件 | 内容 |
| --- | --- |
| `1_walk.mp4` | 行走 0.8 m/s |
| `2_walk_run.mp4` | 加速：0.6 → 2.0 m/s |
| `3_run_stop.mp4` | 2.0 m/s 急停（含急停距离指标） |
| `4_pillar_slalom.mp4` | 绕 3 根柱子（左右交替） |
| `5_pickup_heavy.mp4` | 停下弯腰、双手抱两侧、搬起 16.67 kg（≈本体一半）重物 |
| `demo_full.mp4` | 上述 5 段用 ffmpeg 合并的完整记录 |

抽帧图（已入库）：

- `frame_sheet_slalom.png`：绕桩全过程抽帧（可见每根柱子都在预定侧绕过）
- `frame_sheet_pickup.png`：抱取重物抽帧（握持点在箱体侧面约 70% 高度）
