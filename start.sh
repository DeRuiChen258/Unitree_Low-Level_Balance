#!/usr/bin/env bash
# G1 平衡系统 + 拾取系统 全流程演示
#   行走 → 加速跑步 → 急停 → 绕多个柱子 → 停下来抱重物（16.67 kg ≈ 本体一半）
#
# 用法：
#   ./start.sh                 # 依次弹出 5 个 MuJoCo 窗口（默认）
#   RENDER=video ./start.sh    # 不弹窗，输出 5 段 MP4
#   RENDER=none  ./start.sh    # 只跑指标（最快，用于回归）
#   STEPS=0 ./start.sh         # 各段使用默认步数（0 = 用脚本内默认值）
set -euo pipefail

cd "$(dirname "$0")"

PY=${PY:-python}
RENDER=${RENDER:-viewer}
RECORD=${RECORD:-1}                       # 1 = 同时录制 MP4（viewer 模式下窗口照常显示）
SEED=${SEED:-2}
CARRY_MASS=${CARRY_MASS:-16.67}          # 抱重物质量 kg（机器人本体 33.34 kg）
PILLARS=${PILLARS:-"4.5,0.45;8.0,-0.45;11.5,0.45"}

banner() {
  printf '\n\033[1;36m================================================================\033[0m\n'
  printf '\033[1;36m  %s\033[0m\n' "$1"
  printf '\033[1;36m================================================================\033[0m\n'
}

# 平衡系统演示（内部调用）：viewer 模式下本机 Wayland/EGL 在窗口退出阶段会 segfault
# （exit 139，发生在结果落盘之后；`--render video|none` 不受影响），
# 因此这里只对平衡演示放宽退出码，其它错误照常报出。
run_balance() {
  local scenario="$1"
  shift
  local extra=()
  if [ "$RECORD" = "1" ]; then
    extra+=(--record)
  fi
  set +e
  $PY scripts/demo_balance_extra.py --scenario "$scenario" --render "$RENDER" --seed "$SEED" "${extra[@]}" "$@"
  local code=$?
  set -e
  if [ "$code" -eq 139 ]; then
    printf '\033[1;33m[note]\033[0m %s：MuJoCo 窗口退出阶段触发本机 Wayland/EGL 段错误（结果已落盘）。\n' "$scenario"
  elif [ "$code" -ne 0 ]; then
    printf '\033[1;31m[error]\033[0m %s 退出码 %s\n' "$scenario" "$code" >&2
    exit "$code"
  fi
}

banner "1/5  行走（walk 0.8 m/s）"
run_balance walk --steps 600

banner "2/5  加速跑步（0.6 → 2.0 m/s）"
run_balance walk_run --steps 700

banner "3/5  急停（2.0 m/s → 停止）"
run_balance run_stop --steps 600

banner "4/5  绕多个柱子（3 根，左右交替绕行）"
run_balance pillar_slalom --steps 1100 --pillars "$PILLARS"

banner "5/5  停下来抱重物（双手抱两侧，${CARRY_MASS} kg ≈ 本体一半）"
if [ "$RECORD" = "1" ]; then
  $PY scripts/run_pickup.py --scenario front --baseline C --render "$RENDER" --record --seed 0
else
  $PY scripts/run_pickup.py --scenario front --baseline C --render "$RENDER" --seed 0
fi

# 把 5 段 MP4 汇总到 recordings/ 便于留档（并尝试合并为一条完整视频）
banner "录像归档"
$PY - <<'PY'
import glob
import os
import shutil
import subprocess
from pathlib import Path

rec_dir = Path("recordings")
rec_dir.mkdir(exist_ok=True)
clips: list[tuple[str, str]] = []

for scenario, pattern in (
    ("1_walk", "runs/demo/balance_extra-*/demo.mp4"),
    ("2_walk_run", "runs/demo/balance_extra-*/demo.mp4"),
    ("3_run_stop", "runs/demo/balance_extra-*/demo.mp4"),
    ("4_pillar_slalom", "runs/demo/balance_extra-*/demo.mp4"),
):
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not candidates:
        continue
    # 用 summary 里的 scenario 字段匹配对应录像，避免张冠李戴
    for path in reversed(candidates):
        summary = Path(path).with_name("summary.json")
        if not summary.exists():
            continue
        import json

        if json.loads(summary.read_text(encoding="utf-8"))["scenario"] == scenario.split("_", 1)[1]:
            clips.append((scenario, path))
            break

pickup = sorted(glob.glob("experiments/*pickup_front_baselineC/video/pickup.mp4"), key=os.path.getmtime)
if pickup:
    clips.append(("5_pickup_heavy", pickup[-1]))

ordered: list[Path] = []
for name, src in clips:
    dst = rec_dir / f"{name}.mp4"
    shutil.copyfile(src, dst)
    ordered.append(dst)
    print(f"  {dst}")

if ordered and shutil.which("ffmpeg"):
    list_file = rec_dir / "clips.txt"
    list_file.write_text("".join(f"file '{p.resolve()}'\n" for p in ordered), encoding="utf-8")
    merged = rec_dir / "demo_full.mp4"
    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(merged)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"  合并完整录像：{merged}")
    else:
        print(f"  [warn] ffmpeg 合并失败：{result.stderr.strip().splitlines()[-1] if result.stderr else 'unknown'}")
PY

banner "全部演示完成：结果汇总（runs/demo/balance_extra-*、experiments/*pickup_*、recordings/）"
$PY - <<'PY'
import glob
import json
import os
from pathlib import Path

rows: dict[str, str] = {}
order = ["walk", "walk_run", "run_stop", "pillar_slalom"]
for summary in sorted(glob.glob("runs/demo/balance_extra-*/summary.json"), key=os.path.getmtime):
    data = json.loads(Path(summary).read_text(encoding="utf-8"))
    metrics = data["metrics"]
    rows[data["scenario"]] = (
            "steps={steps:<4d} 安全拦截={blocked!s:<5s} roll峰值={roll:.2f}° pitch峰值={pitch:.2f}° "
            "裕度最小={margin:+.3f} 摔倒={fall:.0f}".format(
                steps=data["steps"],
                blocked=data["blocked_by_safety"],
                roll=metrics["roll_peak"],
                pitch=metrics["pitch_peak"],
                margin=metrics["support_margin_min"],
                fall=metrics["fall"],
            )
    )
for metrics_path in sorted(glob.glob("experiments/*pickup_front_baselineC/metrics.json"), key=os.path.getmtime)[-1:]:
    data = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    rows["pickup(抱重物)"] = (
            "success={success!s} 抓取={grasped!s} 用时={t:.2f}s 脚步={steps} 摔倒={fall!s}".format(
                success=data.get("success"),
                grasped=data.get("grasped"),
                t=float(data.get("time_s", 0.0)),
                steps=int(data.get("steps", 0)),
                fall=data.get("fall"),
            )
    )
print()
for name in order + ["pickup(抱重物)"]:
    if name in rows:
        print(f"  {name:<14s} {rows[name]}")
print()
PY
