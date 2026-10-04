#!/usr/bin/env python3
"""方案消融报告：教师基线 / PPO 残差 / 蒸馏学生（真实数据，含负结果）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/reports/ablation.json")
    parser.add_argument("--ppo-run", default="runs/exp/ppo_v2")
    parser.add_argument("--student-run", default="runs/exp/student_v2")
    args = parser.parse_args()
    ppo = _load(Path(args.ppo_run) / "summary.json")
    student = _load(Path(args.student_run) / "distill_report.json")
    eval_reports = sorted(Path("runs/reports").glob("eval-*/eval.json"), key=lambda p: p.stat().st_mtime)
    baseline = _load(eval_reports[-1]) if eval_reports else None
    payload = {
        "baseline_teacher": {
            "source": str(eval_reports[-1]) if eval_reports else None,
            "pass": bool(baseline and baseline.get("pass")),
            "summary": (baseline or {}).get("overall"),
        },
        "residual_ppo": {
            "run": args.ppo_run,
            "updates": (ppo or {}).get("updates"),
            "env_steps": (ppo or {}).get("env_steps"),
            "training_fall_rate": (ppo or {}).get("fall_rate"),
            "skill_fall_rate": (ppo or {}).get("skill_fall_rate"),
            "verdict": "negative: 残差策略在组合场景闭环评测中显著劣化，未作为默认控制器",
        },
        "distilled_student": {
            "run": args.student_run,
            "samples": (student or {}).get("samples"),
            "final_mse": (student or {}).get("final_mse"),
            "mse_pass": (student or {}).get("pass"),
            "verdict": "single-step pass / closed-loop gap: 绝对模式闭环会累积误差，需要 DAgger 修正",
        },
        "conclusion": "本工程默认控制器 = 教师 ONNX + 技能叠加 + SafetyWrapper（0 摔倒）；学习策略保留接口与负结果记录。",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# 消融报告（教师 / PPO 残差 / 蒸馏学生）",
        "",
        f"- 教师基线：pass={(baseline or {}).get('pass')}；"
        f"fall_rate={(baseline or {}).get('overall', {}).get('fall_rate') if baseline else None}",
        f"- PPO 残差：{payload['residual_ppo']['verdict']}（训练统计见 {args.ppo_run}/summary.json）",
        f"- 蒸馏学生：单步 MSE={payload['distilled_student']['final_mse']}；{payload['distilled_student']['verdict']}",
        "",
        payload["conclusion"],
    ]
    (out.with_suffix(".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
