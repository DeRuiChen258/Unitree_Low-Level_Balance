"""运行时决策策略：阈值 + 回退 + 升级 + 抖动抑制（第 8.3 节）。"""

from __future__ import annotations

import hashlib
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from cb_common.config import Config, load_config
from cb_common.errors import DecisionError
from cb_common.logging import JsonlLogger
from cb_common.types import DecisionOutput, RobotState, SafetyLevel, SkillCommand, TaskContext
from cb_features.text_state import build_text_state, safety_level_from_state

from .adapter import LayAAdapter
from .calibration import TemperatureScaler
from .mapping import map_answers
from .questions import QuestionSet, load_questions


@dataclass
class DecisionPolicy:
    """LayA + Tier 0 规则的运行时策略（决策层不进入硬实时回路）。"""

    model_config_path: str = "configs/laya/model.yaml"
    questions_path: str = "configs/laya/questions.json"
    calibration_path: str = "configs/laya/calibration.yaml"
    adapter: LayAAdapter | None = None
    logger: JsonlLogger | None = None
    questions: QuestionSet = field(init=False)
    calibration_cfg: Config = field(init=False)
    scaler: TemperatureScaler = field(init=False)
    consecutive_failures: int = field(default=0, init=False)
    last_action: str = field(default="continue_current", init=False)
    flips: deque = field(default_factory=lambda: deque(maxlen=16), init=False)
    last_valid: DecisionOutput | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.questions = load_questions(self.questions_path)
        self.calibration_cfg = load_config(self.calibration_path)
        self.scaler = TemperatureScaler.load(self.calibration_path) if False else TemperatureScaler()
        self.adapter = self.adapter or LayAAdapter(self.model_config_path)

    def decide(self, state: RobotState, ctx: TaskContext, *, command: SkillCommand | None = None) -> DecisionOutput:
        """执行一次决策；超时/异常 fail-closed 到 hold_current_skill + 人工复核。"""
        text_state = build_text_state(state, command, ctx)
        state_hash = hashlib.sha256(text_state.encode()).hexdigest()[:16]
        rule_level = safety_level_from_state(state)
        rule_veto = rule_level >= SafetyLevel.ABORT
        start = time.time()
        try:
            response = self.adapter.system_one(
                text_state,
                self.questions.for_agent(),
                timeout_ms=float(self.calibration_cfg.get("timeout_ms", 500)),
            )
            answers = dict(response.get("answers", {}))
            latency = (time.time() - start) * 1000.0
            output = map_answers(
                answers,
                state=state,
                candidates=self.questions.candidates(),
                confidence_threshold=float(self.calibration_cfg.get("confidence_threshold", 0.6)),
                risk_threshold=float(self.calibration_cfg.get("risk_threshold", 0.35)),
                fallback_policy=dict(self.calibration_cfg.get("fallback_policy", {})),
                rule_veto=rule_veto,
                latency_ms=latency,
                checkpoint=self.adapter.checkpoint,
                feature_version=str(self.calibration_cfg.get("state_template", "1.0")),
                state_hash=state_hash,
            )
            self.consecutive_failures = 0
        except DecisionError as exc:
            self.consecutive_failures += 1
            fallback = str(dict(self.calibration_cfg.get("fallback_policy", {})).get("model_timeout", "hold_current_skill"))
            output = DecisionOutput(
                action_id=fallback if not rule_veto else "safe_stop",
                primitive="safe_stop" if rule_veto else "continue_current",
                confidence=0.0,
                risk=1.0 if rule_veto else 0.5,
                need_human_review=True,
                fallback_action=fallback,
                probabilities={},
                reason_code="model_timeout" if "timeout" in str(exc).lower() else "model_error",
                heads={},
                meta={"error": str(exc), "state_hash": state_hash, "feature_version": "1.0"},
            )
        if self.logger is not None:
            self.logger.log(
                "decision",
                action_id=output.action_id,
                confidence=output.confidence,
                risk=output.risk,
                reason_code=output.reason_code,
                need_human_review=output.need_human_review,
                latency_ms=output.meta.get("latency_ms", 0.0),
                state_hash=state_hash,
            )
        self._track_flip(output.action_id)
        self.last_valid = output
        return output

    def note_failure(self) -> None:
        """记录技能失败并在阈值后升级。"""
        self.consecutive_failures += 1

    def escalation_needed(self) -> bool:
        """连续失败达到阈值 → 请求人工接管。"""
        threshold = int(dict(self.calibration_cfg.get("escalation_policy", {})).get("max_consecutive_failures", 3))
        return self.consecutive_failures >= threshold

    def to_skill_command(self, output: DecisionOutput, *, params: Mapping[str, Any] | None = None) -> SkillCommand:
        """把 DecisionOutput 映射为调度器 SkillCommand（不直接控制关节）。"""
        action = output.action_id
        mapping = {
            "continue_current": self.last_action,
            "stand": "stand",
            "walk": "walk",
            "run": "run",
            "run_jump": "jump",
            "wave_while_run": "wave",
            "turn_left": "turn",
            "recover": "recover",
            "safe_stop": "safe_stop",
        }
        skill = mapping.get(action, "stand")
        return SkillCommand(
            skill=skill,
            params=dict(params or {}),
            reason_code=output.reason_code,
            source="laya" if output.meta.get("model") else "rule",
            timestamp=0.0,
        )

    def _track_flip(self, action: str) -> None:
        window = float(dict(self.calibration_cfg.get("escalation_policy", {})).get("flip_window_s", 1.0))
        max_flips = int(dict(self.calibration_cfg.get("escalation_policy", {})).get("max_flips", 3))
        self.flips.append((time.time(), action))
        recent = [item for item in self.flips if time.time() - item[0] <= window]
        unique = {item[1] for item in recent}
        if recent and len(recent) >= max_flips and len(unique) >= 2:
            self.consecutive_failures += 1
        self.last_action = action
