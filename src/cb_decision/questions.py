"""四个决策头 questions 定义与版本管理（第 5.6/8.2 节）。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cb_common.errors import DecisionError

REQUIRED_HEADS = ("motion_primitive_head", "safety_veto_head", "recovery_head", "escalation_head")


@dataclass(frozen=True)
class QuestionSet:
    """questions 集合（版本 + 四头定义）。"""

    version: str
    questions: Mapping[str, Any]

    def hash(self) -> str:
        """questions 版本 hash（写入决策记录）。"""
        payload = json.dumps(self.questions, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(f"{self.version}:{payload}".encode()).hexdigest()[:16]

    def for_agent(self) -> dict[str, Any]:
        """返回 RLAgent.system_one 需要的 Jev 形状（4 头）。"""
        return {key: dict(value) for key, value in self.questions.items()}

    def candidates(self) -> list[str]:
        """motion 头的候选动作集合（不允许模型自由生成）。"""
        criteria = self.questions["motion_primitive_head"].get("criteria", {})
        return list(criteria.keys())


def load_questions(path: str | Path) -> QuestionSet:
    """加载并校验四头 questions。"""
    file = Path(path)
    if not file.is_file():
        raise DecisionError("questions file not found", path=str(file))
    data = json.loads(file.read_text(encoding="utf-8"))
    questions = data.get("questions", {})
    missing = [head for head in REQUIRED_HEADS if head not in questions]
    if missing:
        raise DecisionError("questions missing required heads", missing=missing)
    for head in REQUIRED_HEADS:
        item = questions[head]
        if item.get("type") not in ("choice", "score", "noul"):
            raise DecisionError("invalid question type", head=head, type=item.get("type"))
        if "instructions" not in item:
            raise DecisionError("question missing instructions", head=head)
    return QuestionSet(version=str(data.get("questions_version", "1.0")), questions=questions)
