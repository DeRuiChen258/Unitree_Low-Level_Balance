"""消融开关（第 17 节，Ablation 1–10）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AblationConfig:
    """单个消融配置：关闭某个信号/机制或切换固定/动态阈值。"""

    name: str
    use_com: bool = True
    use_zmp: bool = True
    use_capture_point: bool = True
    use_footstep: bool = True
    use_decision_layer: bool = True
    use_confidence: bool = True
    fixed_threshold: bool = False
    hierarchical: bool = True
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON 友好输出。"""
        return dict(self.__dict__)


ABLATIONS: dict[str, AblationConfig] = {
    "1": AblationConfig("1", use_com=False, description="去掉 COM（用 base_pos 代替 CoM）"),
    "2": AblationConfig("2", use_zmp=False, description="去掉 ZMP（ZMP=COM 投影）"),
    "3": AblationConfig("3", use_capture_point=False, description="去掉 Capture Point"),
    "4": AblationConfig("4", use_footstep=False, description="去掉脚步调整"),
    "5": AblationConfig("5", use_decision_layer=False, description="去掉 JEV-like 决策（纯规则状态机）"),
    "6": AblationConfig("6", use_confidence=False, description="去掉置信度门控"),
    "7": AblationConfig("7", fixed_threshold=True, description="固定阈值（pitch 0.3）"),
    "8": AblationConfig("8", description="动态阈值（默认平衡感知）"),
    "9": AblationConfig("9", hierarchical=False, description="单动作轨迹（无多 primitive 组合）"),
    "10": AblationConfig("10", description="多 primitive 层级控制（完整系统）"),
}


def get_ablation(name: str) -> AblationConfig:
    """按编号/名称返回消融配置。"""
    key = str(name)
    if key not in ABLATIONS:
        raise KeyError(f"unknown ablation {name!r}; expected 1..10")
    return ABLATIONS[key]


def apply_ablation(controller: Any, ablation: AblationConfig | str) -> AblationConfig:
    """把消融配置应用到 PickupController（运行时开关）。"""
    config = ablation if isinstance(ablation, AblationConfig) else get_ablation(ablation)
    controller.monitor.use_com = config.use_com
    controller.monitor.use_zmp = config.use_zmp
    controller.monitor.use_capture_point = config.use_capture_point
    controller.manager.max_steps = 0 if not config.use_footstep else controller.manager.max_steps
    if not config.use_decision_layer:
        from .decision import RuleBasedDecision

        controller.manager.decision_model = RuleBasedDecision()
    if config.fixed_threshold:
        from .baselines import ThresholdStepDecision

        controller.manager.decision_model = ThresholdStepDecision()
    if not config.use_confidence:
        controller.manager.reach_margin_m = -1.0  # 关闭置信/裕度门控（不禁止 reach）
    if not config.hierarchical:
        controller.manager.max_steps = 0
    return config
