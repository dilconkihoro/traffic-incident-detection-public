"""Rule-based incident checks (docs/02 section 5); pure functions of trajectory history."""

from __future__ import annotations

from abc import ABC, abstractmethod

from trafficwatch.config import BehaviourConfig
from trafficwatch.types import Incident, SceneContext


class Rule(ABC):
    """A rule sees the trajectory history up to the current frame and returns new incidents."""

    def __init__(self, cfg: BehaviourConfig, n_init: int) -> None:
        self.cfg = cfg
        self.n_init = n_init

    @abstractmethod
    def evaluate(self, ctx: SceneContext) -> list[Incident]: ...


class AbruptStopRule(Rule):
    def evaluate(self, ctx: SceneContext) -> list[Incident]:
        raise NotImplementedError("Phase 2")


class LaneObstructionRule(Rule):
    def evaluate(self, ctx: SceneContext) -> list[Incident]:
        raise NotImplementedError("Phase 2")


class CollisionRule(Rule):
    def evaluate(self, ctx: SceneContext) -> list[Incident]:
        raise NotImplementedError("Phase 2")
