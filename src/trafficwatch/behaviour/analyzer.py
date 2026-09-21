from __future__ import annotations

from collections.abc import Sequence

from trafficwatch.config import PipelineConfig
from trafficwatch.types import Incident, Track


class BehaviourAnalyzer:
    """Runs the three rules once per frame over the confirmed tracks."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg

    def step(self, tracks: Sequence[Track], t_s: float) -> list[Incident]:
        raise NotImplementedError("Phase 2")
