"""Oracle-owned persistence additions; privileged SAC updates remain unchanged."""

from __future__ import annotations

from typing import Any

from unilab.algos.torch.hora.sac_learner import HoraSACLearner


class MotionQualityOracleLearner(HoraSACLearner):
    def __init__(self, *, checkpoint_contract: dict[str, Any], **kwargs: Any) -> None:
        self.checkpoint_contract = checkpoint_contract
        super().__init__(**kwargs)

    def get_state_dict(self) -> dict[str, Any]:
        return {**super().get_state_dict(),
                "amp_scaler": self.scaler.state_dict() if self.scaler is not None else None}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        scaler_state = state_dict["amp_scaler"]
        if self.scaler is not None and scaler_state is None:
            raise ValueError("Cannot restore fp16 training without saved AMP scaler state")
        super().load_state_dict(state_dict)
        if self.scaler is not None:
            self.scaler.load_state_dict(scaler_state)
