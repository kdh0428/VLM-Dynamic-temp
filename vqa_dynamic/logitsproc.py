"""Dynamic temperature logits processor for vLLM."""

from __future__ import annotations

from typing import List

import torch
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor, RequestLogitsProcessor


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class DynamicTempPerReqLogitsProcessor:
    """Request-level logits processor for dynamic temperature."""

    def __init__(
        self,
        t_min: float,
        t_max: float,
        mode: str = "linear",
        max_steps: int = 128,
        target_entropy: float = 5.0,
        entropy_gain: float = 0.25,
    ) -> None:
        self.t_min = t_min
        self.t_max = t_max
        self.mode = mode
        self.max_steps = max_steps
        self.target_entropy = target_entropy
        self.entropy_gain = entropy_gain

    def __call__(self, output_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        step = len(output_ids) + 1

        if self.mode == "linear":
            frac = min(step / max(self.max_steps, 1), 1.0)
            temperature = self.t_max + (self.t_min - self.t_max) * frac
        elif self.mode == "entropy":
            probs = torch.softmax(logits, dim=-1)
            ent = -(probs * (probs + 1e-20).log()).sum().item()
            delta = (self.target_entropy - ent) * self.entropy_gain
            temperature = _clamp(1.0 + delta, self.t_min, self.t_max)
        else:
            temperature = 1.0

        temperature = max(temperature, 1e-4)
        logits /= temperature
        return logits


class DynamicTempLogitsProcessor(AdapterLogitsProcessor):
    """Adapter for vLLM logits processor API."""

    @classmethod
    def validate_params(cls, params):
        extra = params.extra_args or {}
        for key in ("t_min", "t_max", "target_entropy", "entropy_gain", "max_steps"):
            if key in extra and not isinstance(extra[key], (int, float)):
                raise ValueError(f"{key} must be a number, got {type(extra[key])}")

    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
        super().__init__(vllm_config, device, is_pin_memory)

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params) -> RequestLogitsProcessor | None:
        extra = params.extra_args or {}
        if not extra.get("dyn_temp", True):
            return None
        return DynamicTempPerReqLogitsProcessor(
            t_min=float(extra.get("t_min", 0.2)),
            t_max=float(extra.get("t_max", 1.2)),
            mode=str(extra.get("mode", "entropy")),
            max_steps=int(extra.get("max_steps", params.max_tokens or 128)),
            target_entropy=float(extra.get("target_entropy", 5.0)),
            entropy_gain=float(extra.get("entropy_gain", 0.25)),
        )
