"""Request-scoped Transformer residual reuse for joint video/action inference.

This cache operates before the output heads. It must not be combined with the
older velocity cache, which bypasses the complete model forward and its heads.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from threading import Lock

import torch
from torch import Tensor


class TransformerResidualCache:
    """Reuse both MoT input-to-output residuals on a fixed update schedule.

    The serving engine owns the generation context and validates sampler/CFG
    compatibility. Every actual Euler update must dispatch through ``run`` and
    then call ``record_head_evaluation`` after evaluating both current heads.
    No tensors survive the generation context, including an exceptional exit.
    """

    def __init__(self, refresh_steps: Iterable[int] = (0, 3, 6, 9), total_steps: int = 10):
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 1:
            raise ValueError("total_steps must be a positive integer")
        steps = tuple(refresh_steps)
        if not steps or any(isinstance(x, bool) or not isinstance(x, int) for x in steps):
            raise ValueError("refresh_steps must contain integer update indices")
        if tuple(sorted(set(steps))) != steps or steps[0] != 0 or steps[-1] >= total_steps:
            raise ValueError("refresh_steps must be sorted, unique, start at zero and be below total_steps")
        self.refresh_steps = steps
        self.total_steps = total_steps
        self.active = False
        self._generation_lock = Lock()
        self._refresh_set = frozenset(steps)
        self._video_residual: Tensor | None = None
        self._action_residual: Tensor | None = None
        self._signature: tuple | None = None
        self._step = 0
        self._full_evals = 0
        self._reused_evals = 0
        self._head_evals = 0

    @property
    def next_is_refresh(self) -> bool:
        """Whether the next dispatch requires a full coupled Transformer."""
        if not self.active or self._step >= self.total_steps:
            raise RuntimeError("No pending residual-cache dispatch")
        return self._step in self._refresh_set

    @property
    def stats(self) -> dict:
        """Return bounded, tensor-free statistics for the current/last request."""
        return {
            "refresh_steps": list(self.refresh_steps),
            "total_steps": self.total_steps,
            "steps_completed": self._step,
            "full_evals": self._full_evals,
            "reused_evals": self._reused_evals,
            "head_evals": self._head_evals,
            "active": self.active,
        }

    def _clear_tensors(self) -> None:
        self._video_residual = None
        self._action_residual = None
        self._signature = None

    @contextmanager
    def generation(self) -> Iterator["TransformerResidualCache"]:
        if not self._generation_lock.acquire(blocking=False):
            raise RuntimeError("TransformerResidualCache does not support concurrent/reentrant generations")
        self._clear_tensors()
        self._step = self._full_evals = self._reused_evals = self._head_evals = 0
        self.active = True
        try:
            yield self
            if self._step != self.total_steps or self._head_evals != self.total_steps:
                raise RuntimeError(
                    "Incomplete residual-cache generation: "
                    f"dispatches={self._step}, heads={self._head_evals}, expected={self.total_steps}"
                )
        finally:
            self.active = False
            self._clear_tensors()
            self._generation_lock.release()

    @staticmethod
    def _tensor_signature(value: Tensor) -> tuple:
        return (tuple(value.shape), value.dtype, value.device)

    def run(
        self,
        video_input: Tensor,
        action_input: Tensor,
        full_forward: Callable[[], tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor]:
        """Execute the MoT or add a cached residual to the *current* inputs.

        Refresh steps return the original Transformer outputs without a
        subtract/add round trip. FP32 input snapshots and residual allocations
        are independent of input mutations and CUDA Graph output buffers.
        """
        if not self.active:
            raise RuntimeError("Residual-cache run requires an active generation context")
        if torch.is_grad_enabled():
            raise RuntimeError("Transformer residual reuse only supports no-grad inference")
        if self._step >= self.total_steps:
            raise RuntimeError("More Transformer dispatches than the configured sampler updates")
        if self._head_evals != self._step:
            raise RuntimeError("The previous update did not evaluate its current output heads")
        if not video_input.is_floating_point() or not action_input.is_floating_point():
            raise TypeError("Transformer residual inputs must be floating-point tensors")
        signature = (self._tensor_signature(video_input), self._tensor_signature(action_input))
        if self._signature is None:
            self._signature = signature
        elif signature != self._signature:
            raise ValueError("Residual-cache shape, dtype or device changed within one generation")

        if self._step in self._refresh_set:
            # copy=True also detaches FP32 inputs from storage that a model or
            # captured graph might mutate. Do not retain a graph output alias.
            video_before = video_input.detach().to(dtype=torch.float32, copy=True)
            action_before = action_input.detach().to(dtype=torch.float32, copy=True)
            video_output, action_output = full_forward()
            output_signature = (self._tensor_signature(video_output), self._tensor_signature(action_output))
            if output_signature != signature:
                raise ValueError("Transformer residual output shape/dtype/device must match its input")
            self._video_residual = video_output.detach().float() - video_before
            self._action_residual = action_output.detach().float() - action_before
            self._full_evals += 1
        else:
            if self._video_residual is None or self._action_residual is None:
                raise RuntimeError("Cannot reuse an uninitialized Transformer residual")
            video_output = (video_input.float() + self._video_residual).to(dtype=video_input.dtype)
            action_output = (action_input.float() + self._action_residual).to(dtype=action_input.dtype)
            self._reused_evals += 1
        self._step += 1
        return video_output, action_output

    def record_head_evaluation(self) -> None:
        """Count one pair of successfully evaluated video/action output heads."""
        if not self.active or self._head_evals + 1 != self._step:
            raise RuntimeError("Output-head evaluation does not match one Transformer dispatch")
        self._head_evals += 1
