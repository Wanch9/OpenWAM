"""Exact request-local clean-observation prefill for Wan joint prediction.

Only mutual video/action attention with first-frame-causal video visibility is
supported. The engine must keep weights and observation conditioning fixed
within a generation. Optional invariant checks verify the prepared prefix on
every complete Transformer evaluation; their cost belongs in request timing.
"""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from threading import Lock

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _attention(q, k, v, heads):
    b, sq, width = q.shape
    sk = k.shape[1]
    head_dim = width // heads
    result = F.scaled_dot_product_attention(
        q.reshape(b, sq, heads, head_dim).transpose(1, 2),
        k.reshape(b, sk, heads, head_dim).transpose(1, 2),
        v.reshape(b, sk, heads, head_dim).transpose(1, 2),
    )
    return result.transpose(1, 2).reshape(b, sq, width)


class ObservationPrefixCache:
    """Prefill the clean prefix once, then evaluate future/action rows only.

    ``run`` consumes prepared video/action states and returns the same complete
    state layout expected by their current output heads. This runner is invoked
    only on full Transformer evaluations when combined with residual reuse.
    """

    def __init__(self, architecture, compile_enabled=True, compile_mode="reduce-overhead", verify_invariants=True):
        self.architecture = architecture
        self.vb = architecture.video_backbone
        self.ab = architecture.action_backbone
        self.driver = architecture.mot_driver
        if self.driver is None:
            raise ValueError("Observation prefix caching requires an initialized MoT driver")
        self.compile_enabled = bool(compile_enabled)
        self.compile_mode = str(compile_mode)
        self.verify_invariants = bool(verify_invariants)
        self.active = False
        self._lock = Lock()
        self._compiled_prefill = None
        self._compiled_joint = None
        self._compile_attempted = False
        self._compile_failures = 0
        self._prefill_evals = self._joint_evals = self._invariant_checks = 0
        self._clear_request()
        self._validate_backend()

    @property
    def stats(self):
        return {
            "active": self.active,
            "prefill_evals": self._prefill_evals,
            "joint_evals": self._joint_evals,
            "invariant_checks": self._invariant_checks,
            "verify_invariants": self.verify_invariants,
            "compile_active": self._compiled_prefill is not None and self._compiled_joint is not None,
            "compiled_prefill_active": self._compiled_prefill is not None,
            "compiled_joint_active": self._compiled_joint is not None,
            "compile_failures": self._compile_failures,
        }

    def _clear_request(self):
        self._keys = self._values = self._prefix_final = None
        self._anchors = None
        self._shape_signature = None
        self._prefix_length = None

    @contextmanager
    def generation(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Observation prefix cache does not support concurrent/reentrant generations")
        self._clear_request()
        self._prefill_evals = self._joint_evals = self._invariant_checks = 0
        self.active = True
        try:
            yield self
            if self._prefill_evals != 1 or self._joint_evals < 1:
                raise RuntimeError(
                    "Observation prefix cache was enabled but its complete execution path did not run: "
                    f"prefills={self._prefill_evals}, joint_evals={self._joint_evals}"
                )
        finally:
            self.active = False
            self._clear_request()
            self._lock.release()

    def _validate_backend(self):
        from openwam.model.video_backbone.wan_backbone import WanBase

        if not isinstance(self.vb, WanBase) or not self.vb._is_ti2v:
            raise ValueError("Observation prefix caching currently supports Wan TI2V only")
        if self.driver.attention_mask_mode != "mutual" or self.vb.video_attention_mask_mode != "first_frame_causal":
            raise ValueError("Observation prefix caching requires mutual + first_frame_causal attention")
        if getattr(self.vb, "vace", None) is not None or getattr(self.vb.dit, "require_clip_embedding", False):
            raise ValueError("VACE and separate CLIP conditioning are unsupported by observation prefix caching")

    @staticmethod
    def _signature(tensor):
        return (tuple(tensor.shape), tensor.dtype, tensor.device)

    def _validate_state(self, vstate, astate):
        if not self.active:
            raise RuntimeError("Observation prefix run requires an active generation context")
        if torch.is_grad_enabled() or self.architecture.training or self.vb.training or self.ab.training:
            raise RuntimeError("Observation prefix caching requires no-grad eval-mode inference")
        self._validate_backend()
        x, a = vstate.hidden_states, astate.payload.x_action
        prefix = int(vstate.grid_height) * int(vstate.grid_width)
        if x.ndim != 3 or a.ndim != 3 or x.shape[0] != 1 or a.shape[0] != 1:
            raise ValueError("Observation prefix caching supports batch-one flat token states")
        if prefix <= 0 or int(vstate.grid_frames) < 2 or x.shape[1] != int(vstate.grid_frames) * prefix:
            raise ValueError("Invalid clean-prefix or future-video token geometry")
        if vstate.time_mod.ndim != 4 or vstate.time_mod.shape[:2] != x.shape[:2]:
            raise ValueError("Observation prefix caching requires per-token 4D time modulation")
        if vstate.rope_freqs.shape[0] != x.shape[1]:
            raise ValueError("RoPE positions do not match the complete video token sequence")
        if getattr(vstate, "vace_hints", None) is not None or vstate.extras.get("vace") is not None:
            raise ValueError("VACE hints cannot be split into an invariant observation prefix")
        if int(getattr(vstate, "prefix_kv_len", 0) or 0):
            raise ValueError("Additional prefix K/V is unsupported by observation prefix caching")
        if vstate.use_gradient_checkpointing or vstate.use_gradient_checkpointing_offload:
            raise ValueError("Gradient checkpointing is incompatible with observation prefix caching")
        signature = (
            self._signature(x), self._signature(a), self._signature(vstate.time_mod),
            self._signature(vstate.rope_freqs), prefix,
        )
        if self._shape_signature is not None and signature != self._shape_signature:
            raise ValueError("Observation prefix shape/dtype/device changed within a generation")
        self._shape_signature = signature
        self._prefix_length = prefix
        values = (
            x[:, :prefix], vstate.time_mod[:, :prefix], vstate.rope_freqs[:prefix],
            vstate.context, vstate.context_mask,
        )
        if self.verify_invariants:
            if self._anchors is None:
                self._anchors = tuple(value.detach().clone() if value is not None else None for value in values)
            else:
                for name, value, anchor in zip(
                    ("prefix embedding", "prefix time modulation", "prefix RoPE", "conditioning", "conditioning mask"),
                    values, self._anchors,
                ):
                    if (value is None) != (anchor is None) or (
                        value is not None and (self._signature(value) != self._signature(anchor) or not torch.equal(value, anchor))
                    ):
                        raise ValueError(f"Observation prefix invariant changed: {name}")
                self._invariant_checks += 1

    @staticmethod
    def _copy_action(astate):
        result = copy.copy(astate)
        result.payload = copy.copy(astate.payload)
        return result

    def _slice_video(self, vstate, start, stop):
        result = copy.copy(vstate)
        result.hidden_states = vstate.hidden_states[:, start:stop].contiguous()
        result.time_mod = vstate.time_mod[:, start:stop].contiguous()
        result.rope_freqs = vstate.rope_freqs[start:stop].contiguous()
        return result

    def _prefill_core(self, prefix_state):
        keys, values = [], []
        for layer in range(self.driver.num_layers):
            q, k, v, post = self.vb.pre_attn_at_layer_for_compile(layer, prefix_state)
            keys.append(k)
            values.append(v)
            attended = _attention(q, k, v, self.driver.num_heads)
            prefix_state = self.vb.post_attn_at_layer_for_compile(layer, prefix_state, attended, post)
        return torch.stack(keys), torch.stack(values), prefix_state.hidden_states

    def _joint_core(self, future_state, astate, keys, values):
        for layer in range(self.driver.num_layers):
            qv, kv, vv, vpost = self.vb.pre_attn_at_layer_for_compile(layer, future_state)
            qa, ka, va, apost = self.ab.pre_attn_at_layer_for_compile(layer, astate)
            n_video, n_action = qv.shape[1], qa.shape[1]
            attended = _attention(
                torch.cat((qv, qa), dim=1),
                torch.cat((keys[layer], kv, ka), dim=1),
                torch.cat((values[layer], vv, va), dim=1),
                self.driver.num_heads,
            )
            av, aa = attended.split((n_video, n_action), dim=1)
            future_state = self.vb.post_attn_at_layer_for_compile(layer, future_state, av.contiguous(), vpost)
            astate = self.ab.post_attn_at_layer_for_compile(layer, astate, aa.contiguous(), apost)
        return future_state.hidden_states, astate.payload.x_action

    def _ensure_compiled(self):
        if not self.compile_enabled or self._compile_attempted:
            return
        self._compile_attempted = True
        try:
            self._compiled_prefill = torch.compile(self._prefill_core, mode=self.compile_mode, fullgraph=True)
            self._compiled_joint = torch.compile(self._joint_core, mode=self.compile_mode, fullgraph=True)
        except Exception as exc:
            self._compiled_prefill = self._compiled_joint = None
            self._compile_failures += 1
            logger.warning("Observation prefix compilation setup failed; using eager: %s", exc)

    def _prefill(self, prefix_state):
        if self._compiled_prefill is not None:
            try:
                return self._compiled_prefill(copy.copy(prefix_state))
            except Exception as exc:
                self._compiled_prefill = None
                self._compile_failures += 1
                logger.warning("Observation prefill compilation failed; using eager: %s", exc)
        return self._prefill_core(copy.copy(prefix_state))

    def _joint(self, future_state, astate):
        if self._compiled_joint is not None:
            try:
                return self._compiled_joint(
                    copy.copy(future_state), self._copy_action(astate), self._keys, self._values
                )
            except Exception as exc:
                self._compiled_joint = None
                self._compile_failures += 1
                logger.warning("Cached joint compilation failed; using eager: %s", exc)
        return self._joint_core(copy.copy(future_state), self._copy_action(astate), self._keys, self._values)

    def run(self, vstate, astate):
        self._validate_state(vstate, astate)
        self._ensure_compiled()
        prefix = self._prefix_length
        if self._keys is None:
            prefix_state = self._slice_video(vstate, 0, prefix)
            keys, values, hidden = self._prefill(prefix_state)
            # Captured outputs must not alias graph-owned allocations that can
            # be replayed or released when the next graph step starts.
            self._keys = keys.detach().clone()
            self._values = values.detach().clone()
            self._prefix_final = hidden.detach().clone()
            self._prefill_evals += 1
        future_state = self._slice_video(vstate, prefix, None)
        future_hidden, action_hidden = self._joint(future_state, astate)
        vstate.hidden_states = torch.cat((self._prefix_final, future_hidden), dim=1)
        astate.payload.x_action = action_hidden
        self._joint_evals += 1
        return vstate, astate
