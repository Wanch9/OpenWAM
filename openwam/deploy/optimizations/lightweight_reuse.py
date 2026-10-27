"""Exact removal of unused preparation on Transformer-residual reuse steps.

This does not change the residual approximation. Current video/action inputs
are still embedded and both current output heads still run. The supported
TI2V branch skips only Transformer-only timestep modulation, context and RoPE
preparation, which no head consumes on a reuse step.
"""

from __future__ import annotations

from contextlib import contextmanager
from threading import Lock
from types import SimpleNamespace

import torch
from einops import rearrange

from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.video_backbone.wan import dit_forward as wan_dit_forward
from openwam.model.video_backbone.wan.models.dit import sinusoidal_embedding_1d
from openwam.model.video_backbone.wan_backbone import WanBase


class LightweightResidualReuse:
    """Opt-in, request-scoped lightweight execution for supported reuse steps.

    The owner binds this object as ``architecture._serving_lightweight_reuse``
    and enters ``generation`` alongside the existing residual-cache context.
    ``run`` returns None for an unsupported backend/input so the architecture
    can retain its original forward. No tensors are retained by this object.
    """

    def __init__(self, architecture):
        self.architecture = architecture
        self.active = False
        self._generation_lock = Lock()
        self._reuse_evals = 0
        self._fallbacks = 0

    @property
    def stats(self):
        return {"active": self.active, "reuse_evals": self._reuse_evals, "fallbacks": self._fallbacks}

    @contextmanager
    def generation(self):
        if not self._generation_lock.acquire(blocking=False):
            raise RuntimeError("LightweightResidualReuse does not support concurrent/reentrant generations")
        self._reuse_evals = self._fallbacks = 0
        self.active = True
        try:
            yield self
        finally:
            self.active = False
            self._generation_lock.release()

    def _conditioning_metadata_supported(self, context, inputs, proprio, ab):
        """Keep invalid-input handling on the original preparation route.

        This is shape/type inspection only: do not project proprio, read GPU
        values or construct masks just to decide whether a step can be light.
        """
        if context.shape[-1] != ab.text_dim:
            return False
        mask = inputs.get("context_mask")
        if mask is not None and (
            not isinstance(mask, torch.Tensor) or mask.ndim != 2 or mask.shape != context.shape[:2]
        ):
            return False
        lengths = inputs.get("seq_lens")
        if mask is None and lengths is not None and (
            not isinstance(lengths, torch.Tensor) or lengths.shape != (context.shape[0],)
        ):
            return False
        arch = self.architecture
        if not bool(getattr(arch, "_use_proprioception_context", False)):
            return True
        encoder = getattr(arch, "proprio_encoder", None)
        if encoder is None or not isinstance(proprio, torch.Tensor):
            return False
        shape = tuple(proprio.shape)
        if len(shape) == 1:
            batch, width = 1, shape[0]
        elif len(shape) == 2:
            batch, width = shape
        elif len(shape) == 3 and shape[1] == 1:
            batch, _, width = shape
        else:
            return False
        if batch != context.shape[0] or width != getattr(arch, "proprio_dim", None):
            return False
        if isinstance(encoder, torch.nn.Linear) and (
            width != encoder.in_features or context.shape[-1] != encoder.out_features
        ):
            return False
        sample_mask = inputs.get("_proprio_sample_mask")
        if sample_mask is not None:
            if not isinstance(sample_mask, torch.Tensor):
                return False
            shape = tuple(sample_mask.shape)
            if shape not in ((batch,), (batch, 1)) and not (
                len(shape) == 3 and shape[:2] == (batch, 1)
            ):
                return False
        return True

    def _supported(self, noisy_actions, action_timestep, inputs, *, proprio=None):
        vb, ab = self.architecture.video_backbone, self.architecture.action_backbone
        if not isinstance(vb, WanBase) or not isinstance(ab, ActionDiT):
            return False
        # Custom prepare/finalize semantics must retain their original route.
        for obj, name, expected in (
            (vb, "prepare", WanBase.prepare),
            (vb, "finalize", WanBase.finalize),
            (ab, "prepare_state", ActionDiT.prepare_state),
            (ab, "extract_prediction", ActionDiT.extract_prediction),
            (ab, "_embed_actions", ActionDiT._embed_actions),
        ):
            if getattr(getattr(obj, name), "__func__", None) is not expected:
                return False
        if vb.training or ab.training or ab.variant != "joint_self_attn":
            return False
        dit = vb.dit
        if not dit.seperated_timestep or not inputs.get("fuse_vae_embedding_in_latents", False):
            return False
        if vb._dit_patch_size[0] != 1 or tuple(dit.patch_size) != tuple(vb._dit_patch_size):
            return False
        if vb.vace is not None or any(
            inputs.get(key) is not None
            for key in ("y", "clip_feature", "vace_context", "motion_bucket_id", "control_camera_latents_input")
        ):
            return False
        latents, timestep, context = (inputs.get(key) for key in ("latents", "timestep", "context"))
        if not all(isinstance(value, torch.Tensor) for value in (latents, timestep, context, noisy_actions, action_timestep)):
            return False
        if latents.ndim != 5 or latents.shape[0] != 1 or timestep.shape != (1,):
            return False
        if noisy_actions.ndim != 3 or noisy_actions.shape[0] != 1 or action_timestep.shape != (1,):
            return False
        if context.ndim != 3 or context.shape[0] != 1:
            return False
        if not context.is_floating_point() or context.dtype != noisy_actions.dtype or context.device != noisy_actions.device:
            return False
        if action_timestep.dtype != noisy_actions.dtype or action_timestep.device != noisy_actions.device:
            return False
        if noisy_actions.shape[1] > ab.freqs.shape[0]:
            return False
        if not self._conditioning_metadata_supported(context, inputs, proprio, ab):
            return False
        if inputs.get("use_gradient_checkpointing") or inputs.get("use_gradient_checkpointing_offload"):
            return False
        return True

    def _time_embed(self, vb, timestep, latents, inputs):
        dit = vb.dit
        kwargs = {
            "patch_size": vb._dit_patch_size,
            "fuse_vae_embedding_in_latents": inputs.get("fuse_vae_embedding_in_latents", False),
            "force_per_token_t_mod": inputs.get("force_per_token_t_mod", True),
            "num_clean_prefix_frames": inputs.get("num_clean_prefix_frames", 0),
            "zero_clean_prefix_t_mod": inputs.get("zero_clean_prefix_t_mod", True),
            "has_first_frame_latents": inputs.get("first_frame_latents") is not None,
        }
        time_cache = getattr(vb, "_serving_time_cache", None)
        if time_cache is not None and time_cache.active:
            return time_cache.video(
                dit, timestep, latents, **kwargs,
                compute=lambda: wan_dit_forward.build_time_modulation(dit, timestep, latents, **kwargs),
            )[0]
        # Keep the original TI2V operation order, token count, MLP shape and
        # dtype boundaries. Only the subsequent time_projection is omitted.
        batch_size = latents.shape[0]
        num_clean = max(kwargs["num_clean_prefix_frames"], 1)
        f_lat = latents.shape[2]
        tokens_per_frame = latents.shape[3] * latents.shape[4] // (vb._dit_patch_size[1] * vb._dit_patch_size[2])
        token_timesteps = torch.ones(
            batch_size, f_lat, tokens_per_frame, dtype=latents.dtype, device=latents.device
        ) * timestep.view(batch_size, 1, 1)
        token_timesteps[:, :num_clean, :] = 0
        token_timesteps = token_timesteps.reshape(batch_size, -1)
        time_sinusoid = sinusoidal_embedding_1d(dit.freq_dim, token_timesteps.reshape(-1))
        return dit.time_embedding(time_sinusoid.to(latents.dtype)).reshape(batch_size, -1, dit.dim)

    def run(self, noisy_actions, action_timestep, *, proprio=None, **pipeline_inputs):
        """Run current embeddings, existing residual reuse, then both heads."""
        if not self.active:
            raise RuntimeError("Lightweight reuse requires an active generation")
        arch = self.architecture
        if arch.training or torch.is_grad_enabled():
            raise RuntimeError("Lightweight reuse requires no-grad eval-mode inference")
        cache = getattr(arch, "_serving_residual_cache", None)
        if cache is None or not cache.active or cache.next_is_refresh:
            raise RuntimeError("Lightweight reuse requires an active non-refresh residual step")
        if not self._supported(noisy_actions, action_timestep, pipeline_inputs, proprio=proprio):
            self._fallbacks += 1
            return None
        vb, ab = arch.video_backbone, arch.action_backbone
        latents, timestep = pipeline_inputs["latents"], pipeline_inputs["timestep"]
        time_embed = self._time_embed(vb, timestep, latents, pipeline_inputs)
        hidden = vb.dit.patchify(latents, None)
        grid = hidden.shape[2:]
        hidden = rearrange(hidden, "b c f h w -> b (f h w) c").contiguous()
        actions = ab._embed_actions(noisy_actions)

        def unexpected_refresh():
            raise RuntimeError("Residual schedule changed during lightweight reuse")

        hidden, actions = cache.run(hidden, actions, unexpected_refresh)
        video_state = SimpleNamespace(
            hidden_states=hidden, grid_frames=grid[0], grid_height=grid[1], grid_width=grid[2],
            extras={"dit": vb.dit, "time_embed": time_embed},
        )
        action_state = SimpleNamespace(payload=SimpleNamespace(x_action=actions))
        video_prediction = vb.finalize(video_state)
        action_prediction = ab.extract_prediction(action_state)
        cache.record_head_evaluation()
        self._reuse_evals += 1
        return video_prediction, action_prediction
