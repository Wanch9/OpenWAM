"""Deploy-side inference engine: the deployment adapter around ``architecture.generate``.

``BaseInferenceEngine`` declares the engine interface; ``JointInferenceEngine``
is the production implementation. The engine translates deploy config +
per-request conditions into ``architecture.generate(...)`` arguments, builds
the denoise schedule, and owns server-lifetime caches (prompt embeddings,
VACE context). The actual denoising loop lives on the model side
(``BaseWAMArchitecture.generate``).
"""

import inspect
import logging
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import ExitStack
from typing import Any, Optional

import torch

from openwam.deploy.denoise_schedule import make_schedule, normalize_denoise_config
from openwam.model.architectures.base import BaseWAMArchitecture

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_EMBED_CACHE_MAXSIZE = 32


class BaseInferenceEngine(ABC):
    """Inference engine base class.

    Concrete engines implement ``generate`` which accepts observation
    conditions and produces video frames and/or action trajectories.

    Args:
        cfg: Hydra config.
        architecture: WAM architecture wrapping the action backbone.
        action_backbone: Optional action backbone reference for implementations that still expose one.
    """

    require_architecture = False

    def __init__(self, cfg, architecture: Optional[BaseWAMArchitecture] = None, action_backbone=None):
        if self.require_architecture and architecture is None:
            raise ValueError("architecture is required")
        self.cfg = cfg
        self.architecture = architecture
        self.action_backbone = action_backbone

    @abstractmethod
    def generate(self, conditions: dict) -> dict:
        """Generate video and/or actions from conditions.

        Args:
            conditions: dict with keys like ``prompt``, ``reference_image``,
                ``context_video``, ``seed``, etc.

        Returns:
            dict with ``video`` (Tensor) and ``actions`` (Tensor).
        """
        ...


class _BoundedPromptEmbedCache(OrderedDict):
    """LRU-bounded dict for ``prompt -> inputs_posi``."""

    def __init__(self, maxsize: int = DEFAULT_PROMPT_EMBED_CACHE_MAXSIZE):
        super().__init__()
        self._maxsize = max(1, int(maxsize))
        self._evict_warned = False

    def __getitem__(self, key: Any) -> Any:
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self._maxsize:
            # ``OrderedDict.popitem`` retrieves the value through ``self[key]``.
            # Since ``__getitem__`` updates recency, that lookup runs after the
            # key has been removed from the ordering map and raises ``KeyError``.
            # Delete through the base implementation to bypass the override.
            evicted_key = next(iter(self))
            super().__delitem__(evicted_key)
            if not self._evict_warned:
                self._evict_warned = True
                logger.warning("prompt_embed_cache exceeded maxsize=%d; evicted %r", self._maxsize, evicted_key)


class JointInferenceEngine(BaseInferenceEngine):
    """Joint video-action inference engine.

    Wraps the package-native joint generation loop through the
    :class:`BaseInferenceEngine` interface.

    Args:
        cfg: Hydra config (must contain ``cfg.inference``).
        architecture: WAM architecture wrapping the action backbone.
        action_backbone: Optional action backbone reference retained for base-class storage.
    """

    require_architecture = True

    def __init__(
        self,
        cfg,
        architecture: Optional[BaseWAMArchitecture] = None,
        action_backbone=None,
    ):
        super().__init__(cfg, architecture=architecture, action_backbone=action_backbone)
        normalize_denoise_config(getattr(cfg, "inference", None))

        self._architecture_generate_accepts_extra_kwargs: Optional[bool] = None
        self._architecture_generate_kwarg_names: Optional[set[str]] = None
        self._architecture_generate_warned_dropped_kwargs: set[tuple[str, tuple[str, ...]]] = set()
        self._init_optimizations()
        self._init_cfg()

    def _filter_architecture_generate_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop deploy-only kwargs the architecture's generate() cannot consume."""

        if getattr(self, "_architecture_generate_kwarg_names", None) is None:
            params = inspect.signature(self.architecture.generate).parameters
            self._architecture_generate_accepts_extra_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
            )
            self._architecture_generate_kwarg_names = {
                name
                for name, param in params.items()
                if param.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }
        if self._architecture_generate_accepts_extra_kwargs:
            return kwargs
        accepted = self._architecture_generate_kwarg_names
        dropped = {name: value for name, value in kwargs.items() if name not in accepted}
        meaningful_dropped = tuple(
            sorted(name for name, value in dropped.items() if not self._is_noop_dropped_generate_kwarg(name, value))
        )
        if meaningful_dropped:
            warned = getattr(self, "_architecture_generate_warned_dropped_kwargs", set())
            architecture_name = type(self.architecture).__name__
            warning_key = (architecture_name, meaningful_dropped)
            if warning_key not in warned:
                warned.add(warning_key)
                self._architecture_generate_warned_dropped_kwargs = warned
                logger.warning(
                    "%s.generate does not accept deploy kwarg(s) %s; dropping them for this request. "
                    "Check deploy config for unsupported features such as CFG on specialized architectures.",
                    architecture_name,
                    ", ".join(meaningful_dropped),
                )
        return {name: value for name, value in kwargs.items() if name in accepted}

    @staticmethod
    def _is_noop_dropped_generate_kwarg(name: str, value: Any) -> bool:
        """Return whether dropping an unsupported deploy kwarg preserves default behavior."""

        if name == "cfg_scale":
            try:
                return float(value) == 1.0
            except (TypeError, ValueError):
                return False
        if name == "cfg_merge":
            return value is False
        if name == "decode_video":
            return value is True
        if name == "profile":
            return value is False
        if name == "vace_cache":
            return not bool(value)
        if name == "prompt_embed_cache":
            # None means the cache was disabled by config — dropping it is a no-op.
            return value is None or (
                isinstance(value, _BoundedPromptEmbedCache)
                and len(value) == 0
                and value._maxsize == DEFAULT_PROMPT_EMBED_CACHE_MAXSIZE
            )
        return value is None

    def _init_optimizations(self):
        """Read cfg.optimization and instantiate optimization components."""
        optimization = getattr(self.cfg, "optimization", None)

        # DiT velocity cache
        self._dit_cache = None
        if optimization and getattr(optimization, "dit_cache", None):
            dc = optimization.dit_cache
            if getattr(dc, "enabled", False):
                from openwam.deploy.optimizations import DiTVelocityCache

                self._dit_cache = DiTVelocityCache(
                    cosine_threshold=getattr(dc, "cosine_threshold", 0.99),
                    max_consecutive_skips=getattr(dc, "max_skips", 3),
                    memoize_similarity=bool(getattr(dc, "memoize_similarity", False)),
                )
                logger.info("DiT velocity cache enabled (threshold=%.3f)", dc.cosine_threshold)

        self._residual_cache = None
        self.architecture._serving_residual_cache = None
        residual_cfg = getattr(optimization, "residual_cache", None) if optimization else None
        if residual_cfg is not None and getattr(residual_cfg, "enabled", False):
            from openwam.deploy.optimizations.residual_cache import TransformerResidualCache
            from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
            from openwam.model.video_backbone.wan_backbone import WanBase

            if self._dit_cache is not None:
                raise ValueError("Transformer residual reuse and velocity reuse must not be enabled together.")
            if not isinstance(self.architecture, DualSystemSelfAttnArchitecture) or not isinstance(
                self.architecture.video_backbone, WanBase
            ):
                raise ValueError("Residual reuse currently supports Wan dual-system joint self-attention only.")
            if float(getattr(self.cfg.inference, "cfg_scale", 1.0)) != 1.0:
                raise ValueError("Residual reuse does not support classifier-free guidance.")
            self._residual_cache = TransformerResidualCache(
                refresh_steps=tuple(getattr(residual_cfg, "refresh_steps", [0, 3, 6, 9])), total_steps=10
            )
            self.architecture._serving_residual_cache = self._residual_cache

        self._prefix_cache = None
        self.architecture._serving_prefix_cache = None
        prefix_cfg = getattr(optimization, "observation_prefix_cache", None) if optimization else None
        if prefix_cfg is not None and getattr(prefix_cfg, "enabled", False):
            from openwam.deploy.optimizations.prefix_cache import ObservationPrefixCache
            from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture

            if not isinstance(self.architecture, DualSystemSelfAttnArchitecture):
                raise ValueError("Observation-prefix reuse currently supports dual-system joint self-attention only.")
            if float(getattr(self.cfg.inference, "cfg_scale", 1.0)) != 1.0:
                raise ValueError("Observation-prefix reuse does not support classifier-free guidance.")
            self._prefix_cache = ObservationPrefixCache(
                self.architecture,
                compile_enabled=bool(getattr(prefix_cfg, "compile_enabled", True)),
                compile_mode=str(getattr(prefix_cfg, "compile_mode", "reduce-overhead")),
                verify_invariants=bool(getattr(prefix_cfg, "verify_invariants", True)),
            )
            self.architecture._serving_prefix_cache = self._prefix_cache

        self._rope_cache = None
        self._time_cache = None
        self._lightweight_reuse = None
        self.architecture._serving_time_cache = None
        self.architecture._serving_lightweight_reuse = None
        video_backbone = getattr(self.architecture, "video_backbone", None)
        action_backbone = getattr(self.architecture, "action_backbone", None)
        if video_backbone is not None:
            video_backbone._serving_rope_cache = None
            video_backbone._serving_time_cache = None
        if action_backbone is not None:
            action_backbone._serving_time_cache = None
        rope_cfg = getattr(optimization, "rope_cache", None) if optimization else None
        if rope_cfg is not None and getattr(rope_cfg, "enabled", False):
            from openwam.deploy.optimizations.rope_cache import WanRoPECache
            from openwam.model.video_backbone.wan_backbone import WanBase

            if not isinstance(video_backbone, WanBase):
                raise ValueError("Prepared RoPE caching currently supports Wan only.")
            self._rope_cache = WanRoPECache(max_entries=1)
            video_backbone._serving_rope_cache = self._rope_cache
        time_cfg = getattr(optimization, "fixed_schedule_time_cache", None) if optimization else None
        if time_cfg is not None and getattr(time_cfg, "enabled", False):
            from openwam.deploy.optimizations.time_cache import FixedScheduleTimeCache
            from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture

            if not isinstance(self.architecture, DualSystemSelfAttnArchitecture):
                raise ValueError("Fixed-schedule time caching requires joint self-attention serving.")
            self._time_cache = FixedScheduleTimeCache(self.architecture)
            self.architecture._serving_time_cache = self._time_cache
            video_backbone._serving_time_cache = self._time_cache
            action_backbone._serving_time_cache = self._time_cache
        light_cfg = getattr(optimization, "lightweight_residual_reuse", None) if optimization else None
        if light_cfg is not None and getattr(light_cfg, "enabled", False):
            from openwam.deploy.optimizations.lightweight_reuse import LightweightResidualReuse

            if self._residual_cache is None:
                raise ValueError("Lightweight residual reuse requires the Transformer residual cache.")
            self._lightweight_reuse = LightweightResidualReuse(self.architecture)
            self.architecture._serving_lightweight_reuse = self._lightweight_reuse

        # torch.compile — ActionDiT, Video DiT, VAE
        if optimization and getattr(optimization, "compile", None):
            try:
                self.architecture.apply_compile_optimizations(optimization.compile)
            except Exception as e:
                logger.warning("torch.compile optimization failed, continuing without: %s", e)

        # VAE decode skip
        self._decode_video = True
        if optimization:
            self._decode_video = getattr(optimization, "decode_video", True)

        # Profiling
        self._profile = os.environ.get("WAM_PROFILE", "0") == "1"

        # VACE context cache for closed-loop reuse
        self._vace_cache: dict = {}

        # Prompt-keyed text embedding cache (bounded LRU). enabled: false turns
        # it off entirely: the backbones treat a None cache as "never cache".
        cache_enabled = True
        cache_maxsize = DEFAULT_PROMPT_EMBED_CACHE_MAXSIZE
        if optimization is not None:
            cache_cfg = getattr(optimization, "prompt_embed_cache", None)
            if cache_cfg is not None:
                cache_enabled = bool(getattr(cache_cfg, "enabled", True))
                cache_maxsize = int(getattr(cache_cfg, "maxsize", cache_maxsize))
        if cache_enabled:
            self._prompt_embed_cache = _BoundedPromptEmbedCache(maxsize=cache_maxsize)
        else:
            self._prompt_embed_cache = None
            logger.info("prompt_embed_cache disabled by config")

        self.architecture._precompute_timesteps = bool(
            getattr(optimization, "precompute_timesteps", False) if optimization else False
        )
        self.architecture._skip_redundant_eval = bool(
            getattr(optimization, "skip_redundant_eval", False) if optimization else False
        )
        self.architecture._static_action_pinning = bool(
            getattr(optimization, "static_action_pinning", False) if optimization else False
        )
        rope_mode = str(getattr(optimization, "rope_kernel", "torch") if optimization else "torch")
        if rope_mode not in ("torch", "triton_rope", "triton_norm_rope"):
            raise ValueError(f"Unknown rope_kernel: {rope_mode}")
        if rope_mode != "torch":
            from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
            from openwam.model.ops.fused_norm_rope import fused_norm_rope, triton
            from openwam.model.video_backbone.wan_backbone import WanBase

            if not isinstance(self.architecture, DualSystemSelfAttnArchitecture) or not isinstance(
                self.architecture.video_backbone, WanBase
            ):
                raise ValueError("RoPE kernels currently support Wan dual-system joint self-attention.")
            if self.architecture.device.type != "cuda" or triton is None:
                raise ValueError("Triton RoPE configuration requires CUDA and Triton.")
            modules = [b.self_attn for b in self.architecture.video_backbone.dit.blocks] + [
                b.self_attn for b in self.architecture.action_backbone.blocks
            ]
            for module in modules:
                module._serving_rope_mode = rope_mode
                module._serving_rope_fn = fused_norm_rope

        elif callable(getattr(self.architecture, "modules", None)):
            # Reinitializing an engine with the same model must honor rollback.
            for module in self.architecture.modules():
                if hasattr(module, "_serving_rope_mode"):
                    module._serving_rope_mode = "torch"

        self._conditional_kv_cache = None
        kv_cfg = getattr(optimization, "conditional_kv_cache", None) if optimization else None
        if kv_cfg is not None and getattr(kv_cfg, "enabled", False):
            if float(getattr(self.cfg.inference, "cfg_scale", 1.0)) != 1.0:
                raise ValueError("Conditional K/V reuse does not support classifier-free guidance.")
            from openwam.deploy.optimizations.conditional_kv import ConditionalKVCache

            self._conditional_kv_cache = ConditionalKVCache(
                self.architecture,
                compact_context=bool(getattr(kv_cfg, "compact_context", False)),
                precompute_context=bool(getattr(kv_cfg, "precompute_context", False)),
            )
            self.architecture._serving_conditioning_cache = self._conditional_kv_cache

        split_cfg = getattr(optimization, "split_prefix_attention", None) if optimization else None
        if split_cfg is not None and getattr(split_cfg, "enabled", False):
            from openwam.model.video_backbone.wan_backbone import WanBase

            driver = getattr(self.architecture, "mot_driver", None)
            if (
                driver is None
                or not isinstance(self.architecture.video_backbone, WanBase)
                or driver.attention_mask_mode != "mutual"
                or driver.vb.video_attention_mask_mode != "first_frame_causal"
            ):
                raise ValueError("Split attention requires Wan dual-system mutual + first-frame-causal attention.")
            driver.split_prefix_attention = True

    def _init_cfg(self):
        """Resolve Classifier-Free Guidance from cfg.inference (CosmosPredict25 only; cfg_scale=1.0 is a no-op).

        The uncond embedding is built by the backbone's live encoder
        (``_build_uncond_context`` → ``text_encoder("")``).
        """
        inf_cfg = getattr(self.cfg, "inference", None)
        self._cfg_scale: float = float(getattr(inf_cfg, "cfg_scale", 1.0)) if inf_cfg else 1.0
        self._cfg_merge: bool = bool(getattr(inf_cfg, "cfg_merge", False)) if inf_cfg else False
        if self._cfg_scale < 1.0:
            raise ValueError(f"inference.cfg_scale must be >= 1.0; got {self._cfg_scale}.")
        if self._cfg_scale > 1.0:
            logger.info("CFG enabled: cfg_scale=%.3f cfg_merge=%s", self._cfg_scale, self._cfg_merge)

    @torch.no_grad()
    def generate(self, conditions: dict) -> dict:
        """Generate video and/or actions from conditions.

        Args:
            conditions: dict with keys:
                - prompt (str): text prompt
                - vace_video (list[PIL.Image], optional): VACE conditioning video (Wan2.1-VACE only)
                - first_frame_image (list[PIL.Image], optional): first frame of the
                  observation window; used as TI2V first-frame condition on Wan2.2-TI2V
                  or as VACE spatial reference on Wan2.1-VACE
                - num_frames (int, optional): raw state/action window length;
                  generated action chunk length is ``num_frames - 1``
                - video_num_frames (int, optional): Wan video length after any
                  training-time video_stride sub-sampling; defaults from cfg,
                  then falls back to ``num_frames``
                - height (int, optional): defaults from cfg
                - width (int, optional): defaults from cfg
                - seed (int, optional): random seed, default 42
                - tiled (bool, optional): tiled VAE decoding, default True
                - input_video_latents (Tensor, optional): precomputed video latents
                - denoise_mode (str, optional): "sync" or "async"; "sync" ignores
                  the three async controls below, including any inherited from cfg,
                  so a request can downgrade to the sync trajectory on its own
                - lead_modality (str, optional): "async" only; "action" or "video"
                - variance_shift_alpha (float, optional): "async" only; lead curve shift
                - linear_offset (float, optional): "async" only; lag start delay
                - denoise_steps (int, optional): override num denoising steps

        Returns:
            dict with ``video`` (list of PIL images or None) and ``actions`` (numpy array).
        """
        inf_cfg = self.cfg.inference

        denoise_mode = conditions.get("denoise_mode", getattr(inf_cfg, "denoise_mode", "sync"))
        denoise_steps = conditions.get("denoise_steps", inf_cfg.denoise_steps)
        prefix_cache = getattr(self, "_prefix_cache", None)
        if prefix_cache is not None:
            if denoise_mode != "sync" or self._cfg_scale != 1.0:
                raise ValueError("Observation-prefix reuse requires synchronous sampling and cfg_scale=1.")
            if getattr(self.architecture, "_serving_prefix_cache", None) is not prefix_cache:
                raise RuntimeError("Observation-prefix cache is not bound to this architecture.")
        residual_cache = getattr(self, "_residual_cache", None)
        if residual_cache is not None:
            if denoise_mode != "sync" or int(denoise_steps) != 10 or self._cfg_scale != 1.0:
                raise ValueError("Residual reuse requires synchronous 10-step sampling and cfg_scale=1.")
            if self._dit_cache is not None:
                raise ValueError("Residual reuse and velocity reuse are mutually exclusive.")
            if getattr(self.architecture, "_serving_residual_cache", None) is not residual_cache:
                raise RuntimeError("Residual cache is not bound to this architecture.")
        time_cache = getattr(self, "_time_cache", None)
        if time_cache is not None:
            if denoise_mode != "sync" or self._cfg_scale != 1.0:
                raise ValueError("Fixed-schedule time caching requires synchronous sampling and cfg_scale=1.")
            if any(getattr(owner, "_serving_time_cache", None) is not time_cache for owner in (
                self.architecture, self.architecture.video_backbone, self.architecture.action_backbone
            )):
                raise RuntimeError("Fixed-schedule time cache is not bound to this architecture.")
        lightweight_reuse = getattr(self, "_lightweight_reuse", None)
        if lightweight_reuse is not None:
            if residual_cache is None:
                raise ValueError("Lightweight reuse requires an active residual-cache configuration.")
            if getattr(self.architecture, "_serving_lightweight_reuse", None) is not lightweight_reuse:
                raise RuntimeError("Lightweight reuse is not bound to this architecture.")
        lead_modality = conditions.get("lead_modality", getattr(inf_cfg, "lead_modality", "video"))
        variance_shift_alpha = conditions.get("variance_shift_alpha", getattr(inf_cfg, "variance_shift_alpha", 1.0))
        linear_offset = conditions.get("linear_offset", getattr(inf_cfg, "linear_offset", 0.0))
        # Single source of truth for each stream's α-shift is the backbone
        # property — ``action_backbone.shift_action`` and
        # ``video_backbone.shift_video`` — set via the model yaml and saved in
        # the checkpoint, so the training sigma buffer (set by
        # ``init_training_schedulers`` via the same properties) and the
        # inference trajectory match. Deploy carries no independent shift knob.
        # ``conditions[...]`` allows a per-request override (smoke tests /
        # ablations). ``getattr`` guards lightweight stub architectures.
        _ab = getattr(self.architecture, "action_backbone", None)
        shift = conditions.get("shift", getattr(_ab, "shift_action", None) if _ab is not None else None)
        if shift is None:
            shift = 5.0
        _vb = getattr(self.architecture, "video_backbone", None)
        shift_video = conditions.get(
            "shift_video",
            getattr(_vb, "shift_video", None) if _vb is not None else None,
        )

        schedule = make_schedule(
            denoise_mode,
            video_scheduler=self.architecture.video_scheduler,
            action_scheduler=self.architecture.action_scheduler,
            num_steps=denoise_steps,
            shift=shift,
            shift_video=shift_video,
            lead=lead_modality,
            alpha=variance_shift_alpha,
            offset=linear_offset,
        )

        # Reset dit cache for each generation
        if self._dit_cache is not None:
            self._dit_cache.reset()

        # Deploy proprio: array-like in, normalized model-space tensor out.
        proprio = conditions.get("proprio")
        if proprio is None:
            observation = conditions.get("observation") or {}
            proprio = observation.get("state") if isinstance(observation, dict) else None
        if proprio is not None:
            proprio = self.architecture.normalize_deploy_proprio(proprio)

        action_num_frames = int(conditions.get("num_frames", getattr(inf_cfg, "num_frames", 49)))
        video_num_frames = int(
            conditions.get(
                "video_num_frames",
                getattr(inf_cfg, "video_num_frames", action_num_frames),
            )
        )

        generate_kwargs = self._filter_architecture_generate_kwargs(
            {
                "schedule": schedule,
                "prompt": conditions.get("prompt", ""),
                "vace_video": conditions.get("vace_video", None),
                "first_frame_image": conditions.get("first_frame_image", None),
                "num_frames": video_num_frames,
                "action_num_frames": action_num_frames,
                "height": conditions.get("height", getattr(inf_cfg, "height", 384)),
                "width": conditions.get("width", getattr(inf_cfg, "width", 320)),
                "seed": conditions.get("seed", 42),
                "tiled": conditions.get("tiled", True),
                "input_video_latents": conditions.get("input_video_latents", None),
                "num_inference_steps": denoise_steps,
                "shift": shift,
                "dit_cache": self._dit_cache,
                "decode_video": self._decode_video,
                "profile": self._profile,
                "vace_cache": self._vace_cache,
                "prompt_embed_cache": self._prompt_embed_cache,
                "proprio": proprio,
                "cfg_scale": self._cfg_scale,
                "cfg_merge": self._cfg_merge,
            }
        )
        with ExitStack() as stack:
            if time_cache is not None:
                stack.enter_context(time_cache.generation(
                    schedule,
                    timestep_scales=(float(self.architecture.video_scheduler.num_train_timesteps),
                                     float(self.architecture.action_scheduler.num_train_timesteps)),
                    precomputed=bool(getattr(self.architecture, "_precompute_timesteps", False)),
                ))
            if self._conditional_kv_cache is not None:
                stack.enter_context(self._conditional_kv_cache.generation())
            if prefix_cache is not None:
                stack.enter_context(prefix_cache.generation())
            if residual_cache is not None:
                stack.enter_context(residual_cache.generation())
            if lightweight_reuse is not None:
                stack.enter_context(lightweight_reuse.generation())
            result = self.architecture.generate(**generate_kwargs)

        if residual_cache is not None:
            result["residual_cache_stats"] = dict(residual_cache.stats)
        if prefix_cache is not None:
            result["prefix_cache_stats"] = dict(prefix_cache.stats)
        if time_cache is not None:
            result["time_cache_stats"] = dict(time_cache.stats)
        if lightweight_reuse is not None:
            result["lightweight_reuse_stats"] = dict(lightweight_reuse.stats)

        # Attach optimization stats if profiling
        if self._profile and self._dit_cache is not None:
            result["dit_cache_stats"] = self._dit_cache.stats

        return result
