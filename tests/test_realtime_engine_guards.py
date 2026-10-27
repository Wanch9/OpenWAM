"""CPU-only serving contract tests for the opt-in RealtimeWAM paths."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from openwam.deploy.engine import JointInferenceEngine
from openwam.deploy.optimizations.residual_cache import TransformerResidualCache


def _config(**optimization):
    return SimpleNamespace(
        inference=SimpleNamespace(denoise_mode="sync", denoise_steps=10, cfg_scale=1.0),
        optimization=SimpleNamespace(**optimization),
    )


class _ContextSpy:
    def __init__(self, name, events):
        self.name, self.events = name, events
        self.active = False
        self.prefill_evals = self.joint_evals = 0

    @property
    def stats(self):
        return {"active": self.active, "prefill_evals": self.prefill_evals, "joint_evals": self.joint_evals}

    @contextmanager
    def generation(self):
        assert not self.active
        self.active = True
        self.prefill_evals = self.joint_evals = 0
        self.events.append((self.name, "enter"))
        try:
            yield self
        finally:
            self.active = False
            self.events.append((self.name, "exit"))


class _Architecture:
    def __init__(self):
        self.video_scheduler = object()
        self.action_scheduler = object()
        self.video_backbone = SimpleNamespace(shift_video=5.0)
        self.action_backbone = SimpleNamespace(shift_action=5.0)
        self.calls = 0
        self.fail = False
        self.last_kwargs = None
        self._serving_residual_cache = None
        self._serving_prefix_cache = None

    def generate(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        assert not torch.is_grad_enabled()
        residual = self._serving_residual_cache
        prefix = self._serving_prefix_cache
        if prefix is not None:
            assert prefix.active
        if residual is not None:
            assert residual.active
            x = torch.zeros(1, 2, 3)
            for step in range(10):
                residual.run(x + step, x + step, lambda: (x + step + 1, x + step + 2))
                residual.record_head_evaluation()
                if self.fail and step == 2:
                    raise RuntimeError("simulated generation failure")
        elif self.fail:
            raise RuntimeError("simulated generation failure")
        if prefix is not None:
            prefix.prefill_evals = 1
            prefix.joint_evals = residual.stats["full_evals"] if residual is not None else kwargs["num_inference_steps"]
        return {"actions": torch.zeros(32, 10)}


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(
        "openwam.deploy.engine.make_schedule",
        lambda mode, **kwargs: [(float(i), float(i)) for i in range(kwargs["num_steps"], -1, -1)],
    )
    return JointInferenceEngine(_config(), architecture=_Architecture())


def _enable_residual(engine):
    cache = TransformerResidualCache((0, 3, 6, 9))
    engine._residual_cache = cache
    engine.architecture._serving_residual_cache = cache
    return cache


def _enable_prefix(engine, events):
    cache = _ContextSpy("prefix", events)
    engine._prefix_cache = cache
    engine.architecture._serving_prefix_cache = cache
    return cache


@pytest.mark.parametrize("conditions", [{"denoise_mode": "async"}, {"denoise_steps": 9}, {"denoise_steps": 11}])
def test_residual_rejects_request_schedule_override(engine, conditions):
    residual = _enable_residual(engine)
    with pytest.raises(ValueError, match="synchronous 10-step"):
        engine.generate(conditions)
    assert engine.architecture.calls == 0 and not residual.active


@pytest.mark.parametrize("cache_kind", ["residual", "prefix"])
def test_live_cfg_change_rejected_before_generation(engine, cache_kind):
    cache = _enable_residual(engine) if cache_kind == "residual" else _enable_prefix(engine, [])
    engine._cfg_scale = 2.0
    with pytest.raises(ValueError, match="cfg_scale=1"):
        engine.generate({})
    assert not cache.active and engine.architecture.calls == 0


@pytest.mark.parametrize("cache_kind", ["residual", "prefix"])
def test_cache_rebinding_rejected_before_generation(engine, cache_kind):
    cache = _enable_residual(engine) if cache_kind == "residual" else _enable_prefix(engine, [])
    setattr(engine.architecture, f"_serving_{cache_kind}_cache", object())
    with pytest.raises(RuntimeError, match="not bound"):
        engine.generate({})
    assert not cache.active and engine.architecture.calls == 0


def test_velocity_cache_cannot_be_added_to_residual_engine(engine):
    _enable_residual(engine)
    engine._dit_cache = object()
    with pytest.raises(ValueError, match="mutually exclusive"):
        engine.generate({})
    assert engine.architecture.calls == 0


def test_prefix_rejects_async_request(engine):
    prefix = _enable_prefix(engine, [])
    with pytest.raises(ValueError, match="synchronous sampling"):
        engine.generate({"denoise_mode": "async"})
    assert not prefix.active and engine.architecture.calls == 0


def test_nested_cache_lifetimes_and_stats_reset_per_request(engine):
    events = []
    residual = _enable_residual(engine)
    prefix = _enable_prefix(engine, events)
    conditioning = _ContextSpy("conditioning", events)
    engine._conditional_kv_cache = conditioning
    for seed in (42, 99):
        result = engine.generate({"seed": seed, "prompt": f"request-{seed}"})
        assert result["residual_cache_stats"]["full_evals"] == 4
        assert result["residual_cache_stats"]["reused_evals"] == 6
        assert result["residual_cache_stats"]["head_evals"] == 10
        assert result["prefix_cache_stats"]["prefill_evals"] == 1
        assert result["prefix_cache_stats"]["joint_evals"] == 4
        assert not result["residual_cache_stats"]["active"]
        assert not residual.active and not prefix.active and not conditioning.active
        assert residual._video_residual is None
        assert engine.architecture.last_kwargs["seed"] == seed
        assert engine.architecture.last_kwargs["dit_cache"] is None
    assert events == [("conditioning", "enter"), ("prefix", "enter"), ("prefix", "exit"), ("conditioning", "exit")] * 2


def test_model_exception_closes_all_caches_and_keeps_original_error(engine):
    events = []
    residual = _enable_residual(engine)
    prefix = _enable_prefix(engine, events)
    conditioning = _ContextSpy("conditioning", events)
    engine._conditional_kv_cache = conditioning
    engine.architecture.fail = True
    with pytest.raises(RuntimeError, match="simulated generation failure"):
        engine.generate({})
    assert not residual.active and not prefix.active and not conditioning.active
    assert residual._video_residual is None
    engine.architecture.fail = False
    result = engine.generate({})
    assert result["residual_cache_stats"]["head_evals"] == 10


def test_disabled_features_leave_generic_architecture_and_other_budgets_unchanged(engine):
    result = engine.generate({"denoise_steps": 4, "denoise_mode": "async"})
    assert "residual_cache_stats" not in result and "prefix_cache_stats" not in result
    assert engine.architecture.last_kwargs["num_inference_steps"] == 4
    assert len(engine.architecture.last_kwargs["schedule"]) == 5


@pytest.mark.parametrize("option", ["residual_cache", "observation_prefix_cache"])
def test_init_rejects_unsupported_architecture(option):
    cfg = _config(**{option: SimpleNamespace(enabled=True)})
    with pytest.raises(ValueError, match="dual-system"):
        JointInferenceEngine(cfg, architecture=_Architecture())


@pytest.mark.parametrize("option", ["residual_cache", "observation_prefix_cache"])
def test_init_rejects_cfg_for_supported_architecture(option):
    from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
    from openwam.model.video_backbone.wan_backbone import WanBase

    arch = Mock(spec=DualSystemSelfAttnArchitecture)
    arch.video_backbone = Mock(spec=WanBase)
    cfg = _config(**{option: SimpleNamespace(enabled=True)})
    cfg.inference.cfg_scale = 2.0
    with pytest.raises(ValueError, match="classifier-free guidance"):
        JointInferenceEngine(cfg, architecture=arch)


def test_init_rejects_two_approximation_caches():
    cfg = _config(
        residual_cache=SimpleNamespace(enabled=True),
        dit_cache=SimpleNamespace(enabled=True, cosine_threshold=0.99, max_skips=3),
    )
    with pytest.raises(ValueError, match="must not be enabled together"):
        JointInferenceEngine(cfg, architecture=_Architecture())
