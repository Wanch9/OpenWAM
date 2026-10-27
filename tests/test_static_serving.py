from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from openwam.deploy.optimizations.dit_cache import DiTVelocityCache
from openwam.model.architectures.base import BaseWAMArchitecture


def test_similarity_reuse_and_invalidation():
    c = DiTVelocityCache(cosine_threshold=0.9, max_consecutive_skips=20, memoize_similarity=True)
    v = torch.randn(1, 4, 3)
    a = torch.randn(1, 5, 2)
    c.update(v, 1.0, a)
    c.update(v * 1.01, 0.9, a)
    original = torch.nn.functional.cosine_similarity
    with patch("torch.nn.functional.cosine_similarity", wraps=original) as fn:
        for _ in range(3):
            assert not c.should_recompute(0.8, require_action=True)
        assert fn.call_count == 2
        c.get_cached_action().neg_()
        assert c.should_recompute(0.7, require_action=True)
        assert fn.call_count == 4
        c.cosine_threshold = 1.1
        assert c.should_recompute(0.6, require_action=True)
        assert fn.call_count == 5
    c.reset()
    assert c._decision_signature is None


@pytest.mark.parametrize("memoized", [False, True])
def test_similarity_decisions_match_reference(memoized):
    torch.manual_seed(9)
    ref = DiTVelocityCache(max_consecutive_skips=3)
    actual = DiTVelocityCache(max_consecutive_skips=3, memoize_similarity=memoized)
    v = torch.randn(2, 4)
    a = torch.randn(2, 3)
    for i in range(20):
        if i % 5 == 0:
            sign = -1 if i == 10 else 1
            for c in [ref, actual]:
                c.update(v * sign, 1 - i * 0.04, a)
        assert ref.should_recompute(1 - i * 0.04, require_action=i % 2 == 0) == actual.should_recompute(
            1 - i * 0.04, require_action=i % 2 == 0
        )
        assert ref.stats == actual.stats


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_predicated_pinning_is_bitwise_identical(dtype):
    torch.manual_seed(7)
    mask = torch.tensor([True, False, False, True, False, True, False])
    old = torch.randn(2, 32, 7).to(dtype)
    new = old.clone()
    old_noise = old[..., mask].clone() / 1.0
    new_noise = new.clone() / 1.0
    for sigma in [0.9, 0.5, 0.2, 0.0]:
        velocity = torch.randn_like(old)
        old = old + velocity * 0.1
        new = new + velocity * 0.1
        old[..., mask] = old_noise * sigma
        new = torch.where(mask, new_noise * sigma, new)
        assert torch.equal(old, new)


def test_mask_resolution_preserves_explicit_and_normalizer_masks():
    for explicit in [None, torch.tensor([True, False, False, True])]:
        results = []
        for enabled in [False, True]:
            obj = SimpleNamespace(
                action_dim=4,
                normalizer=SimpleNamespace(_dst_index=[0, 3], _unify_dim=4),
                _static_action_pinning=enabled,
            )
            results.append(BaseWAMArchitecture._resolve_inactive_action_dims(obj, explicit, torch.device("cpu")))
        assert torch.equal(*results)


def test_inference_tensor_falls_back_without_version_counters():
    with torch.inference_mode():
        c = DiTVelocityCache(memoize_similarity=True, max_consecutive_skips=3)
        v = torch.ones(1, 4)
        c.update(v, 1.0, v)
        c.update(v, 0.9, v)
        assert not c.should_recompute(0.8, require_action=True)
        assert not c.should_recompute(0.7, require_action=True)
        assert c._decision_signature is None


def test_memoized_similarity_preserves_nan_comparison_behavior():
    v = torch.tensor([float("nan"), 1.0])
    for enabled in [False, True]:
        c = DiTVelocityCache(memoize_similarity=enabled)
        c.update(v, 1.0)
        c.update(v, 0.9)
        assert not c.should_recompute(0.8)
        assert not c.should_recompute(0.7)


def test_eval_guard_tracks_children_and_topology_changes():
    from torch import nn

    model = nn.Sequential(nn.Linear(3, 4), nn.Dropout()).eval()
    model._skip_redundant_eval = True
    with patch.object(nn.Module, "eval", side_effect=AssertionError("redundant eval")):
        BaseWAMArchitecture._ensure_eval_for_generation(model)
    model[1].train()
    BaseWAMArchitecture._ensure_eval_for_generation(model)
    assert not model[1].training
    model.add_module("new", nn.Dropout())
    BaseWAMArchitecture._ensure_eval_for_generation(model)
    assert not model.new.training


def test_eval_guard_preserves_custom_train_side_effects():
    from torch import nn

    class Custom(nn.Module):
        def train(self, mode=True):
            self.calls = getattr(self, "calls", 0) + 1
            return super().train(mode)

    child = Custom()
    model = nn.Sequential(child).eval()
    model._skip_redundant_eval = True
    previous = child.calls
    BaseWAMArchitecture._ensure_eval_for_generation(model)
    assert child.calls == previous + 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_batched_timesteps_match_scalar_construction(dtype):
    schedule = [(1000.0, 1000.0), (912.345678, 800.00123), (333.333333, 456.78901), (0.0, 0.0)]
    batch = torch.as_tensor(schedule[:-1], dtype=dtype).reshape(-1, 2)
    for i, (tv, ta) in enumerate(schedule[:-1]):
        assert torch.equal(batch[i, :1], torch.tensor([tv], dtype=dtype))
        assert torch.equal(batch[i, 1:], torch.tensor([ta], dtype=dtype))
