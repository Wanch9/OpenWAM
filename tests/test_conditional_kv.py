import torch

from openwam.deploy.optimizations.conditional_kv import _PackedKV
from openwam.model.video_backbone.wan.models.dit import CrossAttention


def test_projection_reuse_context_and_weight_changes():
    torch.manual_seed(42)
    modules = [CrossAttention(32, 2).eval() for _ in range(3)]
    cache = _PackedKV(modules)
    before = [set(module.state_dict()) for module in modules]
    with torch.no_grad():
        for iteration in range(3):
            context = torch.randn(2, 7, 32)
            if iteration == 2:
                modules[0].k.weight.add_(0.01)
            expected = [(module.norm_k(module.k(context)), module.v(context)) for module in modules]
            cache.prepare(context)
            for module, (key, value) in zip(modules, expected):
                torch.testing.assert_close(module._serving_key_cache, key, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(module._serving_value_cache, value, rtol=1e-5, atol=1e-6)
            assert [set(module.state_dict()) for module in modules] == before
    assert cache.rebuilds == 2


def test_training_ignores_serving_cache():
    module = CrossAttention(32, 2).eval()
    with torch.no_grad():
        _PackedKV([module]).prepare(torch.randn(1, 5, 32))
    module.train()
    module(torch.randn(1, 3, 32), torch.randn(1, 5, 32)).square().mean().backward()
    assert module.k.weight.grad is not None and torch.isfinite(module.k.weight.grad).all()


def test_compact_padding_preserves_attention_and_updates_mask():
    torch.manual_seed(19)
    module = CrossAttention(32, 2).eval()
    cache = _PackedKV([module], compact_context=True)
    x = torch.randn(1, 3, 32)
    context = torch.randn(1, 65, 32)
    for count in (7, 13, 40):
        mask = torch.arange(65)[None, :] % 2 == 0
        mask[:, count:] = False
        expanded = mask[:, None, None, :].expand(-1, 1, 3, -1)
        module._serving_kv_active = False
        with torch.no_grad():
            expected = module(x, context, ctx_mask=expanded)
            cache.prepare(context, expanded)
            actual = module(x, context, ctx_mask=expanded)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        assert cache.keys.shape[2] % 32 == 0
    import pytest

    with pytest.raises(ValueError, match="invariant"):
        cache.prepare(context, mask[:, None, :].expand(-1, 3, -1).clone())


def test_split_joint_attention_matches_dense_mask():
    from types import SimpleNamespace

    from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver

    for prefix in (1, 7, 13):
        q, k, v = [torch.randn(2, 19, 32) for _ in range(3)]
        mask = torch.ones(19, 19, dtype=torch.bool)
        mask[:prefix, prefix:] = False
        driver = SimpleNamespace(num_heads=2)
        expected = DualSystemMoTDriver._mixed_attention(driver, q, k, v, mask)
        actual = DualSystemMoTDriver._mixed_attention(driver, q, k, v, mask, first_frame_tokens=prefix)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_generation_clears_flags_after_failure_and_rejects_gradients():
    import threading
    from types import SimpleNamespace

    import pytest

    from openwam.deploy.optimizations.conditional_kv import ConditionalKVCache

    cache = ConditionalKVCache.__new__(ConditionalKVCache)
    cache.architecture = SimpleNamespace(training=False)
    cache.modules = [SimpleNamespace(_serving_kv_active=False)]
    cache.lock = threading.RLock()
    cache.generations = 0
    cache.active = cache.ready = False
    cache.context_inputs = None
    cache.embeddings = {}
    with torch.no_grad(), pytest.raises(RuntimeError, match="simulated"):
        with cache.generation():
            cache.modules[0]._serving_kv_active = True
            cache.ready = True
            raise RuntimeError("simulated request failure")
    assert not cache.active and not cache.ready and not cache.modules[0]._serving_kv_active
    with pytest.raises(RuntimeError, match="no-grad"):
        with cache.generation():
            pass


def test_raw_compaction_keeps_state_and_embeddings_are_request_scoped():
    import threading
    from types import SimpleNamespace

    from openwam.deploy.optimizations.conditional_kv import ConditionalKVCache

    cache = ConditionalKVCache.__new__(ConditionalKVCache)
    cache.architecture = SimpleNamespace(training=False)
    cache.modules = []
    cache.lock = threading.RLock()
    cache.generations = cache.embedding_projections = 0
    cache.active = cache.ready = False
    cache.compact_context = True
    cache.context_inputs = None
    cache.embeddings = {}
    embedding = torch.nn.Linear(16, 8)
    context = torch.randn(1, 65, 16)
    mask = torch.zeros(1, 65, dtype=torch.bool)
    mask[:, :7] = True
    mask[:, -1] = True  # Proprio must survive the gap of masked text padding.
    with torch.no_grad():
        for request in range(2):
            context[:, -1] += 1
            with cache.generation():
                packed = cache.prepare_inputs({"context": context, "context_mask": mask})
                torch.testing.assert_close(packed["context"][:, 7], context[:, -1])
                expected = embedding(packed["context"])
                first = cache.embed_context("video", embedding, packed["context"])
                second = cache.embed_context("video", embedding, packed["context"])
                torch.testing.assert_close(first, expected)
                assert first is second
                assert cache.embedding_projections == request + 1
            assert not cache.embeddings and cache.context_inputs is None


def test_compaction_rejects_malformed_masks():
    import pytest

    from openwam.deploy.optimizations.conditional_kv import compact_padding

    context = torch.randn(1, 7, 32)
    for mask in [
        torch.ones(7, dtype=torch.bool),
        torch.ones(2, 7, dtype=torch.bool),
        torch.ones(1, 1, 1, 1, 7, dtype=torch.bool),
    ]:
        with pytest.raises(ValueError, match="batch-aligned"):
            compact_padding(context, mask)
