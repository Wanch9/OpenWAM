import pytest
import torch

from openwam.model.ops.fused_norm_rope import can_fuse, fused_norm_rope, torch_norm_rope


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape,heads", [((2, 7, 96), 3), ((1, 32, 3072), 24), ((1, 360, 3072), 24)])
def test_kernel_matches_reference(dtype, shape, heads):
    torch.manual_seed(71)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    w = torch.randn(shape[-1], device="cuda", dtype=dtype)
    theta = torch.randn(shape[1], shape[-1] // heads // 2, device="cuda", dtype=torch.float64)
    freqs = torch.polar(torch.ones_like(theta), theta)
    with torch.no_grad():
        assert can_fuse(x, w, freqs, heads)
        pure = fused_norm_rope(x, w, freqs, 1e-6, heads, False)
        expected_pure = torch_norm_rope(x, w, freqs, 1e-6, heads, False)
        assert torch.equal(pure, expected_pure)
        actual = fused_norm_rope(x, w, freqs, 1e-6, heads)
        expected = torch_norm_rope(x, w, freqs, 1e-6, heads)
        tol = 1e-5 if dtype == torch.float32 else (0.002 if dtype == torch.float16 else 0.01)
        torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)


def test_cpu_and_gradient_fallback():
    x = torch.randn(2, 7, 96, requires_grad=True)
    w = torch.randn(96, requires_grad=True)
    freqs = torch.polar(torch.ones(7, 16, dtype=torch.float64), torch.randn(7, 16, dtype=torch.float64))
    assert not can_fuse(x, w, freqs, 3)
    actual = fused_norm_rope(x, w, freqs, 1e-5, 3)
    expected = torch_norm_rope(x, w, freqs, 1e-5, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_strided_fallback_and_graph_replay_sees_new_inputs():
    x = torch.randn(1, 9, 96, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(96, device="cuda", dtype=torch.bfloat16)
    theta = torch.randn(9, 16, device="cuda", dtype=torch.float64)
    f = torch.polar(torch.ones_like(theta), theta)
    with torch.no_grad():
        strided = torch.randn(1, 9, 192, device="cuda", dtype=x.dtype)[..., ::2]
        assert not can_fuse(strided, w, f, 3)
        torch.testing.assert_close(
            fused_norm_rope(strided, w, f, 1e-6, 3), torch_norm_rope(strided, w, f, 1e-6, 3), rtol=0, atol=0
        )
        for _ in range(3):
            fused_norm_rope(x, w, f, 1e-6, 3)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fused_norm_rope(x, w, f, 1e-6, 3)
        x.add_(0.25)
        w.mul_(0.75)
        f.copy_(f.conj().clone())
        graph.replay()
        torch.testing.assert_close(out, torch_norm_rope(x, w, f, 1e-6, 3), rtol=0.01, atol=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compile_and_parameter_update():
    x = torch.randn(1, 9, 96, device="cuda", dtype=torch.bfloat16)
    w = torch.nn.Parameter(torch.randn(96, device="cuda", dtype=x.dtype))
    theta = torch.randn(9, 16, device="cuda", dtype=torch.float64)
    f = torch.polar(torch.ones_like(theta), theta)
    compiled = torch.compile(lambda value: fused_norm_rope(value, w, f, 1e-6, 3), mode="reduce-overhead")
    with torch.no_grad():
        for i in range(3):
            torch.compiler.cudagraph_mark_step_begin()
            x.add_(0.1)
            w.add_(0.01)
            actual = compiled(x)
            torch.testing.assert_close(actual, torch_norm_rope(x, w, f, 1e-6, 3), rtol=0.01, atol=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_complex64_frequencies():
    x = torch.randn(2, 9, 96, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(96, device="cuda", dtype=x.dtype)
    theta = torch.randn(9, 16, device="cuda")
    f = torch.polar(torch.ones_like(theta), theta)
    with torch.no_grad():
        assert can_fuse(x, w, f, 3)
        torch.testing.assert_close(
            fused_norm_rope(x, w, f, 1e-6, 3), torch_norm_rope(x, w, f, 1e-6, 3), rtol=0.01, atol=0.01
        )


def test_engine_reinitialization_clears_operator_mode():
    from omegaconf import OmegaConf

    from openwam.deploy.engine import JointInferenceEngine

    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    model[0]._serving_rope_mode = "triton_norm_rope"
    before = set(model.state_dict())
    engine = JointInferenceEngine.__new__(JointInferenceEngine)
    engine.cfg = OmegaConf.create({"optimization": {"rope_kernel": "torch"}})
    engine.architecture = model
    engine._init_optimizations()
    assert model[0]._serving_rope_mode == "torch"
    assert set(model.state_dict()) == before
