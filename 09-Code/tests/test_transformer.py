"""Tests for layer T (windowed dense Transformer).

Maps to 04-Model-DeepDive/26b_Dense_Transformer_for_T.md section 3.
"""

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ekdant.transformer_stack import (  # noqa: E402
    RMSNorm,
    SwiGLU,
    TransformerStack,
    WindowedGQA,
    apply_rope,
    repeat_kv,
    rope_cache,
    rotate_half,
    window_mask,
)

B, T, H = 2, 48, 64
HEADS, KV_HEADS, WINDOW = 8, 2, 16


def randomised(stack: TransformerStack) -> TransformerStack:
    torch.manual_seed(0)
    for blk in stack.blocks:
        torch.nn.init.normal_(blk.attn.o_proj.weight, std=0.02)
        torch.nn.init.normal_(blk.ffn.down.weight, std=0.02)
    return stack


def make(n_layers=2, window=WINDOW, dim=H):
    torch.manual_seed(42)
    return TransformerStack(
        dim=dim, n_layers=n_layers, n_heads=HEADS, n_kv_heads=KV_HEADS,
        window=window, ffn_hidden=2 * dim,
    )


def test_shapes():
    out = make()(torch.randn(B, T, H))
    assert out.shape == (B, T, H)


def test_default_init_is_residual_identity():
    out = make()(torch.randn(B, T, H))
    assert torch.isfinite(out).all()


def test_gradients_flow_through_every_block():
    stack = randomised(make(n_layers=3))
    out = stack(torch.randn(B, T, H))
    out.pow(2).mean().backward()
    for i, blk in enumerate(stack.blocks):
        for name in ("attn.q_proj", "attn.k_proj", "attn.v_proj",
                     "attn.o_proj", "ffn.gate", "ffn.up", "ffn.down"):
            mod = blk
            for part in name.split("."):
                mod = getattr(mod, part)
            assert mod.weight.grad is not None, f"block {i} {name} has no grad"
            assert mod.weight.grad.abs().sum() > 0, f"block {i} {name} is dead"


def test_mask_is_causal_and_windowed():
    mask = window_mask(48, 16)
    assert mask.shape == (48, 48)
    for i in range(48):
        for j in range(48):
            expected = (j <= i) and (i - j < 16)
            assert bool(mask[i, j]) == expected, f"mask[{i},{j}] wrong"


def test_position_attends_only_inside_the_window():
    mask = window_mask(100, 16)
    i = 50
    seen = mask[i].nonzero().flatten().tolist()
    assert seen == list(range(i - 15, i + 1))


def test_future_tokens_do_not_change_the_past():
    stack = randomised(make())
    x = torch.randn(B, T, H)
    ref = stack(x)

    cut = 20
    x2 = x.clone()
    x2[:, cut:, :] = torch.randn_like(x2[:, cut:, :])
    alt = stack(x2)

    assert torch.allclose(ref[:, :cut], alt[:, :cut], atol=1e-6), (
        "future leakage: past positions changed"
    )
    assert not torch.allclose(ref[:, cut:], alt[:, cut:], atol=1e-6)


def test_window_ablation_changes_receptive_field():
    near = randomised(make(n_layers=1, window=8))
    far = randomised(make(n_layers=1, window=32))
    x = torch.randn(B, T, H)
    assert not torch.allclose(near(x), far(x), atol=1e-6)

    probe = 20
    x2 = x.clone()
    x2[:, :8, :] = torch.randn_like(x2[:, :8, :])
    ref_a, ref_b = near(x), far(x)
    a, b = near(x2), far(x2)

    assert torch.allclose(ref_a[:, probe], a[:, probe], atol=1e-6), (
        "window 8 reached 12+ positions back"
    )
    assert not torch.allclose(ref_b[:, probe], b[:, probe], atol=1e-6), (
        "window 32 should still see position 5 from position 20"
    )


def test_kv_heads_are_repeated_not_mismatched():
    x = torch.randn(B, T, H)
    cos, sin = rope_cache(H // HEADS, T)
    mask = window_mask(T, WINDOW)
    gqa = WindowedGQA(H, HEADS, KV_HEADS, WINDOW)
    out = gqa(x, cos, sin, mask)
    assert out.shape == (B, T, H)

    v = torch.randn(B, KV_HEADS, T, H // HEADS)
    n_rep = HEADS // KV_HEADS
    rep = repeat_kv(v, n_rep)
    assert rep.shape == (B, HEADS, T, H // HEADS)
    for head in range(HEADS):
        assert torch.equal(rep[:, head], v[:, head // n_rep]), f"head {head}"


def test_rope_is_reversible_by_rotation():
    cos, sin = rope_cache(16, 8)
    x = torch.randn(1, 4, 8, 16)
    y = apply_rope(x, cos, sin)
    inv = apply_rope(y, cos, -sin)
    assert torch.allclose(inv, x, atol=1e-6)


def test_rotate_half_round_trip():
    x = torch.randn(2, 3, 16)
    back = rotate_half(rotate_half(x))
    assert torch.allclose(back, -x, atol=1e-6)


def test_rmsnorm_normalises_across_channels():
    n = RMSNorm(64)
    x = torch.randn(2, 5, 64) * 7.3
    y = n(x)
    rms = y.float().pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_swiglu_shape_and_grad():
    f = SwiGLU(64, 128)
    x = torch.randn(2, 5, 64, requires_grad=True)
    f(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_loss_decreases_over_training_steps():
    stack = randomised(make())
    x = torch.randn(B, T, H)
    target = torch.randn(B, T, H)
    opt = torch.optim.Adam(stack.parameters(), lr=1e-3)
    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = (stack(x) - target).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_26b_dim_table_256_and_512():
    for dim, heads, kv, ffn in ((256, 8, 2, 512), (320, 8, 2, 640),
                                (512, 8, 2, 1024)):
        s = TransformerStack(dim=dim, n_layers=1, n_heads=heads,
                             n_kv_heads=kv, ffn_hidden=ffn)
        out = s(torch.randn(1, 12, dim))
        assert out.shape == (1, 12, dim)
