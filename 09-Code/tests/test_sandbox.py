"""Acceptance tests for the sandbox (L2).

Maps to 04-Model-DeepDive/26c_Internal_Sandbox_Build.md section 6
and 04-Model-DeepDive/23_Internal_Build_Plan.md section 7.
"""

import pathlib
import re
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ekdant.sandbox import (  # noqa: E402
    Sandbox,
    mutual_orthogonal,
    spread_gains,
)

BLOCKED = (
    "mamba_ssm",
    "mamba-ssm",
    "causal_conv1d",
    "flash_attn",
    "triton",
    "autogen",
    "crewai",
    "langgraph",
)

DIM, B, T = 64, 2, 16


def make_hT(seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, T, DIM, requires_grad=False)


def make(K=4, rounds=2, steps=8, **kw):
    torch.manual_seed(42)
    return Sandbox(DIM, K=K, rounds=rounds, steps=steps, **kw)


def test_shapes_k1_and_k4():
    hT = make_hT()
    out1 = make(K=1)(hT)
    out4 = make(K=4)(hT)
    assert out1.consensus.shape == (B, T, DIM)
    assert out4.consensus.shape == (B, T, DIM)
    assert out1.trajectories.shape == (1, B, T, DIM)
    assert out4.trajectories.shape == (4, B, T, DIM)


def test_aux_has_three_separately_named_numbers():
    out = make()(make_hT())
    for key in ("anchor_energy", "trajectory_variance", "spread_ratio"):
        assert key in out.aux
    vals = [out.aux[k] for k in
            ("anchor_energy", "trajectory_variance", "spread_ratio")]
    assert all(v.dim() == 0 for v in vals)


def test_gradients_reach_all_k_paths():
    sbx = make(K=4)
    out = sbx(make_hT())
    loss = out.consensus.pow(2).sum() + out.aux["spread_ratio"]
    loss.backward()
    w = sbx.init_proj.weight.grad
    assert w is not None and w.shape == (4, DIM, DIM)
    per_k = w.abs().mean(dim=(1, 2))
    assert (per_k > 0).all(), f"dead trajectory: {per_k}"
    assert sbx.debate[0].mlp[0].weight.grad is not None


def test_orthogonal_init_is_truly_orthogonal():
    M = mutual_orthogonal(4, 16)
    G = M.flatten(1) @ M.flatten(1).T
    off = G - torch.eye(4)
    assert off.abs().max().item() < 1e-5


def test_spread_gains_span_the_range():
    g = spread_gains(4, 0.8, 1.25)
    assert g[0].item() == pytest.approx(0.8, rel=1e-3)
    assert g[-1].item() == pytest.approx(1.25, rel=1e-3)
    assert g[0] < g[1] < g[2] < g[3]


def test_readout_moves_descent_changes_consensus():
    sbx = make(K=4, steps=8)
    hT = make_hT()
    sbx.steps = 0
    c0 = sbx(hT).consensus.detach()
    sbx.steps = 8
    c8 = sbx(hT).consensus.detach()
    delta = (c8 - c0).norm().item()
    assert delta > 0, "inert peer-mean defect reproduced"
    assert (c8 != c0).any()


def test_peer_mean_target_is_a_noop():
    torch.manual_seed(7)
    traj0 = torch.randn(4, B, T, DIM)
    for _ in range(8):
        traj0 = traj0.mean(dim=0, keepdim=True).expand_as(traj0)
    assert torch.allclose(traj0, traj0.mean(0, keepdim=True).expand_as(traj0))


def _gates(sbx, hT):
    traj = sbx.init_proj(hT)
    for layer in sbx.debate:
        traj = layer(traj)
    target = sbx.descent.anchor(hT).unsqueeze(0).expand_as(traj)
    return torch.sigmoid(sbx.descent.gate(torch.cat([traj, target], dim=-1)))


def test_gate_is_per_trajectory_and_per_channel():
    sbx = make(K=4, steps=8)
    g = _gates(sbx, make_hT()).detach()
    assert g.shape == (4, B, T, DIM)

    across_k = g.std(dim=0).mean().item()
    across_channels = g.std(dim=-1).mean().item()
    assert across_k > 1e-3, f"gate shared across trajectories: {across_k}"
    assert across_channels > 1e-3, f"gate reduced to a scalar: {across_channels}"

    flat = g.flatten()
    assert (flat > 0).all() and (flat < 1).all()


def test_gates_are_not_identical_across_k():
    sbx = make(K=4, steps=8)
    g = _gates(sbx, make_hT()).detach()
    for k in range(1, 4):
        delta = (g[k] - g[0]).abs().mean().item()
        assert delta > 1e-4, f"trajectory {k} shares trajectory 0's gate"


def test_retention_differs_across_trajectories():
    sbx = make(K=4, steps=8)
    ret = sbx(make_hT()).aux["per_trajectory_retention"]
    assert (ret > 0.1).all(), f"trajectories collapsed: {ret}"
    assert (ret.max() - ret.min()).item() > 1e-6, "identical retention => shared gate"


def test_spread_floor_is_zero_when_healthy():
    sbx = make(K=4)
    rho = sbx(make_hT()).aux["spread_ratio"]
    assert rho.item() > sbx.rho_min, f"unexpectedly collapsed rho={rho}"
    assert sbx.spread_floor(rho).item() == 0.0


def test_spread_ratio_is_order_one_at_init():
    sbx = make(K=4, steps=8)
    rho = sbx(make_hT()).aux["spread_ratio"].item()
    assert 0.05 < rho < 10.0, f"rho out of sane range: {rho}"


def test_spread_floor_fires_below_threshold():
    sbx = make(K=4)
    tiny = torch.tensor(1e-6)
    assert sbx.spread_floor(tiny).item() > 0
    assert sbx.spread_floor(torch.tensor(0.05)).item() == 0.0
    assert sbx.spread_floor(torch.tensor(1.95)).item() == 0.0


def test_floor_cannot_dominate():
    sbx = make(K=4)
    worst = sbx.spread_floor(torch.tensor(0.0)).item()
    assert worst == pytest.approx(sbx.rho_min ** 2)
    assert worst < 0.01


def test_s0_vs_s8_uses_identical_weights():
    sbx = make(K=4, steps=8)
    w_before = sbx.init_proj.weight.detach().clone()
    sbx.steps = 0
    sbx(make_hT())
    sbx.steps = 8
    sbx(make_hT())
    assert torch.equal(sbx.init_proj.weight.detach(), w_before)


def test_k1_runs_without_debate_partner():
    out = make(K=1, rounds=2, steps=8)(make_hT())
    assert out.trajectories.shape[0] == 1
    assert torch.isfinite(out.consensus).all()


def test_lowest_energy_readout_is_one_of_the_trajectories():
    sbx = make(K=4, readout="lowest_energy")
    out = sbx(make_hT())
    d = (out.trajectories - out.consensus.unsqueeze(0)).pow(2).sum(-1)
    assert d.min().item() < 1e-10


def test_lowest_energy_readout_is_per_position_and_causal():
    # The pick at (b, t) must depend only on (b, t): changing a later token
    # or another batch row must not change it (it used to pick one
    # trajectory from the batch- and sequence-averaged energy).
    sbx = make(K=4, readout="lowest_energy")
    hT = make_hT()
    base = sbx(hT).consensus
    hT2 = hT.clone()
    hT2[0, -1] += 5.0          # a future token of row 0
    hT2[1:] += 3.0             # every other batch row
    out = sbx(hT2).consensus
    assert torch.allclose(out[0, :-1], base[0, :-1], atol=1e-6)
    d = (sbx(hT).trajectories - base.unsqueeze(0)).pow(2).sum(-1)
    assert (d.min(dim=0).values < 1e-10).all(), "each position is one trajectory"


def test_mask_restricts_diagnostics_to_real_tokens():
    # Every sandbox op is per position, so the masked statistics of a
    # padded batch must equal the plain statistics of its real tokens.
    sbx = make(K=4)
    hT = make_hT()
    mask = torch.ones(B, T)
    mask[1, T // 2:] = 0
    hT[1, T // 2:] = 50.0      # padding garbage
    padded = sbx(hT, mask=mask).aux
    real = hT[mask.bool()].unsqueeze(0)          # [1, N, H]
    ref = sbx(real).aux
    for k in ("spread_ratio", "trajectory_variance", "anchor_energy"):
        assert torch.allclose(padded[k], ref[k], rtol=1e-5), k
    unmasked = sbx(hT).aux["spread_ratio"]
    assert not torch.allclose(unmasked, ref["spread_ratio"], rtol=1e-3)


def test_loss_decreases_over_training_steps():
    torch.manual_seed(0)
    sbx = make(K=4, steps=8)
    hT = make_hT(seed=1)
    target = torch.randn(B, T, DIM)
    opt = torch.optim.Adam(sbx.parameters(), lr=1e-3)
    losses = []
    for _ in range(10):
        opt.zero_grad()
        out = sbx(hT)
        loss = (out.consensus - target).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"no decrease: {losses[0]} -> {losses[-1]}"


def test_no_blocked_imports_in_ekdant_package():
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "ekdant"
    assert root.is_dir()
    for path in root.rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        for name in BLOCKED:
            pattern = r"(?:import|from)\s+" + re.escape(name).replace(r"\-", r"[\-_]")
            assert not re.search(pattern, src), f"{path.name} imports {name}"


def _consensus_growth(norm_inputs, scale=3.0):
    """RMS of the consensus after scaling every debate weight by `scale`,
    relative to before (same input, same init)."""
    from ekdant.sandbox import Sandbox
    torch.manual_seed(0)
    h = torch.randn(2, 6, 32)
    sbx = Sandbox(32, K=4, rounds=4, steps=8, mlp_ratio=4.0,
                  norm_inputs=norm_inputs).eval()
    with torch.no_grad():
        before = sbx(h).consensus.pow(2).mean().sqrt()
        for rnd in sbx.debate:
            for mod in rnd.modules():
                if isinstance(mod, torch.nn.Linear):
                    mod.weight.mul_(scale)
        after = sbx(h).consensus.pow(2).mean().sqrt()
    return float(after / before)


def test_prenorm_debate_stops_size_compounding_across_rounds():
    # T33: with no normalisation inside, the debate rounds chain ~14 learned
    # matrices, so a small weight growth compounds round after round; in
    # training the consensus rms went 0.7 -> 36,560 in 180 steps (fp32
    # probe) and overflowed in bf16 on the GPU. Pre-norm makes every round
    # read a fixed-size input, so growth stays proportional, not compounded.
    compounded = _consensus_growth(norm_inputs=False)
    bounded = _consensus_growth(norm_inputs=True)
    assert bounded < 100
    assert compounded > 10 * bounded
