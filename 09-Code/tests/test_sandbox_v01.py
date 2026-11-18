"""Sandbox v0.1.0 (Ekdant v0.1.0, spec 04-Model-DeepDive/30): the original
state plus four role generators, one communication round and a learned
selection gate."""

import copy
import json
import math
import os

import pytest
import torch

from ekdant.sandbox import CANDIDATES, SandboxV01

D, HEADS = 16, 2


def _box(rounds=1, seed=0):
    torch.manual_seed(seed)
    return SandboxV01(D, heads=HEADS, hidden_ratio=2.0, rounds=rounds, gate_hidden=8)


def _x(seed=1, B=2, T=5):
    torch.manual_seed(seed)
    return torch.randn(B, T, D)


def _randomise(m, std=0.3, seed=2):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0.0, std)
    return m


def _deltas(out, x):
    return out.trajectories[1:] - x.unsqueeze(0)


def test_sandbox01_is_a_no_op_at_init():
    x = _x()
    out = _box()(x)
    assert torch.equal(out.consensus, torch.zeros_like(x))
    for k in range(5):
        assert torch.equal(out.trajectories[k], x)


def test_sandbox01_shapes():
    out = _box()(_x())
    assert out.consensus.shape == (2, 5, D)
    assert out.trajectories.shape == (5, 2, 5, D)
    assert out.features.shape == (2, 5, 5)
    assert len(CANDIDATES) == 5 and CANDIDATES[0] == "h"


def test_sandbox01_gate_weights_are_float32_and_normalised():
    p = _box()(_x()).features
    assert p.dtype == torch.float32
    assert torch.isfinite(p).all() and (p >= 0).all()
    assert torch.allclose(p.sum(-1), torch.ones(2, 5), atol=1e-6)
    assert torch.allclose(p, torch.full_like(p, 0.2))
    p = _randomise(_box())(_x()).features
    assert torch.allclose(p.sum(-1), torch.ones(2, 5), atol=1e-6)


def test_original_state_is_candidate_zero():
    x = _x()
    out = _randomise(_box())(x)
    assert torch.equal(out.trajectories[0], x)


def test_r1_change_is_orthogonal_to_the_state():
    x = _x()
    out = _randomise(_box(rounds=0))(x)
    d1 = _deltas(out, x)[0]
    assert d1.abs().max() > 1e-3
    assert torch.allclose((d1 * x).sum(-1), torch.zeros(2, 5), atol=1e-4)


def test_four_roles_produce_different_candidates():
    x = _x()
    d = _deltas(_randomise(_box())(x), x).flatten(1)
    d = d / d.norm(dim=-1, keepdim=True)
    cos = d @ d.T
    for i in range(4):
        for j in range(i + 1, 4):
            assert cos[i, j] < 0.99, (i, j)


def test_gradients_reach_every_role_and_the_gate():
    m = _randomise(_box())
    m(_x()).consensus.pow(2).sum().backward()
    groups = m.role_parameters()
    assert set(groups) == set(CANDIDATES[1:]) | {"gate", "comm"}
    for name, ps in groups.items():
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in ps), name


def test_sandbox01_is_causal():
    m = _randomise(_box())
    x = _x()
    y = x.clone()
    y[:, 3:] = torch.randn_like(y[:, 3:])
    a, b = m(x).consensus, m(y).consensus
    assert torch.allclose(a[:, :3], b[:, :3], atol=1e-6)
    assert not torch.allclose(a[:, 3:], b[:, 3:])


def test_forced_candidate_gives_one_hot_weights():
    m = _randomise(_box())
    x = _x()
    m.force = 2
    out = m(x)
    assert torch.equal(out.features, torch.nn.functional.one_hot(
        torch.full((2, 5), 2), 5).float())
    assert torch.allclose(out.consensus, _deltas(out, x)[1], atol=1e-6)


# ---------------------------------------------------------------------------
# Integration into EkdantLM (any depth), config, cache, checkpoints
# ---------------------------------------------------------------------------

from ekdant.backbone import tiny_llama  # noqa: E402
from ekdant.model import EkdantLM, V2Config  # noqa: E402
from ekdant.sandbox import Sandbox  # noqa: E402

V01 = dict(sandbox_version="0.1.0", sandbox_hidden_ratio_v01=2.0,
           sandbox_gate_hidden_v01=8, sandbox_heads=4)


def _lm_pair(layer, layers=3, seed=0, **kw):
    torch.manual_seed(seed)
    lm = tiny_llama(layers=layers)
    base = EkdantLM(V2Config(addon="none", addon_layer=layer), copy.deepcopy(lm)).eval()
    v01 = EkdantLM(V2Config(addon="sandbox", addon_layer=layer, **V01, **kw),
                   copy.deepcopy(lm)).eval()
    return base, v01


def _ids(B=2, T=7, seed=1):
    torch.manual_seed(seed)
    return torch.randint(3, 64, (B, T))


def _train_l2(m, seed=2):
    """Random weights in the add-on only (the backbone stays as loaded)."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in m.addon.parameters():
            p.normal_(0.0, 0.2)
    return m


@pytest.mark.parametrize("layer", [-1, 1, 2])
def test_sandbox01_arm_equals_backbone_at_init(layer):
    base, m = _lm_pair(layer)
    ids = _ids()
    with torch.no_grad():
        assert torch.equal(m(ids).logits, base(ids).logits)


def test_sandbox01_l2_off_equals_backbone():
    base, m = _lm_pair(2)
    _train_l2(m)
    ids = _ids()
    with torch.no_grad(), m.l2_off():
        assert torch.equal(m(ids).logits, base(ids).logits)


@pytest.mark.parametrize("layer", [-1, 2])
def test_sandbox01_changes_the_output_once_trained(layer):
    base, m = _lm_pair(layer)
    _train_l2(m)
    ids = _ids()
    with torch.no_grad():
        assert not torch.allclose(m(ids).logits, base(ids).logits)


def test_sandbox01_mid_change_is_sized_to_the_stream():
    _, m = _lm_pair(2)
    _train_l2(m)
    site = m.addon
    torch.manual_seed(3)
    h = torch.randn(2, 5, 32)
    with torch.no_grad():
        a, _ = site(h)
        b, _ = site(h * 1e3)
    rel = ((b - h * 1e3) - (a - h) * 1e3).norm() / ((a - h) * 1e3).norm()
    assert float(rel) < 1e-4


@pytest.mark.parametrize("layer", [-1, 2])
def test_sandbox01_kv_cache_step_matches_full_forward(layer):
    _, m = _lm_pair(layer)
    _train_l2(m)
    ids = _ids()[:1]
    with torch.no_grad():
        full = m(ids).logits[:, -1]
        pre = m(ids[:, :-1], use_cache=True)
        step = m(ids[:, -1:], past_key_values=pre.aux["past_key_values"],
                 use_cache=True).logits[:, -1]
    assert torch.allclose(full, step, atol=1e-5)


def test_sandbox01_is_causal_through_the_model():
    _, m = _lm_pair(2)
    _train_l2(m)
    ids = _ids()
    other = ids.clone()
    other[:, -1] = (other[:, -1] + 1) % 61 + 3
    with torch.no_grad():
        a, b = m(ids).logits, m(other).logits
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6)


def test_sandbox01_checkpoint_round_trip():
    _, m = _lm_pair(2)
    _train_l2(m)
    m2 = EkdantLM(V2Config.from_dict(m.cfg.to_dict()), tiny_llama(layers=3)).eval()
    m2.load_state_dict(m.state_dict())
    ids = _ids()
    with torch.no_grad():
        assert torch.equal(m(ids).logits, m2(ids).logits)


@pytest.mark.parametrize("bad", [
    dict(sandbox_version="0.2.0"),
    dict(sandbox_version="0.1.0", sandbox_select="hard"),
    dict(sandbox_version="0.1.0", addon_layers=[1, 2]),
    dict(sandbox_version="0.1.0", sandbox_confidence=True, n_sandbox=3),
    dict(sandbox_version="0.1.0", sandbox_aligned=True),
    dict(sandbox_version="0.1.0", sandbox_train_passes=2),
    dict(sandbox_version="0.1.0", gate_force=True),
    dict(sandbox_version="0.1.0", sandbox_rounds_v01=-1),
])
def test_sandbox01_invalid_configs_are_rejected(bad):
    with pytest.raises(ValueError):
        V2Config(addon="sandbox", **bad).validate()


def test_old_sandbox_is_unchanged_by_default():
    cfg = V2Config(addon="sandbox")
    assert cfg.sandbox_version == "0.0.5"
    assert "version" not in cfg.sandbox_kwargs()
    m = EkdantLM(cfg, tiny_llama())
    assert type(m.addon.sandbox) is Sandbox


CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def test_v010_config_serves_both_arms():
    with open(os.path.join(CFG_DIR, "ekdant_v0.1.0.json"), encoding="utf-8") as f:
        raw = json.load(f)
    with open(os.path.join(CFG_DIR, "v5_final.json"), encoding="utf-8") as f:
        v5 = json.load(f)
    assert "addon" not in raw                   # --variant decides the arm
    assert raw["sandbox_version"] == "0.1.0"
    assert set(raw) - set(v5) == {"sandbox_version"}
    # 768: the reasoning targets have a median of 535 tokens; 512 kept 41.4 %
    # of the stage-1 rows, 768 keeps 94.4 % (author's decision, 2026-11-18)
    assert raw["max_answer"] == 768
    for k in (set(raw) & set(v5)) - {"max_answer"}:
        assert raw[k] == v5[k], k
    for arm in ("none", "sandbox"):
        V2Config.from_dict({**V2Config().to_dict(), **raw, "addon": arm})


def test_force_candidate_restores_soft_selection():
    _, m = _lm_pair(2)
    _train_l2(m)
    box = m.addon.sandbox
    with m.force_candidate(3):
        assert box.force == 3
    assert box.force is None
    with pytest.raises(ValueError):
        with m.force_candidate(5):
            pass


# ---------------------------------------------------------------------------
# Diagnostics (spec 8.2), per-role gradients, --addon-layer
# ---------------------------------------------------------------------------

import ekdant.train_gen as G  # noqa: E402


def test_sandbox01_diagnostics_shapes_and_ranges():
    aux = _randomise(_box())(_x()).aux
    assert aux["sel_p"].shape == (5,) and aux["sel_top"].shape == (5,)
    assert math.isclose(float(aux["sel_p"].sum()), 1.0, abs_tol=1e-5)
    assert math.isclose(float(aux["sel_top"].sum()), 1.0, abs_tol=1e-5)
    assert 0.0 <= float(aux["sel_entropy"]) <= math.log(5) + 1e-6
    assert aux["cand_cos"].shape == (5, 5)
    assert torch.allclose(torch.diagonal(aux["cand_cos"]), torch.ones(5), atol=1e-5)
    assert aux["delta_rms"].shape == (4,)
    assert "spread_ratio" not in aux            # no spread floor for 0.1.0


def test_sandbox01_diagnostics_ignore_padding():
    m = _randomise(_box())
    x = _x()
    mask = torch.ones(2, 5, dtype=torch.long)
    a = m(x, mask=mask).aux
    xp = torch.cat([x, 1e3 * torch.randn(2, 3, D)], dim=1)   # padding after
    mp = torch.cat([mask, torch.zeros(2, 3, dtype=torch.long)], dim=1)
    b = m(xp, mask=mp, attn_mask=mp).aux
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-5), k


def test_role_grad_norms_name_every_role():
    _, m = _lm_pair(2)
    _train_l2(m)
    m.train()
    m(_ids()).logits.pow(2).mean().backward()
    g = m.role_grad_norms()
    assert set(g) == set(CANDIDATES[1:]) | {"gate", "comm"}
    assert all(v > 0 for v in g.values())
    base, _ = _lm_pair(2)
    assert base.role_grad_norms() == {}


def test_addon_layer_flag_overrides_the_config():
    cfg = V2Config(addon="sandbox", addon_layer=-1, **V01)
    out = G.apply_addon_layer(cfg, 20)
    assert out.addon_layer == 20 and out.addon_layers == []
    fresh = V2Config(addon="sandbox", addon_layer=-1, **V01)
    assert G.apply_addon_layer(fresh, None).addon_layer == -1
    assert G.build_parser().parse_args(
        ["--variant", "sandbox", "--out", "x", "--addon-layer", "5"]).addon_layer == 5


def test_addon_layer_flag_is_rejected_on_resume_with_a_different_depth():
    cfg = V2Config(addon="sandbox", addon_layer=20, **V01)
    G.check_addon_layer_on_resume(cfg, None)
    G.check_addon_layer_on_resume(cfg, 20)
    with pytest.raises(SystemExit):
        G.check_addon_layer_on_resume(cfg, 10)


# ---------------------------------------------------------------------------
# Role objectives (Phase 5, spec 8.3): R3 correctness, candidate loss, explore
# ---------------------------------------------------------------------------


def _labelled(B=2, T=7):
    ids = _ids(B, T)
    labels = ids.clone()
    labels[:, :2] = -100
    return ids, labels


def _trained_pair(layer=-1, **kw):
    _, m = _lm_pair(layer, **kw)
    _train_l2(m)
    return m.train()


def test_role_objectives_are_off_by_default():
    cfg = V2Config(addon="sandbox", **V01)
    assert (cfg.sandbox_r3_loss, cfg.sandbox_cand_loss, cfg.sandbox_explore) == (0.0, 0.0, 0.0)
    m = _trained_pair()
    assert m.addon.sandbox.r3_probe is None
    ids, labels = _labelled()
    out = m(ids, labels=labels)
    ce = torch.nn.functional.cross_entropy(
        out.logits[:, :-1].reshape(-1, out.logits.size(-1)), labels[:, 1:].reshape(-1),
        ignore_index=-100)
    assert torch.allclose(out.loss, ce, atol=1e-6)
    assert not {"loss_r3", "loss_cand", "explore_entropy"} & set(out.aux)


def test_r3_objective_trains_r3_to_predict_correctness():
    m = _trained_pair(sandbox_r3_loss=0.5)
    ids, labels = _labelled()
    out = m(ids, labels=labels)
    assert "loss_r3" in out.aux and float(out.aux["loss_r3"]) > 0
    out.loss.backward()
    box = m.addon.sandbox
    assert box.r3_probe.weight.grad is not None and box.r3_probe.weight.grad.abs().sum() > 0
    assert box.r3[2].weight.grad.abs().sum() > 0


def test_candidate_objective_needs_end_placement():
    with pytest.raises(ValueError):
        V2Config(addon="sandbox", addon_layer=2, sandbox_cand_loss=0.1, **V01).validate()
    with pytest.raises(ValueError):
        V2Config(addon="sandbox", sandbox_r3_loss=0.1).validate()     # 0.0.5 sandbox
    with pytest.raises(ValueError):
        V2Config(addon="sandbox", sandbox_explore=-1.0, **V01).validate()


def test_candidate_objective_trains_candidates_the_gate_ignores():
    """With the gate forced to h, the next-token loss gives R1-R4 no gradient;
    the candidate loss still does (a role is never starved, spec 8)."""
    ids, labels = _labelled()
    plain = _trained_pair()
    plain.addon.sandbox.force = 0
    plain(ids, labels=labels).loss.backward()
    assert all(float(p.grad.abs().sum()) == 0 for p in plain.addon.sandbox.r1.parameters()
               if p.grad is not None)
    m = _trained_pair(sandbox_cand_loss=0.3)
    m.addon.sandbox.force = 0
    out = m(ids, labels=labels)
    assert "loss_cand" in out.aux
    out.loss.backward()
    for role in (m.addon.sandbox.r1, m.addon.sandbox.r2, m.addon.sandbox.r3, m.addon.sandbox.r4):
        assert role[2].weight.grad.abs().sum() > 0


def test_role_objectives_are_off_in_evaluation():
    m = _trained_pair(sandbox_r3_loss=0.5, sandbox_cand_loss=0.3, sandbox_explore=0.1).eval()
    m.set_explore(0.1)
    ids, labels = _labelled()
    with torch.no_grad():
        out = m(ids, labels=labels)
    assert not {"loss_r3", "loss_cand", "explore_entropy"} & set(out.aux)


def test_explore_bonus_raises_gate_entropy_and_fades_out():
    assert G.explore_weight(0, 0.2, 100) == pytest.approx(0.2)
    assert G.explore_weight(50, 0.2, 100) == pytest.approx(0.1)
    assert G.explore_weight(100, 0.2, 100) == 0.0 and G.explore_weight(500, 0.2, 100) == 0.0
    assert G.explore_weight(0, 0.0, 100) == 0.0
    m = _trained_pair(sandbox_explore=0.2)
    ids, labels = _labelled()
    m.set_explore(0.2)
    out = m(ids, labels=labels)
    ent = out.aux["explore_entropy"]
    m.set_explore(0.0)
    base = m(ids, labels=labels).loss
    assert torch.allclose(out.loss, base - 0.2 * ent, atol=1e-5)


def test_role_objective_flags_are_written_into_the_config():
    args = G.build_parser().parse_args(["--variant", "sandbox", "--out", "x", "--r3-loss", "0.1",
                                        "--cand-loss", "0.2", "--explore", "0.05",
                                        "--explore-steps", "500"])
    cfg = G.apply_role_objectives(V2Config(addon="sandbox", **V01), args)
    assert (cfg.sandbox_r3_loss, cfg.sandbox_cand_loss, cfg.sandbox_explore,
            cfg.sandbox_explore_steps) == (0.1, 0.2, 0.05, 500)
    none = G.build_parser().parse_args(["--variant", "sandbox", "--out", "x"])
    assert G.apply_role_objectives(V2Config(addon="sandbox", **V01), none).sandbox_r3_loss == 0.0
