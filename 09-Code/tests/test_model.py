"""Tests for the assembled model and the A/B configuration."""

import json
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ekdant.model import (  # noqa: E402
    EkdantConfig,
    EkdantModel,
    baseline_config,
    count_parameters,
    estimate_model_size,
)
from ekdant.output_head import OutputHead  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
B, T = 2, 48


def tiny(use_sandbox=True, **kw):
    d = dict(vocab_size=512, dim=64, n_t=2,
             n_heads=4, n_kv_heads=2, window=16, ffn_hidden=128,
             max_seq_len=128, use_sandbox=use_sandbox,
             n_sandbox=3, debate_rounds=2, descent_steps=4, dropout=0.0)
    d.update(kw)
    return EkdantConfig(**d)


def run(cfg, ids=None, labels=None):
    torch.manual_seed(0)
    m = EkdantModel(cfg)
    ids = ids if ids is not None else torch.randint(1, 500, (B, T))
    return m, m(ids, labels=ids if labels is None else labels)


def test_forward_shapes_sandbox():
    m, out = run(tiny(True))
    assert out.logits.shape == (B, T, 512)
    assert out.loss is not None and torch.isfinite(out.loss)


def test_forward_shapes_baseline():
    m, out = run(tiny(False))
    assert out.logits.shape == (B, T, 512)
    assert out.aux == {}


def test_sandbox_emits_three_separately_named_numbers():
    _, out = run(tiny(True))
    for key in ("anchor_energy", "trajectory_variance", "spread_ratio"):
        assert key in out.aux, f"missing {key}"
    assert out.aux["anchor_energy"].dim() == 0
    assert out.aux["trajectory_variance"].dim() == 0
    assert out.aux["spread_ratio"].dim() == 0
    vals = [out.aux[k].item() for k in
            ("anchor_energy", "trajectory_variance", "spread_ratio")]
    assert len(set(vals)) == 3, "aux numbers collapsed onto one another"


def test_head_is_tied_adds_no_parameters():
    torch.manual_seed(0)
    emb = torch.nn.Embedding(100, 32)
    head = OutputHead(emb)
    assert head.num_parameters == 0
    assert head.weight.data_ptr() == emb.weight.data_ptr()
    assert head(torch.randn(2, 5, 32)).shape == (2, 5, 100)


def test_use_sandbox_is_the_only_difference():
    a, b = tiny(True), tiny(False)
    sa, sb = estimate_model_size(a), estimate_model_size(b)
    assert sa["t"] == sb["t"]
    assert sa["embedding_tied"] == sb["embedding_tied"]
    assert sa["sandbox"] > 0 and sb["sandbox"] == 0
    assert sa["total"] - sb["total"] == sa["sandbox"]


def test_baseline_config_flips_the_flag():
    c = baseline_config(tiny(True))
    assert c.use_sandbox is False
    assert c.dim == 512 // 2 or c.dim == 64


def test_floor_penalty_zero_when_healthy():
    cfg = tiny(True, rho_min=0.05, lambda_sbx=1.0)
    m = EkdantModel(cfg)
    torch.manual_seed(0)
    ids = torch.randint(1, 500, (B, T))
    hT = m.t(m.embed(ids))
    rho = m.sandbox(hT).aux["spread_ratio"]
    assert rho.item() > cfg.rho_min
    assert m._floor_penalty({"spread_ratio": rho}).item() == 0.0


def test_floor_penalty_fires_when_collapsed():
    cfg = tiny(True, rho_min=0.5, lambda_sbx=1.0)
    m = EkdantModel(cfg)
    pen = m._floor_penalty({"spread_ratio": torch.tensor(0.05)})
    assert abs(pen.item() - 0.45 ** 2) < 1e-6


def test_gradients_reach_backbone_in_both_variants():
    for use in (True, False):
        m, out = run(tiny(use))
        out.loss.backward()
        assert m.embed.weight.grad is not None
        assert m.t.blocks[0].attn.q_proj.weight.grad is not None
        assert m.t.blocks[0].ffn.down.weight.grad is not None


def test_rho_is_healthy_at_init_in_the_real_model():
    """Guards the eta=0.5 bug: a flat eta crushed rho to 0.01 at init."""
    torch.manual_seed(0)
    cfg = tiny(True, n_sandbox=4, descent_steps=8, eta=0.5 / 8)
    m = EkdantModel(cfg)
    hT = m.t(m.embed(torch.randint(1, 500, (B, T))))
    rho = m.sandbox(hT).aux["spread_ratio"].item()
    assert 0.05 < rho < 10.0, f"rho out of sane range at init: {rho}"


def test_shipped_sandbox_model_is_healthy_at_init():
    with open(ROOT / "configs" / "sandbox_50m.json", "r",
              encoding="utf-8") as f:
        cfg = EkdantConfig.from_dict(json.load(f))
    torch.manual_seed(0)
    m = EkdantModel(cfg)
    hT = m.t(m.embed(torch.randint(1, 500, (2, 64))))
    rho = m.sandbox(hT).aux["spread_ratio"].item()
    assert 0.05 < rho < 10.0, f"shipped config rho={rho}"


def test_loss_decreases_sandbox():
    torch.manual_seed(0)
    cfg = tiny(True)
    m = EkdantModel(cfg)
    ids = torch.randint(1, 500, (B, T))
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    first, last = None, None
    for i in range(12):
        opt.zero_grad()
        loss = m(ids, labels=ids).loss
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    assert last < first


def test_loss_decreases_baseline():
    torch.manual_seed(0)
    cfg = tiny(False)
    m = EkdantModel(cfg)
    ids = torch.randint(1, 500, (B, T))
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    first, last = None, None
    for i in range(12):
        opt.zero_grad()
        loss = m(ids, labels=ids).loss
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    assert last < first


def test_score_options_shape():
    m = EkdantModel(tiny(False))
    ids = torch.randint(1, 500, (B * 4, 32))
    mask = torch.ones_like(ids)
    s = m.score_options(ids, mask, n_options=4)
    assert s.shape == (B, 4)


def test_answer_weight_adds_scaled_answer_ce():
    torch.manual_seed(0)
    m = EkdantModel(tiny(False))
    ids = torch.randint(1, 500, (B, T))
    answer_pos = torch.tensor([10, 20])
    answer_ids = torch.tensor([3, 7])
    with torch.no_grad():
        base = m(ids, labels=ids, answer_weight=0.0).loss
        w5 = m(ids, labels=ids, answer_pos=answer_pos,
               answer_ids=answer_ids, answer_weight=5.0).loss
        logits = m(ids, return_aux=False).logits
        ans_ce = torch.nn.functional.cross_entropy(
            logits[torch.arange(B), answer_pos - 1], answer_ids)
    assert torch.isfinite(w5) and w5 > base
    # loss == lm_ce + weight * ce(logits[answer_pos-1], answer_id)
    assert abs(float(w5 - base) - 5.0 * float(ans_ce)) < 1e-4


def test_answer_weight_zero_matches_legacy_call():
    torch.manual_seed(0)
    m = EkdantModel(tiny(False))
    ids = torch.randint(1, 500, (B, T))
    with torch.no_grad():
        legacy = m(ids, labels=ids).loss
        explicit = m(ids, labels=ids,
                     answer_pos=torch.tensor([5, 6]),
                     answer_ids=torch.tensor([1, 2]),
                     answer_weight=0.0).loss
    assert torch.equal(legacy, explicit)


def test_answer_weight_gradients_reach_answer_position():
    torch.manual_seed(0)
    m = EkdantModel(tiny(False))
    ids = torch.randint(1, 500, (B, T))
    answer_pos = torch.tensor([10, 20])
    answer_ids = torch.tensor([3, 7])
    m.zero_grad()
    m(ids, labels=ids, answer_pos=answer_pos,
      answer_ids=answer_ids, answer_weight=10.0).loss.backward()
    # the term supervises the logits that predict the letter at p-1
    assert m.head.weight.grad is not None
    assert float(m.head.weight.grad.abs().sum()) > 0


def test_config_validation():
    with pytest.raises(ValueError):
        EkdantConfig(dim=64, n_heads=5).validate()
    with pytest.raises(ValueError):
        EkdantConfig(n_heads=8, n_kv_heads=3).validate()
    with pytest.raises(ValueError):
        EkdantConfig(readout="median").validate()


def test_eta_defaults_to_half_over_steps():
    c = EkdantConfig(descent_steps=8).validate()
    assert abs(c.eta - 0.5 / 8) < 1e-9
    c = EkdantConfig(descent_steps=4).validate()
    assert abs(c.eta - 0.125) < 1e-9


def test_eta_in_shipped_configs_matches_half_over_steps():
    for name in ("sandbox_50m.json", "baseline_50m.json"):
        with open(ROOT / "configs" / name, "r", encoding="utf-8") as f:
            cfg = EkdantConfig.from_dict(json.load(f))
        assert abs(cfg.eta - 0.5 / cfg.descent_steps) < 1e-9, (
            f"{name}: eta {cfg.eta} must be 0.5/{cfg.descent_steps}"
        )


def test_shipped_configs_are_consistent():
    for name, expect in (("sandbox_50m.json", True), ("baseline_50m.json", False)):
        path = ROOT / "configs" / name
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cfg = EkdantConfig.from_dict(raw)
        assert cfg.use_sandbox is expect
        assert cfg.dim % cfg.n_heads == 0
        assert cfg.n_heads % cfg.n_kv_heads == 0


def test_shipped_configs_sizes():
    with open(ROOT / "configs" / "sandbox_50m.json", "r",
              encoding="utf-8") as f:
        sbx = estimate_model_size(EkdantConfig.from_dict(json.load(f)))
    with open(ROOT / "configs" / "baseline_50m.json", "r",
              encoding="utf-8") as f:
        base = estimate_model_size(EkdantConfig.from_dict(json.load(f)))
    assert 48e6 < sbx["total"] < 52e6, f"sandbox {sbx['total']/1e6:.1f}M"
    assert 42e6 < base["total"] < 46e6, f"baseline {base['total']/1e6:.1f}M"
    assert sbx["total"] - base["total"] == sbx["sandbox"], (
        "the parameter delta must be exactly the L2 layer"
    )


def test_ab_is_depth_matched():
    """Same backbone in both variants; the only difference is L2."""
    with open(ROOT / "configs" / "sandbox_50m.json", "r",
              encoding="utf-8") as f:
        scfg = EkdantConfig.from_dict(json.load(f))
    with open(ROOT / "configs" / "baseline_50m.json", "r",
              encoding="utf-8") as f:
        bcfg = EkdantConfig.from_dict(json.load(f))
    assert scfg.use_sandbox and not bcfg.use_sandbox
    for knob in ("dim", "n_t", "n_heads", "n_kv_heads",
                 "window", "ffn_hidden", "vocab_size", "max_seq_len",
                 "dropout"):
        assert getattr(scfg, knob) == getattr(bcfg, knob), (
            f"{knob} differs: {getattr(scfg, knob)} vs {getattr(bcfg, knob)}"
        )
    sbx = estimate_model_size(scfg)
    base = estimate_model_size(bcfg)
    assert sbx["t"] == base["t"], "T depth/budget must be identical"
    assert sbx["embedding_tied"] == base["embedding_tied"]
    assert sbx["sandbox"] > 0 and base["sandbox"] == 0
    assert sbx["total"] - base["total"] == sbx["sandbox"]


def test_pad_row_is_zeroed():
    cfg = tiny(True)
    m = EkdantModel(cfg)
    assert m.embed.weight[cfg.pad_id].abs().sum().item() == 0.0


def test_model_has_no_blocked_imports():
    import ekdant.model as mod
    src = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
    for banned in ("mamba_ssm", "flash_attn", "triton", "autogen", "crewai"):
        assert banned not in src
