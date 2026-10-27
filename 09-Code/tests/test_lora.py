"""T41: LoRA on the backbone.  SmolLM2 stays frozen; only small low-rank
adapters on its attention and MLP projections learn (plus all of L2).
A tiny random Llama stands in for SmolLM2."""

import copy
import json

import pytest
import torch

import ekdant.train_gen as G
from ekdant.backbone import tiny_llama
from ekdant.lora import LoRALinear, apply_lora
from ekdant.model import EkdantLM, V2Config, count_parameters

SMALL = dict(debate_rounds=1, sandbox_mlp_ratio=2.0)


def _ids():
    torch.manual_seed(1)
    return torch.randint(3, 64, (2, 7))


def test_lora_is_identity_at_init():
    torch.manual_seed(0)
    lm = tiny_llama(layers=2)
    ref = EkdantLM(V2Config(addon="sandbox", **SMALL), copy.deepcopy(lm)).eval()
    lo = EkdantLM(V2Config(addon="sandbox", lora_r=4, **SMALL), copy.deepcopy(lm)).eval()
    with torch.no_grad():
        assert torch.allclose(lo(_ids()).logits, ref(_ids()).logits, atol=1e-6)


def test_lora_wraps_every_projection_and_freezes_the_backbone():
    lm = tiny_llama(layers=2)
    n = apply_lora(lm, r=4, alpha=8)
    assert n == 2 * 7                      # q k v o gate up down, per layer
    for name, p in lm.named_parameters():
        assert p.requires_grad == ("lora_" in name), name


def test_only_lora_and_l2_are_trained():
    m = EkdantLM(V2Config(addon="sandbox", lora_r=4, **SMALL), tiny_llama(layers=2))
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert all("lora_" in n or n.startswith("addon.") for n in trainable)
    assert any(n.startswith("addon.") for n in trainable)
    groups = m.param_groups(2e-4, 1e-4, gate_lr=1e-3)
    seen = {id(p) for g in groups for p in g["params"]}
    assert seen == {id(p) for p in m.parameters() if p.requires_grad}
    backbone = [p for g in groups if g["kind"] == "backbone" for p in g["params"]]
    names = {id(p): n for n, p in m.named_parameters()}
    assert backbone and all("lora_" in names[id(p)] for p in backbone)


def test_lora_learns_and_frozen_weights_do_not_move():
    torch.manual_seed(0)
    m = EkdantLM(V2Config(addon="none", lora_r=4, **SMALL), tiny_llama(layers=2)).train()
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    opt = torch.optim.AdamW(m.param_groups(1e-2, 1e-2), betas=(0.9, 0.95))
    ids = _ids()
    for _ in range(3):
        opt.zero_grad()
        m(ids, labels=ids.clone()).loss.backward()
        opt.step()
    for n, p in m.named_parameters():
        moved = not torch.equal(p, before[n])
        assert moved == ("lora_b" in n or "lora_a" in n), n


def test_lora_kv_cache_step_matches_full_forward():
    m = EkdantLM(V2Config(addon="none", lora_r=4, **SMALL), tiny_llama(layers=2)).eval()
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, LoRALinear):
                mod.lora_b.normal_(0, 0.1)
    ids = _ids()[:1]
    with torch.no_grad():
        full = m(ids).logits[:, -1]
        pre = m(ids[:, :-1], use_cache=True)
        step = m(ids[:, -1:], past_key_values=pre.aux["past_key_values"],
                 use_cache=True).logits[:, -1]
    assert torch.allclose(full, step, atol=1e-5)


def test_lora_counts_and_off_by_default():
    assert V2Config().lora_r == 0
    plain = EkdantLM(V2Config(addon="none", **SMALL), tiny_llama(layers=2))
    assert not any(isinstance(x, LoRALinear) for x in plain.modules())
    lo = EkdantLM(V2Config(addon="none", lora_r=4, **SMALL), tiny_llama(layers=2))
    assert count_parameters(lo) < count_parameters(lo, trainable_only=False) / 5
    with pytest.raises(ValueError):
        V2Config(lora_r=-1).validate()


def test_lora_run_end_to_end_and_resume(tiny, tmp_path):
    out = tmp_path / "run"
    G.main(_args(tiny, out, extra=["--lora-r", "4", "--lora-alpha", "8"]))
    r = json.loads((out / "results.json").read_text())
    assert r["config"]["lora_r"] == 4 and r["config"]["lora_alpha"] == 8
    assert r["cost"]["params_trainable"] < r["cost"]["params_total"]
    # eval-only rebuilds the same LoRA model from the checkpoint
    G.main(_args(tiny, out, extra=["--lora-r", "4", "--lora-alpha", "8", "--eval-only"]))
    with pytest.raises(SystemExit):     # a resume must keep the LoRA settings
        G.main(_args(tiny, out, extra=["--resume"]))


from test_train_gen import _args, tiny  # noqa: E402,F401  (shared fixture)
