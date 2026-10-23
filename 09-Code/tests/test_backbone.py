"""v2 model: pretrained backbone + gated add-on (spec 07-Management/30 §3).
A tiny random Llama stands in for SmolLM2-135M, so nothing is downloaded."""

import copy
import json
import os

import torch
import torch.nn.functional as F

from ekdant.backbone import tiny_llama
from ekdant.model import EkdantLM, V2Config

SMALL = dict(debate_rounds=1, sandbox_mlp_ratio=2.0)


def _arms(seed=0):
    torch.manual_seed(seed)
    lm = tiny_llama()
    return {a: EkdantLM(V2Config(addon=a, **SMALL), copy.deepcopy(lm)).eval()
            for a in ("none", "sandbox")}


def _batch():
    torch.manual_seed(1)
    ids = torch.randint(3, 64, (2, 7))
    mask = torch.ones(2, 7, dtype=torch.long)
    mask[1, 5:] = 0
    return ids, mask


def test_all_arms_equal_backbone_at_init():
    ids, mask = _batch()
    arms = _arms()
    ref = arms["none"](ids, attention_mask=mask).logits
    assert torch.equal(arms["sandbox"](ids, attention_mask=mask).logits, ref)


def test_loss_counts_answer_tokens_only():
    ids, mask = _batch()
    m = _arms()["none"]
    labels = torch.full_like(ids, -100)
    labels[0, 3:6] = ids[0, 3:6]
    out = m(ids, attention_mask=mask, labels=labels)
    lg = out.logits[0, 2:5]                     # logits[t] predicts token t+1
    manual = F.cross_entropy(lg, ids[0, 3:6])
    assert torch.allclose(out.loss, manual, atol=1e-6)


def test_kv_cache_step_matches_full_forward_with_sandbox():
    m = _arms()["sandbox"]
    with torch.no_grad():
        m.addon.gate.fill_(0.5)
    ids, _ = _batch()
    ids = ids[:1]
    with torch.no_grad():
        full = m(ids).logits[:, -1]
        pre = m(ids[:, :-1], use_cache=True)
        step = m(ids[:, -1:], past_key_values=pre.aux["past_key_values"],
                 use_cache=True).logits[:, -1]
    assert torch.allclose(full, step, atol=1e-5)


def test_param_groups_split_lrs():
    m = _arms()["sandbox"]
    groups = m.param_groups(5e-5, 1e-3)
    seen = [id(p) for g in groups for p in g["params"]]
    assert sorted(seen) == sorted(id(p) for p in m.parameters())
    addon_ids = {id(p) for p in m.addon.parameters()}
    for g in groups:
        for p in g["params"]:
            assert g["lr"] == (1e-3 if id(p) in addon_ids else 5e-5)
    gate_group = [g for g in groups if any(p is m.addon.gate for p in g["params"])]
    assert gate_group[0]["weight_decay"] == 0.0


def test_v2_config_round_trip():
    cfg = V2Config(addon="sandbox")
    assert V2Config.from_dict(cfg.to_dict()) == cfg
    for arm in ("baseline", "sandbox"):
        with open(f"configs/v2_{arm}.json", encoding="utf-8") as f:
            shipped = V2Config.from_dict(json.load(f))
        assert shipped.backbone == "HuggingFaceTB/SmolLM2-135M"
        assert (shipped.debate_rounds, shipped.sandbox_mlp_ratio) == (4, 4.0)


def test_shipped_v2_configs_differ_only_in_addon():
    # fairness: the two arms' config files may differ in nothing but L2
    a = json.load(open("configs/v2_baseline.json", encoding="utf-8"))
    b = json.load(open("configs/v2_sandbox.json", encoding="utf-8"))
    assert (a.pop("addon"), b.pop("addon")) == ("none", "sandbox")
    assert a == b
    # the config names a git-ignored token FILE; it never holds a token
    assert a["hf_token_file"] == "configs/hf_token.txt"
    assert not any(isinstance(v, str) and v.startswith("hf_") for v in a.values())


def test_hf_token_env_wins_then_file(tmp_path, monkeypatch):
    from ekdant.backbone import apply_hf_token
    f = tmp_path / "hf_token.txt"
    f.write_text("  hf_fromfile\n")
    monkeypatch.setenv("HF_TOKEN", "hf_fromenv")
    assert apply_hf_token(str(f)) == "env"
    assert os.environ["HF_TOKEN"] == "hf_fromenv"
    monkeypatch.delenv("HF_TOKEN")
    assert apply_hf_token(str(f)) == "file"
    assert os.environ["HF_TOKEN"] == "hf_fromfile"
    monkeypatch.delenv("HF_TOKEN")
    assert apply_hf_token(str(tmp_path / "missing.txt")) == "none"
    assert "HF_TOKEN" not in os.environ


def test_load_backbone_returns_float32(monkeypatch):
    # transformers 5 loads checkpoints in their stored dtype (SmolLM2: bf16);
    # the add-on is float32, so a bf16 backbone crashed the first forward.
    import transformers

    from ekdant.backbone import load_backbone

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        staticmethod(lambda _id, **kw: tiny_llama().to(torch.bfloat16)))
    lm = load_backbone("any")
    assert all(p.dtype == torch.float32 for p in lm.parameters())


def test_addon_mask_matches_loss_positions():
    # logits[t] predicts labels[t+1], so the add-on's diagnostics / spread
    # floor must cover positions t where labels[t+1] != -100.
    m = _arms()["sandbox"]
    seen = {}
    real = m.addon.forward
    m.addon.forward = lambda h, mask: (seen.setdefault("mask", mask), real(h, mask))[1]
    ids, _ = _batch()
    labels = torch.full_like(ids, -100)
    labels[:, 4:6] = ids[:, 4:6]
    m(ids, labels=labels)
    want = torch.zeros_like(ids)
    want[:, 3:5] = 1
    assert torch.equal(seen["mask"], want)
