"""T60: LoRA on the backbone as the second stage of "train L2 first": with
--l2-warmup N only L2 learns for N steps (SmolLM2 frozen), then L2 and
rank-r adapters learn while SmolLM2's own weights never change."""

import json

import pytest
import torch

import ekdant.gen_eval as M
import ekdant.train_gen as G
from ekdant.backbone import tiny_llama
from ekdant.model import (LORA_TARGETS, EkdantLM, LoRALinear, V2Config, apply_lora,
                          is_lora_param)

SMALL = dict(debate_rounds=1, sandbox_mlp_ratio=2.0)


def test_lora_linear_starts_as_the_base_layer():
    base = torch.nn.Linear(8, 5)
    lin = LoRALinear(base, r=2, alpha=4.0)
    x = torch.randn(3, 8)
    assert torch.equal(lin(x), base(x))                 # B = 0 at the start
    assert lin.scale == 2.0 and lin.lora_a.shape == (2, 8) and lin.lora_b.shape == (5, 2)


def test_apply_lora_wraps_every_projection_and_freezes_the_rest():
    lm = tiny_llama(layers=3)
    n = apply_lora(lm, 4, 8.0)
    assert n == 3 * len(LORA_TARGETS)
    train = [k for k, p in lm.named_parameters() if p.requires_grad]
    assert train and all(is_lora_param(k) for k in train)
    with pytest.raises(ValueError):
        apply_lora(torch.nn.Sequential(torch.nn.ReLU()), 4, 8.0)


@pytest.mark.parametrize("addon", ["none", "sandbox"])
def test_lora_model_step_zero_is_the_backbone_and_only_adapters_and_l2_train(addon):
    torch.manual_seed(0)
    lm = tiny_llama(layers=3)
    ref = {k: v.clone() for k, v in lm.state_dict().items()}
    m = EkdantLM(V2Config(addon=addon, lora_r=4, **SMALL), lm).eval()
    plain = tiny_llama(layers=3)
    plain.load_state_dict(ref)
    ids = torch.randint(3, 64, (2, 7))
    with torch.no_grad():
        assert torch.allclose(m(ids).logits, plain(ids).logits, atol=1e-5)
    addon_ids = {id(p) for p in m.addon.parameters()} if m.addon is not None else set()
    for k, p in m.named_parameters():
        assert p.requires_grad == (is_lora_param(k) or id(p) in addon_ids), k


def test_lora_config_validation_and_default_off():
    assert V2Config().lora_r == 0
    with pytest.raises(ValueError):
        V2Config(lora_r=-1).validate()
    with pytest.raises(ValueError):
        V2Config(lora_r=4, lora_alpha=0).validate()
    assert G.build_parser().parse_args(["--variant", "sandbox", "--out", "x"]).lora_r == 0
    assert {"lora_r", "lora_alpha"} <= set(G.RUN_KEYS)


def _base_weights_unchanged(ck):
    """SmolLM2's own weights in the checkpoint equal the pretrained ones."""
    fresh = G.load_backbone("").state_dict()
    for k, v in ck["model"].items():
        if not k.startswith("lm.") or is_lora_param(k):
            continue
        assert torch.equal(v, fresh[k[3:].replace(".base.", ".")]), k


def test_warmup_then_lora_never_changes_smollm2(tiny, tmp_path):
    out = tmp_path / "run"
    G.main(_args(tiny, out, extra=["--l2-warmup", "2", "--lora-r", "4", "--lr", "1e-3"]))
    ck = torch.load(out / "last.pt", weights_only=False)
    _base_weights_unchanged(ck)                         # unfreezing after warm-up
    assert any(not torch.equal(v, torch.zeros_like(v))  # touches only the adapters
               for k, v in ck["model"].items() if k.endswith("lora_b"))
    assert ck["run_args"]["lora_r"] == 4 and ck["cfg"]["lora_r"] == 4
    r = json.loads((out / "results.json").read_text())
    assert r["cost"]["params_trainable"] < r["cost"]["params_total"]
    assert "l2_off" in r["test"]


def test_adapters_wait_while_l2_warms_up(tiny, tmp_path):
    out = tmp_path / "run"
    G.main(_args(tiny, out, extra=["--l2-warmup", "100000", "--lora-r", "4",
                                   "--ft-epochs", "0"]))
    ck = torch.load(out / "last.pt", weights_only=False)
    assert all(torch.equal(v, torch.zeros_like(v))
               for k, v in ck["model"].items() if k.endswith("lora_b"))
    assert any(not torch.equal(v, torch.zeros_like(v))
               for k, v in ck["model"].items() if k.endswith("gate"))


def test_lora_run_is_scored_by_gen_eval_and_resume_needs_the_same_rank(tiny, tmp_path):
    out = tmp_path / "run"
    G.main(_args(tiny, out, extra=["--lora-r", "4"]))
    cfg, state, _ = M.load_run(str(out))
    m = EkdantLM(cfg, G.load_backbone(cfg.backbone))
    m.load_state_dict(state)                            # strict: the keys match
    with pytest.raises(SystemExit):
        G.main(_args(tiny, out, extra=["--resume", "--lora-r", "8"]))


def test_baseline_with_lora_trains_only_the_adapters(tiny, tmp_path):
    out = tmp_path / "base"
    G.main(_args(tiny, out, variant="baseline", extra=["--lora-r", "4", "--lr", "1e-3"]))
    _base_weights_unchanged(torch.load(out / "last.pt", weights_only=False))


def test_warmup_end_message_names_what_trains(tiny, tmp_path, capsys):
    # after the warm-up, a LoRA run must not claim that SmolLM2 trains again
    G.main(_args(tiny, tmp_path / "lora", extra=["--l2-warmup", "1", "--lora-r", "4"]))
    out = capsys.readouterr().out
    assert "LoRA adapters train" in out and "SmolLM2 trains again" not in out
    G.main(_args(tiny, tmp_path / "full", extra=["--l2-warmup", "1"]))
    assert "SmolLM2 trains again" in capsys.readouterr().out

from test_train_gen import _args, tiny  # noqa: E402,F401  (shared fixture)
