"""T57: v5 = the v22 recipe as the final model, and L2 dropout: L2 is
switched off for a share of the training steps, so SmolLM2 cannot hand its
own job to L2 and L2 must add something on top."""

import json
import pathlib

import pytest
import torch

import ekdant.train_gen as G
from ekdant.model import V2Config

CONFIGS = pathlib.Path(__file__).resolve().parents[1] / "configs"


def test_v5_is_exactly_the_v22_recipe():
    v5 = json.loads((CONFIGS / "v5_final.json").read_text())
    v22 = json.loads((CONFIGS / "v2_sandbox_final.json").read_text())
    assert v5 == v22
    cfg = V2Config.from_dict({**V2Config().to_dict(), **v5})
    # none of the later L2 versions is switched on
    assert not (cfg.sandbox_context or cfg.sandbox_confidence or cfg.sandbox_aligned
                or cfg.sandbox_stream_scale or cfg.sandbox_read_layers)
    assert cfg.addon_layer == -1 and not cfg.addon_layers and not cfg.addon_final


def test_dropped_steps_are_fixed_by_seed_stage_and_step():
    a = [G.l2_dropped(17, 0, s, 0.3) for s in range(4000)]
    assert a == [G.l2_dropped(17, 0, s, 0.3) for s in range(4000)]   # resume-safe
    assert 0.27 < sum(a) / len(a) < 0.33
    assert a != [G.l2_dropped(23, 0, s, 0.3) for s in range(4000)]
    assert a != [G.l2_dropped(17, 1, s, 0.3) for s in range(4000)]
    assert not any(G.l2_dropped(17, 0, s, 0.0) for s in range(1000))


def test_l2_drop_is_off_by_default():
    assert G.build_parser().parse_args(["--variant", "sandbox", "--out", "x"]).l2_drop == 0.0
    assert "l2_drop" in G.RUN_KEYS


@pytest.mark.parametrize("extra,variant", [(["--l2-drop", "0.3"], "baseline"),
                                           (["--l2-drop", "1.0"], "sandbox"),
                                           (["--l2-drop", "0.3", "--gate-floor", "0.5",
                                             "--gate-force"], "sandbox")])
def test_l2_drop_refused_where_it_makes_no_sense(tiny, tmp_path, extra, variant):
    with pytest.raises(SystemExit):
        G.main(_args(tiny, tmp_path / "r", variant=variant, extra=extra))


def test_run_with_l2_drop_counts_dropped_steps_and_l2_still_learns(tiny, tmp_path):
    out = tmp_path / "run"
    G.main(_args(tiny, out, extra=["--l2-drop", "0.5"]))
    r = json.loads((out / "results.json").read_text())
    ck = torch.load(out / "last.pt", weights_only=False)
    assert r["cost"]["l2_dropped_steps"] == ck["l2_dropped_steps"] > 0
    assert ck["run_args"]["l2_drop"] == 0.5
    assert any(not torch.equal(v, torch.zeros_like(v))
               for k, v in ck["model"].items() if k.endswith("gate"))
    assert "l2_off" in r["test"]


def test_dropped_step_leaves_l2_untouched(tiny, tmp_path, monkeypatch):
    # every step dropped: L2's weights must stay exactly at their start
    monkeypatch.setattr(G, "l2_dropped", lambda *a: True)
    out = tmp_path / "run"
    G.main(_args(tiny, out, extra=["--l2-drop", "0.5", "--ft-epochs", "0"]))
    ck = torch.load(out / "last.pt", weights_only=False)
    assert ck["l2_dropped_steps"] > 0
    assert all(torch.equal(v, torch.zeros_like(v))
               for k, v in ck["model"].items() if k.endswith("gate"))


def test_old_runs_without_l2_drop_still_resume(tiny, tmp_path):
    out = tmp_path / "run"
    G.main(_args(tiny, out))
    ck = torch.load(out / "last.pt", weights_only=False)
    ck["run_args"].pop("l2_drop")
    ck.pop("l2_dropped_steps")
    torch.save(ck, out / "last.pt")
    G.main(_args(tiny, out, extra=["--resume"]))
    r = json.loads((out / "results.json").read_text())
    assert r["cost"]["l2_dropped_steps"] == 0


# --- T58: L2's output at the model's own scale -------------------------------

def _scaled_pair(**kw):
    from ekdant.backbone import tiny_llama
    from ekdant.model import EkdantLM
    torch.manual_seed(0)
    lm = tiny_llama(layers=3)
    with torch.no_grad():
        lm.model.norm.weight.uniform_(1.5, 2.5)        # like SmolLM2 (~1.8)
    ref = {k: v.clone() for k, v in lm.state_dict().items()}
    m = EkdantLM(V2Config(addon="sandbox", debate_rounds=1, sandbox_mlp_ratio=2.0, **kw), lm)
    plain = tiny_llama(layers=3)
    plain.load_state_dict(ref)
    return m, plain


def test_out_norm_starts_as_the_backbones_final_norm():
    m, _ = _scaled_pair(sandbox_out_from_backbone=True)
    assert torch.equal(m.addon.out_norm.weight, m.lm.model.norm.weight)
    assert m.addon.out_norm.weight is not m.lm.model.norm.weight     # a copy, trained apart
    old, _ = _scaled_pair()
    assert torch.equal(old.addon.out_norm.weight, torch.ones_like(old.addon.out_norm.weight))


def test_out_from_backbone_step_zero_is_the_backbone():
    m, plain = _scaled_pair(sandbox_out_from_backbone=True)
    ids = torch.randint(3, 64, (2, 7))
    with torch.no_grad():
        assert torch.allclose(m.eval()(ids).logits, plain(ids).logits, atol=1e-5)


def test_l2_output_has_the_models_scale():
    sizes = {}
    for on in (False, True):
        m, _ = _scaled_pair(sandbox_out_from_backbone=on)
        ids = torch.randint(3, 64, (2, 7))
        with torch.no_grad():
            h = m.lm.model(ids).last_hidden_state
            out = m.addon.sandbox(h)
            y = m.addon.out_norm(out.consensus)
        sizes[on] = float(y.pow(2).mean().sqrt() / h.pow(2).mean().sqrt())
    assert sizes[False] < 0.7                       # too small: the gate shrinks h
    assert 0.8 < sizes[True] < 1.25                 # the model's own size


def test_out_from_backbone_with_the_mix_sets_only_the_end_site():
    m, _ = _scaled_pair(sandbox_out_from_backbone=True, addon_layers=[2], addon_final=True)
    assert torch.equal(m.addon.final_site.out_norm.weight, m.lm.model.norm.weight)
    s = m.addon.sites[0].out_norm.weight
    assert torch.equal(s, torch.ones_like(s))


def test_out_from_backbone_validation():
    with pytest.raises(ValueError):                 # no end site
        V2Config(addon="sandbox", addon_layer=2, sandbox_out_from_backbone=True).validate()
    with pytest.raises(ValueError):                 # aligned already sizes it
        V2Config(addon="sandbox", sandbox_aligned=True, sandbox_out_from_backbone=True).validate()


def test_shipped_v5_out_config_is_v5_plus_the_fix():
    out = json.loads((CONFIGS / "v5_out.json").read_text())
    v5 = json.loads((CONFIGS / "v5_final.json").read_text())
    assert {k for k in out if out[k] != v5.get(k)} == {"sandbox_out_from_backbone"}
    assert out["sandbox_out_from_backbone"] is True


from test_train_gen import _args, tiny  # noqa: E402,F401  (shared fixture)
