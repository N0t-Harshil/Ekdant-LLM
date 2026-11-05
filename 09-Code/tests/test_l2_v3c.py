"""T54: L2 v3c - thinking twice is trained (1..N debate passes drawn per
training forward), and the benchmark asks whether L2's confidence knows
more than the model's own probability (per next word and per question)."""

import json
import pathlib

import pytest
import torch

import ekdant.gen_eval as M
from ekdant.backbone import tiny_llama
from ekdant.model import EkdantLM, V2Config
from ekdant.sandbox import Sandbox

V3B = dict(n_sandbox=5, sandbox_context=True, sandbox_confidence=True,
           sandbox_aligned=True, sandbox_diverse_read=True, token_gate_min=0.5)
SMALL = dict(debate_rounds=1, sandbox_mlp_ratio=2.0)


def _count_passes(box):
    calls = []
    box.debate[0].register_forward_hook(lambda *a: calls.append(1))
    return calls


def test_training_draws_one_or_two_passes_and_answers_with_two():
    torch.manual_seed(0)
    box = Sandbox(32, K=5, rounds=1, train_passes=2)
    calls = _count_passes(box)
    seen = set()
    box.train()
    for _ in range(40):
        calls.clear()
        box(torch.randn(1, 4, 32), passes=box.draw_passes())
        seen.add(len(calls))
    assert seen == {1, 2}                              # both depths are trained
    box.eval()
    assert box.draw_passes() is None and box.debate_passes == 2
    calls.clear()
    box(torch.randn(1, 4, 32))
    assert len(calls) == 2                             # it answers thinking twice


def test_default_is_one_pass_as_before():
    box = Sandbox(32, K=5, rounds=1).train()
    assert box.train_passes == 1 and box.debate_passes == 1
    assert box.draw_passes() is None
    with pytest.raises(ValueError):
        V2Config(addon="sandbox", sandbox_train_passes=0).validate()


@pytest.mark.parametrize("ckpt", [False, True])
def test_two_pass_training_step_is_finite_and_checkpoint_safe(ckpt):
    torch.manual_seed(0)
    m = EkdantLM(V2Config(addon="sandbox", **SMALL, **V3B, sandbox_train_passes=2,
                          sandbox_checkpoint=ckpt), tiny_llama(layers=2)).train()
    ids = torch.randint(3, 64, (2, 7))
    for _ in range(4):                                 # draws 1 and 2 passes
        m.zero_grad()
        out = m(ids, labels=ids.clone())
        out.loss.backward()
        assert torch.isfinite(out.loss)
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_step_zero_still_equals_the_backbone():
    torch.manual_seed(0)
    bb = tiny_llama(layers=2)
    ref = {k: v.clone() for k, v in bb.state_dict().items()}
    m = EkdantLM(V2Config(addon="sandbox", **SMALL, **V3B, sandbox_train_passes=2), bb).eval()
    plain = tiny_llama(layers=2)
    plain.load_state_dict(ref)
    ids = torch.randint(3, 64, (2, 7))
    with torch.no_grad():
        assert torch.allclose(m(ids).logits, plain(ids).logits, atol=1e-5)


def test_think_grid_around_the_trained_passes():
    assert [x[0] for x in M.think_settings(8)][:2] == ["debate x0", "debate x2"]
    labels = [x[0] for x in M.think_settings(8, passes=2)]
    assert labels[:3] == ["debate x0", "debate x1", "debate x3"]
    label = M.describe_l2(V2Config(addon="sandbox", **V3B, sandbox_train_passes=2))
    assert label.endswith("thinks x2")


def test_knows_measures_the_sure_and_unsure_half():
    k = M.knows([0.9, 0.8, 0.3, 0.2], [1, 1, 0, 1])
    assert k["sure_half"] == 1.0 and k["unsure_half"] == 0.5
    assert k["auroc"] == pytest.approx(2 / 3)


def test_answer_nll_reports_the_models_own_confidence_for_every_model():
    ids = torch.randint(3, 64, (2, 8))
    labels = ids.clone()
    labels[:, :3] = -100
    batch = [{"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels,
              "idx": torch.tensor([0, 1])}]
    import contextlib
    for addon, kw in (("none", {}), ("sandbox", V3B)):
        torch.manual_seed(0)
        m = EkdantLM(V2Config(addon=addon, **SMALL, **kw), tiny_llama(layers=2))
        r = M.answer_nll(m, batch, torch.device("cpu"), contextlib.nullcontext)
        a = r["own_confidence"]["auroc"]
        assert a != a or 0.0 <= a <= 1.0                # nan if all right / all wrong
        assert r["own_confidence"]["tokens"] == int((labels[:, 1:] != -100).sum())
        assert ("confidence" in r) == (addon == "sandbox")


def test_questions_carry_both_confidences():
    from test_train_gen import HashTok
    qs = [M.Question(f"does drug {i} work", ["yes", "no", "maybe"], i % 3,
                     context=f"we studied drug {i}") for i in range(6)]
    for addon, kw in (("none", {}), ("sandbox", V3B)):
        torch.manual_seed(0)
        m = EkdantLM(V2Config(addon=addon, **SMALL, **kw), tiny_llama()).eval()
        r = M.score_questions(m, HashTok(), qs, torch.device("cpu"),
                              M._autocast(torch.device("cpu")))
        assert all(0.0 < row["own_conf"] <= 1.0 for row in r["rows"])
        assert set(r["know"]) == ({"own_conf", "l2_conf"} if addon == "sandbox"
                                  else {"own_conf"})
        if addon == "sandbox":
            assert all(0.0 < row["l2_conf"] < 1.0 for row in r["rows"])


def test_know_table_and_rescore_of_old_files(tmp_path):
    d = tmp_path / "v3c_end"
    d.mkdir()
    k = {"auroc": 0.7, "sure_half": 0.8, "unsure_half": 0.4, "n": 10}
    m = {"version": M.MCQ_VERSION, "l2": "end",
         "pubmedqa": {"accuracy": 0.6, "rows": [], "know": {"own_conf": k, "l2_conf": k}},
         "medquad_ppl": {"on": {"own_confidence": {"auroc": 0.81},
                                "confidence": {"auroc": 0.82}}}}
    (d / "mcq.json").write_text(json.dumps(m))
    t = M.know_table([str(d)])
    assert "| v3c_end | PubMedQA | 60.0 | 0.700 | 80.0 / 40.0 | 0.700 | 80.0 / 40.0 |" in t
    assert "| v3c_end | 0.810 | 0.820 |" in t
    m["version"] = 4
    (d / "mcq.json").write_text(json.dumps(m))
    assert "confidence test (T54)" in M._score_state(str(d), ["pubmedqa"])


def test_think_table_caption_does_not_claim_one_trained_pass(tmp_path):
    d = tmp_path / "v3c_end"
    d.mkdir()
    m = {"version": M.MCQ_VERSION, "l2": "end",
         "pubmedqa": {"accuracy": 0.6, "rows": [],
                      "think": {"trained": "debate x2, steps 8",
                                "debate x1": {"accuracy": 0.6, "rows": []}}}}
    (d / "mcq.json").write_text(json.dumps(m))
    t = M.think_table([str(d)])
    assert "60.0 (debate x2, steps 8)" in t
    assert "trained: 1" not in t and "0 or 2 times" not in t


def test_shipped_v3c_config():
    path = pathlib.Path(__file__).resolve().parents[1] / "configs" / "v3c_sandbox_end.json"
    cfg = V2Config.from_dict(json.loads(path.read_text()))
    assert cfg.sandbox_train_passes == 2
    v3b = json.loads((path.parent / "v3b_sandbox_end.json").read_text())
    new = json.loads(path.read_text())
    assert {k for k in new if new[k] != v3b.get(k)} == {"sandbox_train_passes"}
