"""v2 evaluation: perplexity, text metrics, batched KV-cache generation and
the paired bootstrap (spec 07-Management/30 §5)."""

import random

import pytest
import torch

from ekdant.backbone import tiny_llama
from ekdant.gen_data import GenExample
from ekdant.gen_eval import (
    answer_nll,
    by_type,
    generate_greedy,
    paired_bootstrap,
    rouge_l,
    score_generations,
    token_f1,
)
from ekdant.model import EkdantLM, V2Config

OFF = lambda: torch.autocast("cpu", enabled=False)  # noqa: E731


def _model(addon="none", gate=0.0):
    torch.manual_seed(0)
    m = EkdantLM(V2Config(addon=addon, debate_rounds=1, sandbox_mlp_ratio=2.0),
                 tiny_llama()).eval()
    if m.addon is not None:
        with torch.no_grad():
            m.addon.gate.fill_(gate)
    return m


def test_rouge_l_and_f1_known_values():
    p, r = "the cat sat on the mat", "the cat is on the mat"
    assert rouge_l(p, r) == pytest.approx(5 / 6, abs=1e-4)
    assert token_f1(p, r) == pytest.approx(5 / 6, abs=1e-4)
    assert rouge_l("The CAT, sat.", "the cat sat") == pytest.approx(1.0)
    assert rouge_l("", "") == 1.0 and token_f1("", "") == 1.0
    assert rouge_l("", "x") == 0.0 and token_f1("x", "") == 0.0
    s = score_generations([p, ""], [r, "x"])
    assert s["rouge_l"] == pytest.approx(5 / 12, abs=1e-4) and s["n"] == 2


def test_answer_nll_matches_model_loss():
    m = _model()
    torch.manual_seed(3)
    ids = torch.randint(3, 64, (2, 9))
    labels = torch.full_like(ids, -100)
    labels[:, 5:] = ids[:, 5:]
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
             "labels": labels, "idx": torch.tensor([0, 1])}
    r = answer_nll(m, [batch], torch.device("cpu"), OFF)
    with torch.no_grad():
        loss = m(ids, attention_mask=batch["attention_mask"], labels=labels).loss
    assert r["nll"] == pytest.approx(float(loss), abs=1e-5)
    assert r["tokens"] == 8 and r["n"] == 2 and r["row_idx"] == [0, 1]
    assert sum(r["row_tokens"]) == 8


def test_generate_batched_equals_single():
    m = _model("sandbox", gate=0.5)
    prompts = [[5, 6, 7], [8, 9, 10, 11, 12, 13], [14]]
    together = generate_greedy(m, prompts, max_new=6, device=torch.device("cpu"),
                               eos_id=2, pad_id=0, batch_size=3)
    alone = [generate_greedy(m, [p], max_new=6, device=torch.device("cpu"),
                             eos_id=2, pad_id=0, batch_size=1)[0] for p in prompts]
    assert together == alone
    assert all(len(g) <= 6 for g in together)


def test_bootstrap_ci_contains_true_shift():
    rng = random.Random(0)
    a = [rng.gauss(2.0, 0.5) for _ in range(400)]
    b = [x + 0.3 + rng.gauss(0, 0.1) for x in a]
    groups = [f"q{i // 2}" for i in range(400)]
    r = paired_bootstrap(a, b, groups, n=1000, seed=17)
    assert r["lo"] < 0.3 < r["hi"] and r["lo"] > 0
    assert r["diff"] == pytest.approx(sum(y - x for x, y in zip(a, b)) / 400)


def test_by_type_groups_rows():
    ex = [GenExample("q", "a", qtype="symptoms"), GenExample("q", "a", qtype="treatment"),
          GenExample("q", "a", qtype="symptoms")]
    res = {"row_idx": [0, 1, 2], "row_nll": [2.0, 4.0, 4.0],
           "row_tokens": [1, 2, 3]}
    t = by_type(res, ex)
    assert t["symptoms"] == pytest.approx(torch.exp(torch.tensor(6.0 / 4)).item())
    assert t["treatment"] == pytest.approx(torch.exp(torch.tensor(2.0)).item())


def test_bootstrap_token_weighted_matches_headline():
    # a/b = per-row NLL sums, weights = token counts: the statistic is the
    # per-token NLL difference, the same weighting as the headline perplexity.
    a, b, w = [10.0, 1.0], [8.0, 1.5], [10, 1]
    r = paired_bootstrap(a, b, ["q0", "q1"], n=200, seed=1, weights=w)
    assert r["diff"] == pytest.approx((-2.0 + 0.5) / 11)
