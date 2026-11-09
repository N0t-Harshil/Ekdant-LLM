"""T61: phase 1 trains the yes / no decision from the evidence.
pqa_artificial is 92.8 % yes and its training target was only the
conclusion, so the decision word the PubMedQA benchmark scores was never
trained; ``--evidence`` balances the labels and adds conclusion-only rows
whose meaning (and right answer) is flipped by one negation."""

import json
from collections import Counter

import torch

import ekdant.gen_data as D
import ekdant.train_gen as G
from ekdant.gen_data import GenExample, evidence_examples, question_key


def _train():
    ex = [GenExample(f"does drug {i} work", "Drug %d can reduce pain." % i,
                     context=f"abstract {i}", label="yes") for i in range(20)]
    ex += [GenExample(f"is test {i} useful", "Test %d does not predict risk." % i,
                      context=f"abstract t{i}", label="no") for i in range(4)]
    ex.append(GenExample("two sentences", "It works. It is safe.", "a", label="yes"))
    ex.append(GenExample("maybe one", "Unclear.", "a", label="maybe"))
    return ex


def test_balanced_rows_carry_decision_then_conclusion():
    rows, n = evidence_examples(_train(), seed=1)
    full = [r for r in rows if r.context.startswith("abstract")]
    assert Counter(r.label for r in full) == {"yes": 4, "no": 4}
    assert n["balanced"] == 8
    for r in full:
        assert r.answer.startswith(r.label + ". ")


def test_flipped_pairs_have_opposite_answers_and_conclusion_as_context():
    rows, n = evidence_examples(_train(), seed=1)
    short = [r for r in rows if not r.context.startswith("abstract")]
    assert n["flip_pairs"] == 24                      # two-sentence row refused
    assert len(short) == 48 and Counter(r.answer for r in short) == {"yes": 24, "no": 24}
    by_q = {}
    for r in short:
        by_q.setdefault(r.question, []).append(r)
    for q, (a, b) in by_q.items():
        assert {a.answer, b.answer} == {"yes", "no"} and a.context != b.context
    pair = {r.answer: r.context for r in by_q["does drug 3 work"]}
    assert pair == {"yes": "Drug 3 can reduce pain.", "no": "Drug 3 cannot reduce pain."}
    pair = {r.answer: r.context for r in by_q["is test 1 useful"]}
    assert pair == {"no": "Test 1 does not predict risk.", "yes": "Test 1 does predict risk."}
    assert all(r.label in ("yes", "no") for r in rows)


def test_benchmark_questions_are_excluded_and_seed_fixes_the_rows():
    ex = _train()
    rows, n = evidence_examples(ex, seed=1, exclude={question_key("Does drug 3 work ")})
    assert all(question_key(r.question) != "does drug 3 work" for r in rows)
    assert n["excluded"] == 1
    a, _ = evidence_examples(ex, seed=5)
    b, _ = evidence_examples(ex, seed=5)
    assert [(r.question, r.answer, r.context) for r in a] == \
        [(r.question, r.answer, r.context) for r in b]


def test_loader_keeps_the_label():
    rows = [{"q": " q ", "a": " a ", "l": "no"}]
    ex = D._rows_to_examples(rows, lambda r: GenExample(r["q"], r["a"], label=r["l"]))
    assert ex[0].label == "no"


def _pub_labeled(n=40):
    return [GenExample(f"does drug {i} work", f"Drug {i} can help patients.",
                       context=f"we studied drug {i}", label="yes" if i % 4 else "no")
            for i in range(n)]


def test_evidence_run_trains_on_the_new_rows_and_is_recorded(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(G, "load_pubmedqa", lambda limit=None, seed=17: _pub_labeled())
    monkeypatch.setattr(G, "pubmedqa_test_keys", lambda: {"does drug 5 work"})
    seen = {}
    real = G.GenDataset

    def spy(ex, *a, **k):
        seen.setdefault("first", ex)
        return real(ex, *a, **k)

    monkeypatch.setattr(G, "GenDataset", spy)
    out = tmp_path / "run"
    G.main(_args(tiny, out, extra=["--evidence"]))
    train = seen["first"]
    assert any(r.answer in ("yes", "no") for r in train)          # flipped rows
    assert all(question_key(r.question) != "does drug 5 work" for r in train)
    r = json.loads((out / "results.json").read_text())
    assert r["evidence_data"]["flip_pairs"] > 0
    ck = torch.load(out / "last.pt", weights_only=False)
    assert ck["run_args"]["evidence"] is True
    assert G.build_parser().parse_args(["--variant", "sandbox", "--out", "x"]).evidence is False


from test_train_gen import _args, tiny  # noqa: E402,F401  (shared fixture)
