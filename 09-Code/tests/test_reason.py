"""T59: does L2 reason better? Three PubMedQA tests - where the gain comes
from (per true answer, answer shares), the questions without the abstract,
and the authors' conclusion with its meaning flipped by one negation."""

import json

import pytest
import torch

import ekdant.gen_eval as M
import ekdant.train_gen as G


@pytest.mark.parametrize("text,want", [
    ("MSWM does not appear to help.", "MSWM does appear to help."),
    ("DW imaging can be helpful.", "DW imaging cannot be helpful."),
    ("CT cannot replace BS.", "CT can replace BS."),
    ("There was no significant difference.", "There was a significant difference."),
    ("No significant change was seen.", "A significant change was seen."),
    ("The drug is safe in children.", "The drug is not safe in children."),
    ("It may  help\n patients.", "It may not help patients."),
])
def test_flip_adds_or_removes_one_negation(text, want):
    assert M.flip_conclusion(text) == want


@pytest.mark.parametrize("text", [
    "It is safe. It is cheap.",                 # two sentences
    "It doesn't work.",                         # negation not attached to an aux
    "It can and may help.",                     # two auxiliaries
    "It is not safe and does not work.",        # two negations
    "Results were without bias.",               # negation without an aux
    "Mitochondria move along strands.",         # nothing to flip
    "",
])
def test_flip_refuses_what_it_cannot_flip_safely(text):
    assert M.flip_conclusion(text) is None


def _raw():
    return [
        {"question": "does drug a work", "context": "we gave drug a to many",
         "answer": "yes", "conclusion": "Drug a can help patients."},
        {"question": "does drug b work", "context": "we gave drug b to some",
         "answer": "no", "conclusion": "Drug b does not help patients."},
        {"question": "does drug c work", "context": "drug c data unclear",
         "answer": "maybe", "conclusion": "Drug c may help."},
        {"question": "does drug d work", "context": "drug d trial",
         "answer": "yes", "conclusion": "Drug d helps. It is cheap."},
    ]


def test_reason_sets():
    sets, pairs = M.reason_sets(_raw())
    assert [len(sets[k]) for k in ("full", "no_context", "flip_orig", "flip_neg")] == [4, 4, 2, 2]
    assert [q.answer for q in sets["full"]] == [0, 1, 2, 0]
    assert all(q.context == "" for q in sets["no_context"])
    assert [q.question for q in sets["no_context"]] == [q.question for q in sets["full"]]
    # maybe and the two-sentence conclusion are left out; the right answer flips
    assert [p["source"] for p in pairs] == [0, 1]
    assert [(o.answer, n.answer) for o, n in zip(sets["flip_orig"], sets["flip_neg"])] == [(0, 1), (1, 0)]
    assert sets["flip_orig"][0].context == "Drug a can help patients."
    assert sets["flip_neg"][0].context == "Drug a cannot help patients."


def test_reason_evaluate_keeps_the_right_answer():
    from test_mcq_eval import _model, _tok
    sets, _ = M.reason_sets(_raw())
    r = M.reason_evaluate(_model(), _tok(), sets, torch.device("cpu"), 128, 4, log=lambda *_: None)
    assert set(r) == set(sets)
    for name, qs in sets.items():
        assert [row["answer"] for row in r[name]["rows"]] == [qs[row["q"]].answer
                                                             for row in r[name]["rows"]]


def _write(d, seed, full_picks, noctx_picks, orig_picks, neg_picks):
    """reason.json with given picks; answers: full yes,no,maybe,yes; flips yes/no pairs."""
    d.mkdir()
    full_ans = [0, 1, 2, 0]

    def rows(picks, answers):
        return {"n": len(picks), "rows": [{"q": i, "pick": p, "answer": a, "right": int(p == a)}
                                          for i, (p, a) in enumerate(zip(picks, answers))]}
    m = {"version": M.REASON_VERSION, "seed": seed, "sets": {
        "full": rows(full_picks, full_ans), "no_context": rows(noctx_picks, full_ans),
        "flip_orig": rows(orig_picks, [0, 1, 0, 1]), "flip_neg": rows(neg_picks, [1, 0, 1, 0])}}
    (d / "reason.json").write_text(json.dumps(m))
    return str(d)


def test_report_separates_evidence_use_from_answer_preference(tmp_path):
    # baseline: always "yes" -> ignores every piece of evidence
    b = _write(tmp_path / "b", 17, [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0])
    # model: reads the abstract and follows the flipped conclusion
    m = _write(tmp_path / "m", 17, [0, 1, 2, 0], [0, 0, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0])
    t = M.reason_report([b], [("follower", [m])])
    line = lambda label: next(l for l in t.splitlines() if f"| {label} |" in l)  # noqa: E731
    assert "| 100.0 | 100.0 |" in line("% right, true answer yes")
    assert "| 0.0 | 100.0 |" in line("% right, true answer no")
    assert "| 100.0 | 50.0 |" in line("% of answers that are yes")
    assert "| 0.0 | 50.0 |" in line("uses the abstract (with - without)")
    assert "| 0.0 | 100.0 | +100.00" in line("follows the evidence (right on both)")
    assert "| 100.0 | 0.0 | -100.00" in line("same answer both times (ignores it)")
    assert "*" in line("follows the evidence (right on both)")


def test_report_checks_seed_order(tmp_path):
    b = _write(tmp_path / "b", 17, [0] * 4, [0] * 4, [0] * 4, [0] * 4)
    m = _write(tmp_path / "m", 23, [0] * 4, [0] * 4, [0] * 4, [0] * 4)
    with pytest.raises(ValueError):
        M.reason_report([b], [("x", [m])])
    with pytest.raises(ValueError):
        M.reason_report([b], [("x", [m, m])])


def test_reason_main_scores_once_and_writes_the_table(tiny, tmp_path, monkeypatch, capsys):
    G.main(_args(tiny, tmp_path / "base", variant="baseline"))
    G.main(_args(tiny, tmp_path / "l2"))
    monkeypatch.setattr(M, "load_backbone", G.load_backbone)
    monkeypatch.setattr(M, "load_tokenizer", G.load_tokenizer)
    monkeypatch.setattr(M, "_pubmedqa_raw", _raw)
    base, l2 = str(tmp_path / "base"), str(tmp_path / "l2")
    out, flips = tmp_path / "reason.md", tmp_path / "flips.md"
    argv = ["--baseline", base, "--model", "l2", l2, "--gpu", "cpu", "--min-free-gb", "0",
            "--markdown", str(out), "--flip-file", str(flips)]
    M.reason_main(argv)
    r = json.loads((tmp_path / "l2" / "reason.json").read_text())
    assert r["version"] == M.REASON_VERSION and r["seed"] == 17
    assert r["sets"]["flip_orig"]["n"] == 2 and r["sets"]["full"]["n"] == 4
    assert "follows the evidence" in out.read_text()
    assert "Drug a cannot help patients." in flips.read_text()
    # second call: nothing is scored again (no model is loaded)
    monkeypatch.setattr(M, "load_run", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    M.reason_main(argv)
    assert "Does L2 reason better" in capsys.readouterr().out


from test_train_gen import _args, tiny  # noqa: E402,F401  (shared fixture)
