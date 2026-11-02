"""T46: multiple-choice accuracy of finished v2 runs (MedMCQA, MedQA,
PubMedQA yes/no/maybe) - option likelihood in the training format, the
letter method, L2 off, the comparison table with paired CIs."""

import json

import pytest
import torch
import torch.nn.functional as F

import ekdant.gen_eval as M
import ekdant.train_gen as G
from ekdant.backbone import tiny_llama
from ekdant.model import EkdantLM, V2Config

SMALL = dict(debate_rounds=1, sandbox_mlp_ratio=2.0)


def _qs():
    return {
        "medmcqa": [M.Question(f"which drug treats disease {i}",
                               [f"drug {i}", "rest", "surgery", "nothing"], i % 4)
                    for i in range(6)],
        "pubmedqa": [M.Question(f"does drug {i} work", ["yes", "no", "maybe"], i % 3,
                                context=f"we studied drug {i} in many patients")
                     for i in range(5)],
    }


def _model(addon="sandbox"):
    torch.manual_seed(0)
    return EkdantLM(V2Config(addon=addon, **SMALL), tiny_llama()).eval()


def _tok():
    from test_train_gen import HashTok
    return HashTok()


def test_option_score_is_the_mean_log_prob_of_the_option_tokens():
    m, tok = _model(), _tok()
    q = M.Question("which drug treats disease 3", ["drug three now", "rest"], 0)
    r = M.score_questions(m, tok, [q], torch.device("cpu"),
                          lambda: torch.autocast("cpu", enabled=False), max_len=64)
    want = []
    p = M._mcq_prompt(tok, q, 64, letters=False)
    for o in q.options:
        oi = M._enc(tok, " " + o)
        ids = torch.tensor([p + oi])
        with torch.no_grad():
            lp = F.log_softmax(m(ids).logits[0].float(), -1)
        want.append(float(sum(lp[len(p) - 1 + k, t] for k, t in enumerate(oi)) / len(oi)))
    assert r["rows"][0]["pick"] == max(range(2), key=want.__getitem__)
    assert r["rows"][0]["right"] == int(r["rows"][0]["pick"] == 0)


def test_batching_does_not_change_the_answers():
    m, tok = _model(), _tok()
    qs = _qs()["medmcqa"] + _qs()["pubmedqa"]
    auto = lambda: torch.autocast("cpu", enabled=False)  # noqa: E731
    a = M.score_questions(m, tok, qs, torch.device("cpu"), auto, 64, batch_size=1)
    b = M.score_questions(m, tok, qs, torch.device("cpu"), auto, 64, batch_size=7)
    assert [r["pick"] for r in a["rows"]] == [r["pick"] for r in b["rows"]]
    assert [r.get("letter_pick") for r in a["rows"]] == [r.get("letter_pick") for r in b["rows"]]


def test_letter_method_only_for_four_options_and_long_questions_skipped():
    m, tok = _model(), _tok()
    auto = lambda: torch.autocast("cpu", enabled=False)  # noqa: E731
    long_q = M.Question(" ".join(["word"] * 200), ["a", "b", "c", "d"], 0)
    r = M.score_questions(m, tok, _qs()["pubmedqa"] + [long_q], torch.device("cpu"),
                          auto, 64)
    assert r["n"] == 5 and r["skipped"] == 1
    assert all("letter_pick" not in row for row in r["rows"])
    r4 = M.score_questions(m, tok, _qs()["medmcqa"], torch.device("cpu"), auto, 64)
    assert all("letter_pick" in row for row in r4["rows"]) and "letter_accuracy" in r4
    assert 0.0 <= r4["accuracy"] <= 1.0 and r4["ci"][0] <= r4["accuracy"] <= r4["ci"][1]


def test_summary_counts_right_answers():
    s = M.mcq_summarise([{"q": 0, "pick": 1, "right": 1}, {"q": 1, "pick": 0, "right": 0},
                     {"q": 2, "pick": 2, "right": 1}, {"q": 3, "pick": 2, "right": 1}], 5)
    assert s["accuracy"] == pytest.approx(0.75) and s["n"] == 4 and s["skipped"] == 1


@pytest.fixture
def two_runs(tiny, tmp_path, monkeypatch):
    G.main(_args(tiny, tmp_path / "base", variant="baseline"))
    G.main(_args(tiny, tmp_path / "l2"))
    monkeypatch.setattr(M, "load_backbone", G.load_backbone)
    monkeypatch.setattr(M, "load_tokenizer", G.load_tokenizer)
    monkeypatch.setattr(M, "load_questions",
                        lambda name, limit=None, seed=17: _qs()[name])
    from test_train_gen import _quad
    monkeypatch.setattr("ekdant.gen_data.load_medquad", lambda limit=None, seed=17: _quad())
    return tmp_path / "base", tmp_path / "l2"


def test_runs_are_scored_with_their_selected_checkpoint(two_runs, capsys):
    base, l2 = two_runs
    M.mcq_main([str(base), str(l2), "--datasets", "medmcqa,pubmedqa", "--gpu", "cpu",
           "--min-free-gb", "0"])
    text = capsys.readouterr().out
    rb = json.loads((base / "mcq.json").read_text())
    rl = json.loads((l2 / "mcq.json").read_text())
    sel = json.loads((l2 / "results.json").read_text())["selected"]["checkpoint"]
    assert rl["checkpoint"] == sel
    assert "l2_off" in rl["medmcqa"] and "l2_off" not in rb["medmcqa"]
    assert "% right" in text and "points" in text and "with L2 switched off" in text
    # the scored weights are the run's own: re-scoring them by hand agrees
    cfg, state, _ = M.load_run(str(l2))
    m = EkdantLM(cfg, G.load_backbone(cfg.backbone))
    m.load_state_dict(state)
    again = M.score_questions(m, G.load_tokenizer(""), _qs()["medmcqa"],
                              torch.device("cpu"),
                              lambda: torch.autocast("cpu", enabled=False),
                              cfg.max_seq_len)
    assert [r["pick"] for r in again["rows"]] == [r["pick"] for r in rl["medmcqa"]["rows"]]


def test_compare_only_reads_existing_files(two_runs, capsys):
    base, l2 = two_runs
    M.mcq_main([str(base), str(l2), "--datasets", "medmcqa", "--gpu", "cpu",
           "--min-free-gb", "0", "--no-l2-off"])
    capsys.readouterr()
    M.mcq_main(["--compare", str(base), str(l2)])
    text = capsys.readouterr().out
    assert "medmcqa" in text and "l2 - base on medmcqa" in text


def _fake(dirpath, rights, l2="-"):
    dirpath.mkdir()
    m = {"l2": l2}
    for name, rr in rights.items():
        rows = [{"q": i, "pick": 0, "right": r} for i, r in enumerate(rr)]
        m[name] = M.mcq_summarise(rows, len(rows))
    (dirpath / "mcq.json").write_text(json.dumps(m))
    return str(dirpath)


def test_leaderboard_ranks_models_and_marks_real_gains(tmp_path):
    base = _fake(tmp_path / "v_baseline", {"medmcqa": [0, 1] * 50, "pubmedqa": [0, 1] * 50})
    good = _fake(tmp_path / "v_good", {"medmcqa": [1] * 100, "pubmedqa": [1] * 100},
                 l2="end, free")
    same = _fake(tmp_path / "v_same", {"medmcqa": [1, 0] * 50, "pubmedqa": [0, 1] * 50},
                 l2="mix (9 sites), forced")
    board = M.mcq_leaderboard([base, same, good])
    lines = [ln for ln in board.splitlines() if ln.startswith("| ")]
    assert lines[0].startswith("| # | Model | L2 | MedMCQA | PubMedQA | Average")
    assert lines[1].startswith("| 1 | v_good | end, free | 100.0 (+50.0*) | 100.0 (+50.0*) | **100.0** | +50.0 points* |")
    assert "**v_baseline** (baseline)" in board
    same_row = next(ln for ln in lines if "v_same" in ln)
    assert "+0.0 points |" in same_row and "*" not in same_row.split("|")[7]   # a tie: no star
    assert "chance 25%" in board and "real difference" in board


def test_leaderboard_includes_the_generative_benchmark(two_runs, tmp_path, capsys):
    base, l2 = two_runs
    md = tmp_path / "board.md"
    M.mcq_main([str(base), str(l2), "--datasets", "medmcqa", "--gpu", "cpu",
                "--min-free-gb", "0", "--no-l2-off", "--baseline", "base",
                "--markdown", str(md)])
    board = md.read_text()
    row = next(ln for ln in board.splitlines() if ln.startswith("| ") and "| l2 |" in ln)
    ppl = json.loads((l2 / "results.json").read_text())["test"]["medquad_bench"]["ppl"]
    assert f"{ppl:.3f} (" in row and "end, free" in row
    assert "(baseline)" in board and "| - |" in board


def test_leaderboard_survives_a_results_file_without_the_benchmark(tmp_path):
    base = _fake(tmp_path / "v_baseline", {"pubmedqa": [0, 1] * 5})
    other = _fake(tmp_path / "v_other", {"pubmedqa": [1, 1] * 5})
    (tmp_path / "v_other" / "results.json").write_text(json.dumps({"test": {}}))
    assert "| v_other |" in M.mcq_leaderboard([base, other])


# ---- T47: thinking-time test, and all models in one command ----

@pytest.mark.parametrize("kw", [{}, {"addon_layers": [1, 2], "addon_final": True}])
def test_think_changes_l2_and_restores_it(kw):
    torch.manual_seed(0)
    m = EkdantLM(V2Config(addon="sandbox", **SMALL, **kw), tiny_llama()).eval()
    for g in m.addon.gates():
        torch.nn.init.constant_(g, 0.5)
    assert len(m.sandboxes()) == 1                     # shared L2 counted once
    ids = torch.randint(3, 64, (2, 7))
    with torch.no_grad():
        ref = m(ids).logits
        with m.think(passes=1, steps=m.sandboxes()[0].steps):
            assert torch.equal(m(ids).logits, ref)     # trained setting = unchanged
        for p, s in ((0, None), (2, None), (None, 0), (None, 32)):
            with m.think(p, s):
                assert not torch.allclose(m(ids).logits, ref), (p, s)
        assert torch.equal(m(ids).logits, ref)         # restored
    assert m.sandboxes()[0].debate_passes == 1


def test_think_settings_cover_less_and_more():
    labels = [x[0] for x in M.think_settings(8)]
    assert labels == ["debate x0", "debate x2", "steps 0", "steps 4", "steps 16", "steps 32"]


def test_all_scores_every_finished_model_and_keeps_going(two_runs, tmp_path, capsys):
    (tmp_path / "half_done").mkdir()                   # not finished: skipped
    broken = tmp_path / "x_broken_T45"
    broken.mkdir()
    bad = tmp_path / "z_bad"                           # finished-looking, unloadable
    bad.mkdir()
    (bad / "last.pt").write_text("not a checkpoint")
    (bad / "results.json").write_text(json.dumps({"test": {}}))
    md = tmp_path / "board.md"
    argv = ["--all", "--runs-dir", str(tmp_path), "--datasets", "medmcqa,pubmedqa",
            "--gpu", "cpu", "--min-free-gb", "0", "--markdown", str(md)]
    M.mcq_main(argv)
    text = capsys.readouterr().out
    assert "FAILED" in text and "z_bad" in text and "=== ALL DONE" in text
    zs = json.loads((tmp_path / "zero_shot_smollm2" / "mcq.json").read_text())
    rl = json.loads((tmp_path / "l2" / "mcq.json").read_text())
    rb = json.loads((tmp_path / "base" / "mcq.json").read_text())
    assert zs["checkpoint"] == "pretrained" and "think" not in zs["pubmedqa"]
    assert "think" not in rb["pubmedqa"] and "think" not in rl["medmcqa"]
    th = rl["pubmedqa"]["think"]
    assert set(th) >= {"trained", "debate x0", "debate x2", "steps 0"}
    assert not (tmp_path / "half_done" / "mcq.json").exists()
    assert not (broken / "mcq.json").exists()
    board = md.read_text()
    assert "Thinking-time test" in board and "| l2 | PubMedQA |" in board
    assert "zero_shot_smollm2" in board and "(baseline)" not in board.split("Thinking")[1]
    M.mcq_main(argv)                                    # a second call keeps finished work
    assert "already scored - kept" in capsys.readouterr().out


# ---- T48: every number in one run (perplexity on / off / thinking, facts)

def test_one_run_collects_every_number(two_runs, tmp_path):
    base, l2 = two_runs
    md = tmp_path / "report.md"
    M.mcq_main([str(base), str(l2), "--datasets", "pubmedqa", "--gpu", "cpu",
                "--min-free-gb", "0", "--markdown", str(md), "--baseline", "base"])
    for run in (base, l2):
        m = json.loads((run / "mcq.json").read_text())
        res = json.loads((run / "results.json").read_text())
        # the benchmark split is rebuilt exactly: same perplexity as training saved
        assert m["medquad_ppl"]["on"]["ppl"] == pytest.approx(
            res["test"]["medquad_bench"]["ppl"], rel=1e-4)
        assert m["params_total"] == res["cost"]["params_total"] and m["seed"] == 17
    ml = json.loads((l2 / "mcq.json").read_text())["medquad_ppl"]
    assert ml["off"]["ppl"] > 0 and {"trained", "debate x0", "steps 0"} <= set(ml["think"])
    assert ml["off"]["ppl"] == pytest.approx(json.loads((l2 / "results.json").read_text())
                                             ["test"]["l2_off"]["medquad_bench"]["ppl"], rel=1e-4)
    mb = json.loads((base / "mcq.json").read_text())["medquad_ppl"]
    assert "off" not in mb and "think" not in mb
    report = md.read_text()
    for section in ("### Leaderboard", "### Does the model depend on L2?",
                    "### Thinking time: % right", "### Thinking time: MedQuAD perplexity",
                    "### Model facts"):
        assert section in report, section
    facts = next(ln for ln in report.split("### Model facts")[1].splitlines() if "| l2 |" in ln)
    assert "end, free" in facts and "M |" in facts


def test_perplexity_pairs_only_runs_with_the_same_split(tmp_path):
    base = _fake(tmp_path / "v_baseline", {"pubmedqa": [0, 1] * 5})
    other = _fake(tmp_path / "v_s23", {"pubmedqa": [0, 1] * 5})
    rows = {"row_nll": [1.0, 2.0], "row_tokens": [1, 1], "row_idx": [0, 1], "ppl": 4.0}
    for d, seed in (("v_baseline", 17), ("v_s23", 23)):
        m = json.loads((tmp_path / d / "mcq.json").read_text())
        m["seed"], m["medquad_ppl"] = seed, {"on": rows, "row_group": {"0": "a", "1": "b"}}
        (tmp_path / d / "mcq.json").write_text(json.dumps(m))
    assert "other seed: different split" in M.mcq_leaderboard([base, other])


from test_train_gen import _args, tiny  # noqa: E402,F401  (shared fixture)
