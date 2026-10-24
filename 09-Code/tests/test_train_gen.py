"""v2 training entry point (spec 07-Management/30 §6-7): GPU pre-flight,
the full zero-shot -> phase 1 -> phase 2 -> benchmark protocol, exact
mid-epoch resume, the fresh-run guard, and the --task dispatch."""

import json
import os

import pytest
import torch

import ekdant.train as T
import ekdant.train_gen as G
from ekdant.backbone import tiny_llama
from ekdant.gen_data import GenExample


class HashTok:
    """Stable word -> id in [3, 64) so the tiny 64-word Llama can embed it."""
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        words = text.replace("\n", " \n ").split()
        return {"input_ids": [3 + sum(map(ord, w)) % 61 for w in words]}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"w{i}" for i in ids)


def _pub(n=60):
    return [GenExample(f"does drug {i} work", f"yes drug {i} works well",
                       context=f"we studied drug {i} in many patients")
            for i in range(n)]


def _quad(n=80):
    qt = ["symptoms", "treatment", "information"]
    ex = [GenExample(f"what treats disease {i}", f"rest and drug {i}",
                     qtype=qt[i % 3]) for i in range(n)]
    ex.append(GenExample("WHAT TREATS DISEASE 3", "another source", qtype="treatment"))
    return ex


@pytest.fixture
def tiny(monkeypatch, tmp_path):
    torch.manual_seed(0)
    lm = tiny_llama()
    state = {k: v.clone() for k, v in lm.state_dict().items()}

    def fresh_lm(_id):
        m = tiny_llama()
        m.load_state_dict(state)
        return m

    monkeypatch.setattr(G, "load_backbone", fresh_lm)
    monkeypatch.setattr(G, "load_tokenizer", lambda _id: HashTok())
    monkeypatch.setattr(G, "load_pubmedqa", lambda limit=None, seed=17: _pub())
    monkeypatch.setattr(G, "load_medquad", lambda limit=None, seed=17: _quad())
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"debate_rounds": 1, "sandbox_mlp_ratio": 2.0,
                               "max_seq_len": 64, "max_answer": 16}))
    return str(cfg)


def _args(cfg, out, variant="sandbox", extra=()):
    return ["--variant", variant, "--config", cfg, "--out", str(out),
            "--batch-size", "4", "--grad-accum", "2", "--warmup", "2",
            "--epochs", "1", "--ft-epochs", "2", "--gen-subset", "4",
            "--gen-subset-pubmed", "3", "--max-new", "4", "--log-every", "0",
            "--min-free-gb", "0", "--gpu", "cpu", *extra]


def test_pick_device_refuses_low_memory(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda i: None)
    free = {0: 5e9, 1: 50e9}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i=0: (free[i], 80e9))
    assert G.pick_device("auto", 20) == torch.device("cuda", 1)
    with pytest.raises(SystemExit):
        G.pick_device("0", 20)
    free[1] = 1e9
    with pytest.raises(SystemExit):
        G.pick_device("auto", 20)


def test_gen_run_end_to_end_writes_results(tiny, tmp_path):
    out = tmp_path / "run"
    G.main(_args(tiny, out))
    r = json.loads((out / "results.json").read_text())
    for k in ("zero_shot", "phase1", "selected", "history", "cost"):
        assert k in r, k
    b = r["test"]["medquad_bench"]
    assert all(k in b for k in ("ppl", "rouge_l", "token_f1", "by_type",
                                "row_nll", "row_group"))
    assert r["test"]["gate_magnitude"] >= 0
    # generation subset size must not overwrite the perplexity row count
    assert b["n"] == len(b["row_nll"]) and b["gen_n"] == 4
    for s in (r["zero_shot"]["medquad_bench"], b, r["test"]["pubmedqa_held"]):
        assert s["ppl"] == s["ppl"] and s["ppl"] < float("inf")
    assert r["params_addon"] > 0
    assert any(h["stage"] == "pubmedqa" for h in r["history"])


def test_resume_mid_epoch_matches_uninterrupted(tiny, tmp_path, monkeypatch):
    G.main(_args(tiny, tmp_path / "a", extra=["--eval-fraction", "0.5"]))
    real = G.save_checkpoint
    calls = {"n": 0}

    def stop_after_first(state, path):
        real(state, path)
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(G, "save_checkpoint", stop_after_first)
    with pytest.raises(KeyboardInterrupt):
        G.main(_args(tiny, tmp_path / "b", extra=["--eval-fraction", "0.5"]))
    assert torch.load(tmp_path / "b" / "last.pt", weights_only=False)["batches_done"] > 0
    monkeypatch.setattr(G, "save_checkpoint", real)
    G.main(_args(tiny, tmp_path / "b", extra=["--eval-fraction", "0.5", "--resume"]))
    a = torch.load(tmp_path / "a" / "last.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "b" / "last.pt", weights_only=False)["model"]
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-6), k


def test_fresh_run_guard(tiny, tmp_path):
    out = tmp_path / "run"
    G.main(_args(tiny, out, variant="baseline"))
    with pytest.raises(SystemExit):
        G.main(_args(tiny, out, variant="baseline"))


def test_task_dispatch(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(G, "main", lambda argv: seen.setdefault("argv", argv))
    T.main(["--task", "gen", "--variant", "sandbox", "--out", "x"])
    assert seen["argv"] == ["--variant", "sandbox", "--out", "x"]
    with pytest.raises(SystemExit) as e:
        T.main(["--task", "mcq", "--help"])
    assert e.value.code == 0
    assert "--answer-weight" in capsys.readouterr().out


# ---- final-review fixes (each failed before its fix) ----

def test_eval_only_without_results_does_not_fake_zero_shot(tiny, tmp_path):
    out = tmp_path / "run"
    G.main(_args(tiny, out))
    os.remove(out / "results.json")
    G.main(_args(tiny, out, extra=["--eval-only"]))
    r = json.loads((out / "results.json").read_text())
    assert "zero_shot" not in r          # trained weights must never be labelled zero-shot


def test_resume_refuses_changed_flags_or_variant(tiny, tmp_path):
    out = tmp_path / "run"
    G.main(_args(tiny, out))
    a = _args(tiny, out, extra=["--resume"])
    with pytest.raises(SystemExit):
        G.main([x if x != "4" else "2" for x in a])       # --batch-size 4 -> 2
    with pytest.raises(SystemExit):
        G.main(_args(tiny, out, variant="baseline", extra=["--resume"]))


def test_phase1_eval_points_exclude_epoch_end():
    assert G.phase1_eval_points(4, 0.25, 2) == {2}      # 4 = epoch end, excluded
    assert G.phase1_eval_points(100, 0.25, 4) == {28, 52, 76}
    assert all(p % 4 == 0 for p in G.phase1_eval_points(57, 0.25, 4))


def test_grad_clip_is_per_group(tiny):
    from ekdant.model import EkdantLM, V2Config
    torch.manual_seed(0)
    m = EkdantLM(V2Config(addon="sandbox", debate_rounds=1, sandbox_mlp_ratio=2.0),
                 tiny_llama())
    for p in m.parameters():
        p.grad = torch.ones_like(p)
    for p in m.addon.parameters():
        p.grad = torch.full_like(p, 100.0)          # huge add-on gradient
    bb = [p for n, p in m.named_parameters() if not n.startswith("addon.")]
    ref = [p.grad.clone() for p in bb]
    torch.nn.utils.clip_grad_norm_([torch.nn.Parameter(g.clone()) for g in ref], 1.0)
    ref_norm = torch.sqrt(sum(g.pow(2).sum() for g in ref))
    G.clip_grads(m, 1.0)
    scale = min(1.0, 1.0 / float(ref_norm))
    for p, g in zip(bb, ref):                       # backbone clipped on its own norm
        assert torch.allclose(p.grad, g * scale, rtol=1e-5)
    addon_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in m.addon.parameters()))
    assert float(addon_norm) <= 1.0 + 1e-5


def test_wall_clock_accumulates_across_resume(tiny, tmp_path, monkeypatch):
    real = G.save_checkpoint
    calls = {"n": 0}

    def stop_after_first(state, path):
        real(state, path)
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(G, "save_checkpoint", stop_after_first)
    out = tmp_path / "b"
    with pytest.raises(KeyboardInterrupt):
        G.main(_args(tiny, out, extra=["--eval-fraction", "0.5"]))
    first = torch.load(out / "last.pt", weights_only=False)["wall_clock_s"]
    assert first > 0
    monkeypatch.setattr(G, "save_checkpoint", real)
    G.main(_args(tiny, out, extra=["--eval-fraction", "0.5", "--resume"]))
    r = json.loads((out / "results.json").read_text())
    assert r["cost"]["wall_clock_s"] > first


def test_pick_device_never_falls_back_to_cpu_silently(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit):
        G.pick_device("auto", 20)
    assert G.pick_device("cpu", 20) == torch.device("cpu")


def test_v2_has_only_baseline_and_sandbox_arms():
    # the same-size MLP control arm was removed at the user's request
    # (2026-10-23): v2 is a two-arm A/B, baseline vs sandbox
    from ekdant.model import V2Config
    assert set(G.ADDON) == {"baseline", "sandbox"}
    with pytest.raises(ValueError):
        V2Config(addon="mlp").validate()
    with pytest.raises(SystemExit):
        G.build_parser().parse_args(["--variant", "mlp_control", "--out", "x"])


def test_config_addon_must_match_variant(tiny, tmp_path):
    cfg = json.loads(open(tiny).read())
    cfg["addon"] = "none"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(cfg))
    args = _args(str(bad), tmp_path / "run", variant="sandbox")
    with pytest.raises(SystemExit):
        G.main(args)


def test_hf_token_from_file_is_used_but_never_saved(tiny, tmp_path, monkeypatch):
    secret = "hf_TESTsecret123"
    (tmp_path / "tok.txt").write_text(secret + "\n")
    cfg = json.loads(open(tiny).read())
    cfg["hf_token_file"] = str(tmp_path / "tok.txt")
    path = tmp_path / "cfg_tok.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    out = tmp_path / "run"
    G.main(_args(str(path), out))
    assert os.environ.get("HF_TOKEN") == secret
    monkeypatch.delenv("HF_TOKEN")
    for f in out.iterdir():
        assert secret.encode() not in f.read_bytes(), f.name


def _poison_training_losses(monkeypatch, every):
    """Make every `every`-th training forward return an infinite loss."""
    from ekdant.model import EkdantLM
    real = EkdantLM.forward
    seen = {"n": 0}

    def forward(self, *a, **kw):
        out = real(self, *a, **kw)
        if self.training and out.loss is not None:
            seen["n"] += 1
            if seen["n"] % every == 0:
                out.loss = out.loss * float("inf")
        return out

    monkeypatch.setattr(EkdantLM, "forward", forward)


def test_non_finite_steps_are_skipped_and_counted(tiny, tmp_path, monkeypatch):
    # T32 guard: a non-finite loss or gradient must never reach the weights
    _poison_training_losses(monkeypatch, every=3)
    out = tmp_path / "run"
    G.main(_args(tiny, out))
    r = json.loads((out / "results.json").read_text())
    assert r["cost"]["skipped_steps"] > 0
    w = torch.load(out / "last.pt", weights_only=False)["model"]
    assert all(torch.isfinite(v).all() for v in w.values())


def test_run_stops_after_too_many_bad_steps(tiny, tmp_path, monkeypatch):
    _poison_training_losses(monkeypatch, every=1)
    with pytest.raises(SystemExit) as e:
        G.main(_args(tiny, tmp_path / "run", extra=["--max-bad-steps", "3"]))
    assert e.value.code == 1          # the guard, not an argparse error (2)


def test_addon_lr_default_is_1e_4():
    # T34: 1e-3 (20x the backbone LR) kept inflating L2's weights; v1 used 3e-4
    assert G.build_parser().parse_args(["--variant", "sandbox", "--out", "x"]).addon_lr == 1e-4


def test_history_records_l2_size_for_sandbox(tiny, tmp_path):
    out = tmp_path / "run"
    G.main(_args(tiny, out))
    r = json.loads((out / "results.json").read_text())
    ends = [h for h in r["history"] if h.get("at") == 1.0]
    assert ends and all(h["l2_rms"] > 0 for h in ends)


def test_multisite_run_end_to_end(tiny, tmp_path):
    # v2.2: two L2 sites sharing one Sandbox train, checkpoint and report
    # one gate per site
    cfg = json.loads(open(tiny).read())
    cfg.update(addon_layers=[1, 2], addon_shared=True)
    path = tmp_path / "multi.json"
    path.write_text(json.dumps(cfg))
    out = tmp_path / "run"
    G.main(_args(str(path), out))
    r = json.loads((out / "results.json").read_text())
    assert len(r["test"]["gate_per_site"]) == 2
    assert all(len(h["gate_per_site"]) == 2 for h in r["history"] if h.get("at") == 1.0)
    w = torch.load(out / "last.pt", weights_only=False)["model"]
    k0 = [k for k in w if k.startswith("addon.sites.0.sandbox.")][0]
    assert torch.equal(w[k0], w[k0.replace("sites.0", "sites.1")])   # still shared
