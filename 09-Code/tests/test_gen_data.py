"""v2 generative data: prompts, answer-only labels, caps, and the
question-grouped MedQuAD split (spec 07-Management/30 §4)."""

import torch

from ekdant.gen_data import (
    GenDataset,
    GenExample,
    _rows_to_examples,
    format_gen_prompt,
    gen_collate,
    holdout,
    question_key,
    split_by_question,
)


class WordTok:
    """Whitespace 'tokenizer': one id per word; newlines are words too."""
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=False):
        words = text.replace("\n", " \n ").split(" ")
        ids = [self.vocab.setdefault(w, 3 + len(self.vocab)) for w in words if w]
        return {"input_ids": ids}


def test_prompt_strings_exact():
    assert format_gen_prompt(GenExample("Q?", "A.", context="C.")) == \
        "Question: Q?\nContext: C.\nAnswer:"
    assert format_gen_prompt(GenExample("Q?", "A.")) == "Question: Q?\nAnswer:"


def test_labels_mask_prompt():
    tok = WordTok()
    ds = GenDataset([GenExample("what is it", "a big answer")], tok)
    item = ds[0]
    ans = tok(" a big answer")["input_ids"]
    n_prompt = len(item["input_ids"]) - len(ans) - 1
    assert (item["labels"][:n_prompt] == -100).all()
    assert item["labels"][n_prompt:-1].tolist() == ans
    assert item["labels"][-1].item() == tok.eos_token_id


def test_long_answer_capped_and_fits():
    ds = GenDataset([GenExample("q", " ".join(f"w{i}" for i in range(5000)))],
                    WordTok())
    item = ds[0]
    assert int((item["labels"] != -100).sum()) == 512
    assert item["labels"][-1].item() != WordTok.eos_token_id
    assert len(item["input_ids"]) <= 1024


def test_context_trimmed_not_answer():
    ctx = " ".join(f"c{i}" for i in range(2000))
    ans = " ".join(f"a{i}" for i in range(40))
    ds = GenDataset([GenExample("q", ans, context=ctx)], WordTok())
    item = ds[0]
    assert len(item["input_ids"]) <= 1024
    assert int((item["labels"] != -100).sum()) == 41      # 40 words + EOS
    assert ds.dropped == 0


def test_empty_rows_dropped():
    rows = [{"q": "real question", "a": "real answer"},
            {"q": "", "a": "x"}, {"q": "   ", "a": "x"},
            {"q": "q2", "a": ""}, {"q": "q3", "a": "  \n "}]
    ex = _rows_to_examples(rows, lambda r: GenExample(r["q"], r["a"]))
    assert [e.question for e in ex] == ["real question"]


def test_split_by_question_no_leak_and_stable():
    ex = [GenExample(f"Question {i}?", f"a{i}") for i in range(400)]
    ex += [GenExample("  question 7? ", "dup answer"), GenExample("QUESTION 7?", "x")]
    ft, val, bench = split_by_question(ex, seed=17)
    keys = [{question_key(e.question) for e in s} for s in (ft, val, bench)]
    assert not (keys[0] & keys[1]) and not (keys[0] & keys[2]) \
        and not (keys[1] & keys[2])
    assert len(ft) + len(val) + len(bench) == len(ex)
    q7 = question_key("Question 7?")
    home = [i for i, k in enumerate(keys) if q7 in k]
    assert len(home) == 1
    assert 190 <= len(keys[2]) <= 210            # ~50 % of 400 keys
    assert 15 <= len(keys[1]) <= 25              # ~10 % of the fine-tune half
    again = split_by_question(ex, seed=17)
    assert [e.question for e in again[2]] == [e.question for e in bench]


def test_holdout_sizes():
    ex = [GenExample(f"q{i}", f"a{i}") for i in range(2500)]
    train, held = holdout(ex, n=2000, seed=17)
    assert len(held) == 2000 and len(train) == 500
    assert not {e.question for e in train} & {e.question for e in held}
    assert [e.question for e in holdout(ex, n=2000, seed=17)[1]] == \
        [e.question for e in held]


def test_collate_pads_and_masks():
    tok = WordTok()
    ds = GenDataset([GenExample("q one", "a"), GenExample("q", "a b c d")], tok)
    b = gen_collate([ds[0], ds[1]], pad_id=0)
    assert b["input_ids"].shape == b["labels"].shape == b["attention_mask"].shape
    pad = b["attention_mask"] == 0
    assert (b["labels"][pad] == -100).all() and pad.any()
    assert b["idx"].tolist() == [0, 1]
    assert ds.prompt_ids(0) == ds[0]["input_ids"][: len(ds.prompt_ids(0))].tolist()


def test_rows_stored_compactly():
    # 209k PubMedQA rows as Python int lists cost ~3.4 GB per arm process
    from array import array
    ds = GenDataset([GenExample("q", "a b")], WordTok())
    assert isinstance(ds.rows[0][0], array) and ds.rows[0][0].typecode == "i"
    assert ds[0]["input_ids"].dtype == torch.long


# ---------------------------------------------------------------------------
# Ekdant v0.1.0 protocol: reasoning text (stage 1) and MedLFQA (benchmark)
# ---------------------------------------------------------------------------

import datasets  # noqa: E402
import pytest  # noqa: E402

import ekdant.gen_data as GD  # noqa: E402
import ekdant.train_gen as G  # noqa: E402


def _fake_load(rows_by_name):
    def load(name, *args, split=None, **kw):
        return rows_by_name[name]
    return load


def test_reasoning_rows_become_reasoning_then_final_answer(monkeypatch):
    rows = [{"Question": "Why fever?", "Complex_CoT": "Infection raises the set point.",
             "Response": "Because of pyrogens."},
            {"Question": "", "Complex_CoT": "x", "Response": "y"}]
    monkeypatch.setattr(datasets, "load_dataset", _fake_load({GD.REASONING[0]: rows}))
    ex = GD.load_reasoning()
    assert len(ex) == 1
    assert ex[0].question == "Why fever?"
    assert ex[0].answer == "Infection raises the set point.\nFinal answer: Because of pyrogens."
    assert ex[0].context == ""


def test_reasoning_rows_too_long_are_dropped_not_cut():
    tok = WordTok()
    short = GenExample("q one", "a b c")
    long = GenExample("q two", " ".join(["w"] * 20))
    ds = GenDataset([short, long], tok, max_len=64, max_answer=10, cut_long=False)
    assert len(ds) == 1 and ds.example(0) is short
    assert ds.dropped_long == 1 and ds.dropped == 1
    cut = GenDataset([short, long], tok, max_len=64, max_answer=10)
    assert len(cut) == 2 and cut.dropped_long == 0


def test_medlfqa_rows_keep_their_statements(monkeypatch):
    rows = [{"Question": "Are boils curable?", "Free_form_answer": "Yes, usually.",
             "Must_have": ["Boils are skin infections."], "Nice_to_have": ["Warm compresses help."]}]
    monkeypatch.setattr(datasets, "load_dataset", _fake_load({GD.MEDLFQA: rows}))
    ex = GD.load_medlfqa()
    assert ex[0].answer == "Yes, usually."
    assert ex[0].must_have == ("Boils are skin infections.",)
    assert ex[0].nice_to_have == ("Warm compresses help.",)


def test_medlfqa_questions_never_reach_training():
    train = [GenExample("What is X?", "a"), GenExample("what is x? ", "b"),
             GenExample("What is Y?", "c")]
    bench = [GenExample("What is X?", "ref")]
    kept, removed = GD.exclude_questions(train, {question_key(e.question) for e in bench})
    assert [e.question for e in kept] == ["What is Y?"]
    assert removed == 2


def test_rows_to_examples_keeps_every_field():
    rows = [{"q": "q", "a": "a"}]
    ex = _rows_to_examples(rows, lambda r: GenExample(
        r["q"], r["a"], qtype="t", label="yes", pair=4, must_have=("m",), nice_to_have=("n",)))
    e = ex[0]
    assert (e.qtype, e.label, e.pair, e.must_have, e.nice_to_have) == ("t", "yes", 4, ("m",), ("n",))


def test_stage1_flag_defaults_to_pubmedqa():
    args = G.build_parser().parse_args(["--variant", "baseline", "--out", "x"])
    assert args.stage1 == "pubmedqa" and args.medlfqa is False
    args = G.build_parser().parse_args(["--variant", "baseline", "--out", "x",
                                        "--stage1", "reasoning", "--medlfqa"])
    assert args.stage1 == "reasoning" and args.medlfqa is True


def test_evidence_needs_pubmedqa_stage1(tmp_path):
    with pytest.raises(SystemExit) as e:
        G.main(["--variant", "baseline", "--out", str(tmp_path / "r"),
                "--stage1", "reasoning", "--evidence"])
    assert e.value.code == 1            # our check, not an argparse error (2)
