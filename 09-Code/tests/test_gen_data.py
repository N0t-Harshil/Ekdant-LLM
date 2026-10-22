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
