"""Row normalisation for both MCQ sources (this is where the
MedMCQA schema-drift bug lived: opa..opd/cop vs options/answer_idx)."""

from ekdant.data import (
    LETTERS,
    MCQExample,
    _normalise_row,
    format_prompt,
    letter_from_answer,
    normalise_options,
)


def _medmcqa_row(cop=0, question="Chronic urethral obstruction benign hyperplasia"):
    return {"id": "x", "question": question, "opa": "Hyperplasia",
            "opb": "Hypertrophy", "opc": "Atrophy", "opd": "Dysplasia",
            "cop": cop, "choice_type": "single"}


def _medqa_row():
    return {"question": "Which drug is safe in pregnancy for UTI?",
            "options": {"A": "Ampicillin", "B": "Ceftriaxone",
                        "C": "Doxycycline", "D": "Nitrofurantoin"},
            "answer_idx": "D", "answer": "Nitrofurantoin"}


def test_medmcqa_flat_schema_recognised():
    ex = _normalise_row(_medmcqa_row(cop=2), "medmcqa")
    assert isinstance(ex, MCQExample)
    assert ex.options == ["Hyperplasia", "Hypertrophy", "Atrophy", "Dysplasia"]
    assert ex.answer == "C"


def test_medmcqa_cop_is_zero_indexed():
    for cop, letter in enumerate(LETTERS):
        assert _normalise_row(_medmcqa_row(cop=cop), "medmcqa").answer == letter


def test_medqa_dict_schema_recognised():
    ex = _normalise_row(_medqa_row(), "medqa")
    assert isinstance(ex, MCQExample)
    assert ex.answer == "D"
    assert len(ex.options) == 4


def test_rejects_short_question():
    assert _normalise_row(_medmcqa_row(question="too short"), "medmcqa") is None


def test_rejects_unknown_schema():
    assert _normalise_row({"question": "some long enough question"}, "x") is None


def test_rejects_missing_cop():
    row = _medmcqa_row()
    del row["cop"]
    assert _normalise_row(row, "medmcqa") is None


def test_rejects_blank_option():
    row = _medmcqa_row()
    row["opd"] = "  "
    assert _normalise_row(row, "medmcqa") is None


def test_letter_from_answer_resolves():
    opts = ["Hyperplasia", "Hypertrophy", "Atrophy", "Dysplasia"]
    assert letter_from_answer(2, opts) == "C"
    assert letter_from_answer("D", opts) == "D"
    assert letter_from_answer("3", opts) == "D"
    assert letter_from_answer(2.0, opts) == "C"           # parquet float
    assert letter_from_answer("Atrophy", opts) == "C"      # option-text match


def test_letter_from_answer_unresolvable_is_none():
    # T18: MedMCQA's public test split has cop=-1 for every row; the old
    # silent "A" fallback turned the metric into A-prediction rate.
    opts = ["a", "b", "c", "d"]
    assert letter_from_answer(-1, opts) is None
    assert letter_from_answer(99, opts) is None
    assert letter_from_answer(True, opts) is None
    assert letter_from_answer("99", opts) is None
    assert letter_from_answer(1.5, opts) is None
    assert letter_from_answer("unknown text", opts) is None


def test_medmcqa_unlabelled_test_row_dropped():
    assert _normalise_row(_medmcqa_row(cop=-1), "medmcqa") is None


def test_load_examples_errors_on_fully_unlabelled_split(monkeypatch):
    import datasets as ds_mod
    import pytest
    from ekdant.data import load_examples

    monkeypatch.setattr(ds_mod, "load_dataset",
                        lambda *a, **k: [_medmcqa_row(cop=-1)] * 5)
    with pytest.raises(ValueError, match="no usable labelled rows"):
        load_examples("medmcqa", "test")


def test_normalise_options_dict_sorted():
    opts = normalise_options({"B": "two", "A": "one", "D": "four", "C": "three"})
    assert opts == ["one", "two", "three", "four"]


def test_format_prompt_template():
    p = format_prompt("What is 2+2?", ["3", "4", "5", "6"])
    assert p.endswith("Answer:")
    assert "\nA. 3\nB. 4\nC. 5\nD. 6" in p


# ---------------------------------------------------------------------------
# Eval-pipeline invariants: the 0.00%-accuracy bug was (a) target stored as a
# vocab id while pred is a letter index, and (b) reading logits[answer_pos]
# (predicts the token AFTER the answer) instead of logits[answer_pos - 1].
# ---------------------------------------------------------------------------

def _tiny_tokenizer(tmp_path):
    from ekdant.data import Tokenizer, bind_letter_ids, format_prompt

    texts = [format_prompt("which option is the correct one?",
                           ["alpha", "beta", "gamma", "delta"]) + f" {L}"
             for L in LETTERS] * 20
    tok = Tokenizer.train(texts, 500, str(tmp_path / "tok.json"))
    bind_letter_ids(tok)
    return tok


def test_letter_ids_bind_and_match_answer_token(tmp_path):
    from ekdant.data import LETTER_IDS, MCQDataset, collate

    tok = _tiny_tokenizer(tmp_path)
    assert len(LETTER_IDS) == 4 and len(set(LETTER_IDS)) == 4

    exs = [MCQExample("which option is the correct one?",
                      ["alpha", "beta", "gamma", "delta"], L)
           for L in LETTERS]
    ds = MCQDataset(exs, tok, max_len=256)
    assert len(ds) == 4
    for i, ex in enumerate(exs):
        row = ds.rows[i]
        assert row["letter_idx"] == LETTERS.index(ex.answer)
        # the token sitting at answer_pos IS the bound letter token
        assert row["input_ids"][row["answer_pos"]] == LETTER_IDS[row["letter_idx"]]

    batch = collate(ds.rows, tok.pad_id)
    # collate targets are letter indices 0..3, never vocab ids
    assert batch["answer_target"].tolist() == [0, 1, 2, 3]


def test_collate_carries_answer_vocab_ids(tmp_path):
    from ekdant.data import LETTER_IDS, MCQDataset, collate

    tok = _tiny_tokenizer(tmp_path)
    exs = [MCQExample("which option is the correct one?",
                      ["alpha", "beta", "gamma", "delta"], L)
           for L in LETTERS]
    ds = MCQDataset(exs, tok, max_len=256)
    batch = collate(ds.rows, tok.pad_id)
    # answer_ids are the letter tokens' vocab ids - the target the
    # --answer-weight CE term trains on (must match eval's LETTER_IDS)
    assert batch["answer_ids"].tolist() == LETTER_IDS
    for i, row in enumerate(ds.rows):
        assert int(batch["answer_ids"][i]) == row["answer_id"]


def test_evaluate_scores_answer_position_and_index(tmp_path):
    import torch
    from ekdant.data import LETTER_IDS, MCQDataset, collate
    from ekdant.train import evaluate

    tok = _tiny_tokenizer(tmp_path)
    exs = [MCQExample("which option is the correct one?",
                      ["alpha", "beta", "gamma", "delta"], L)
           for L in LETTERS]
    ds = MCQDataset(exs, tok, max_len=256)
    batch = collate(ds.rows, tok.pad_id)
    b, t, v = 4, batch["input_ids"].size(1), tok.vocab_size

    class _Out:
        def __init__(self, logits):
            self.logits, self.loss, self.aux = logits, torch.tensor(0.0), {}

    class _StubModel:
        def __init__(self, logits):
            self.logits = logits

        def eval(self):
            return self

        def __call__(self, input_ids, labels=None, **kwargs):
            return _Out(self.logits)

    def logits_with_peak_at(position_fn):
        logits = torch.zeros(b, t, v)
        for i in range(b):
            tgt_letter = int(batch["answer_target"][i])
            pos = position_fn(int(batch["answer_pos"][i]))
            logits[i, pos, LETTER_IDS[tgt_letter]] = 10.0
        return logits

    autocast = lambda: torch.autocast("cpu", enabled=False)

    # signal at logits[answer_pos - 1] (the position that PREDICTS the
    # answer) -> perfect accuracy with the fixed eval
    model = _StubModel(logits_with_peak_at(lambda p: p - 1))
    r = evaluate(model, [batch], torch.device("cpu"), autocast)
    assert r["accuracy"] == 1.0 and r["n"] == 4

    # signal at logits[answer_pos] (the old off-by-one position): the fixed
    # eval reads p-1, so this peak is invisible -> all-zero columns ->
    # argmax ties -> always predicts A -> accuracy == share of A (0.25 on
    # balanced targets). If the eval still read position p this would be 1.0.
    model = _StubModel(logits_with_peak_at(lambda p: p))
    r = evaluate(model, [batch], torch.device("cpu"), autocast)
    assert r["accuracy"] == 0.25
