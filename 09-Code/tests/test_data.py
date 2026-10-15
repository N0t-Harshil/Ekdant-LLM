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


def test_letter_from_answer_bounds():
    assert letter_from_answer(99, ["a", "b", "c", "d"]) in LETTERS
    assert letter_from_answer("99", ["a", "b", "c", "d"]) in LETTERS
    assert letter_from_answer(True, ["a", "b", "c", "d"]) in LETTERS


def test_normalise_options_dict_sorted():
    opts = normalise_options({"B": "two", "A": "one", "D": "four", "C": "three"})
    assert opts == ["one", "two", "three", "four"]


def test_format_prompt_template():
    p = format_prompt("What is 2+2?", ["3", "4", "5", "6"])
    assert p.endswith("Answer:")
    assert "\nA. 3\nB. 4\nC. 5\nD. 6" in p
