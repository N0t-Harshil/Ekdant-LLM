"""requirements.txt must list exactly the dependencies in pyproject.toml, so a
`pip install -r requirements.txt` and a `pip install -e .[dev]` resolve the
same versions (T30: an unpinned transformers broke the pod's PyTorch 2.2.2)."""

import os

import pytest

tomllib = pytest.importorskip("tomllib")  # Python >= 3.11

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _requirements():
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as f:
        lines = [line.split("#", 1)[0].strip() for line in f]
    return {line.replace(" ", "") for line in lines if line}


def _pyproject():
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        project = tomllib.load(f)["project"]
    deps = project["dependencies"] + project["optional-dependencies"]["dev"]
    return {d.replace(" ", "") for d in deps}


def test_requirements_txt_matches_pyproject():
    assert _requirements() == _pyproject()


def test_transformers_pinned_below_5_for_torch_2_2():
    # transformers 5.x needs PyTorch >= 2.6 and silently disables its model
    # classes on older torch; the training pod runs PyTorch 2.2.2.
    pins = [r for r in _pyproject() if r.startswith("transformers")]
    assert pins == ["transformers>=4.45,<5"]
