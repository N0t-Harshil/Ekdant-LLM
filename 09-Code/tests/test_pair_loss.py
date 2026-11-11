"""T63: pair loss on the flipped pairs (PairCFR-style).  The two rows of a
pair (same question, conclusion as written / flipped, answers yes / no) share
a batch, and the yes / no margin at the first answer token must be larger on
the yes row than on its flipped twin.  No extra forward pass."""

import json

import pytest
import torch

import ekdant.train_gen as G
from ekdant.gen_data import GenExample, evidence_examples, pair_order


class _DS:
    """Stand-in for GenDataset: row r -> example r."""

    def __init__(self, ex):
        self.examples = ex

    def __len__(self):
        return len(self.examples)

    def example(self, r):
        return self.examples[r]


def _train():
    ex = [GenExample(f"does drug {i} work", "Drug %d can reduce pain." % i,
                     context=f"abstract {i}", label="yes") for i in range(10)]
    ex += [GenExample(f"is test {i} useful", "Test %d does not predict risk." % i,
                      context=f"abstract t{i}", label="no") for i in range(3)]
    return ex


def test_flipped_rows_share_a_pair_id_and_balanced_rows_have_none():
    rows, n = evidence_examples(_train(), seed=1)
    by_pair = {}
    for r in rows:
        if r.pair >= 0:
            by_pair.setdefault(r.pair, []).append(r)
        else:
            assert r.context.startswith("abstract")
    assert len(by_pair) == n["flip_pairs"] == 13
    for a, b in by_pair.values():
        assert a.question == b.question and {a.answer, b.answer} == {"yes", "no"}
    assert GenExample("q", "a").pair == -1


def test_pair_order_keeps_pairs_together_at_even_positions():
    rows, _ = evidence_examples(_train(), seed=1)
    rows.append(GenExample("odd one", "x", "c"))         # odd count of singles
    ds = _DS(rows)
    order = pair_order(ds, torch.Generator().manual_seed(3))
    assert sorted(order) == list(range(len(rows)))
    pos = {r: i for i, r in enumerate(order)}
    pairs = {}
    for r, e in enumerate(rows):
        if e.pair >= 0:
            pairs.setdefault(e.pair, []).append(pos[r])
    for a, b in pairs.values():
        assert abs(a - b) == 1 and min(a, b) % 2 == 0
    # with any even batch size a pair never straddles two batches
    for bs in (2, 4, 8):
        for a, b in pairs.values():
            assert a // bs == b // bs
    again = pair_order(ds, torch.Generator().manual_seed(3))
    other = pair_order(ds, torch.Generator().manual_seed(4))
    assert again == order and other != order


def test_pair_order_treats_a_half_pair_as_a_single_row():
    rows = [GenExample("q", "yes", "c", label="yes", pair=0),
            GenExample("q", "no", "d", label="no", pair=0),
            GenExample("r", "yes", "e", label="yes", pair=1)]   # twin dropped
    order = pair_order(_DS(rows), torch.Generator().manual_seed(0))
    assert sorted(order) == [0, 1, 2]
    assert abs(order.index(0) - order.index(1)) == 1


def _logits(margins, yes=5, no=7, vocab=10, T=4, first=2):
    lg = torch.zeros(len(margins), T, vocab)
    for r, m in enumerate(margins):
        lg[r, first - 1, yes] = m
    lab = torch.full((len(margins), T), -100)
    lab[:, first] = 1
    return lg, lab


def test_loss_is_small_when_the_answer_follows_the_evidence():
    # rows: pair 0 yes, pair 0 no, a balanced row, pair 1 yes (twin absent)
    pairs, ans = [0, 0, -1, 1], ["yes", "no", "yes. text", "yes"]
    good, lab = _logits([4.0, -4.0, 0.0, 0.0])
    same, _ = _logits([4.0, 4.0, 0.0, 0.0])            # same answer both times
    lg_good, n = G.pair_margin_loss(good, lab, pairs, ans, 5, 7)
    lg_same, _ = G.pair_margin_loss(same, lab, pairs, ans, 5, 7)
    assert n == 1
    assert lg_good.item() == pytest.approx(torch.nn.functional.softplus(torch.tensor(-8.0)).item())
    assert lg_same.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert lg_good < lg_same


def test_gradient_pushes_the_two_margins_apart():
    lg, lab = _logits([1.0, 1.0])
    lg.requires_grad_(True)
    loss, n = G.pair_margin_loss(lg, lab, [0, 0], ["yes", "no"], 5, 7)
    loss.backward()
    assert n == 1
    assert lg.grad[0, 1, 5] < 0 < lg.grad[1, 1, 5]      # raise yes-row margin
    assert lg.grad[0, 1, 7] > 0 > lg.grad[1, 1, 7]      # lower no-row margin


def test_no_complete_pair_gives_zero():
    lg, lab = _logits([1.0, 2.0])
    loss, n = G.pair_margin_loss(lg, lab, [0, 1], ["yes", "no"], 5, 7)
    assert n == 0 and loss.item() == 0.0


def test_flag_needs_evidence_and_an_even_batch(tiny, tmp_path):
    with pytest.raises(SystemExit):
        G.main(_args(tiny, tmp_path / "a", extra=["--pair-loss", "1"]))
    with pytest.raises(SystemExit):
        G.main(_args(tiny, tmp_path / "b",
                     extra=["--evidence", "--pair-loss", "1", "--batch-size", "3"]))
    with pytest.raises(SystemExit):
        G.main(_args(tiny, tmp_path / "c", extra=["--evidence", "--pair-loss", "-1"]))
    assert G.build_parser().parse_args(["--variant", "sandbox", "--out", "x"]).pair_loss == 0.0


def _pub_labeled(n=40):
    return [GenExample(f"does drug {i} work", f"Drug {i} can help patients.",
                       context=f"we studied drug {i}", label="yes" if i % 4 else "no")
            for i in range(n)]


def test_pair_run_trains_and_records_the_pair_loss(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(G, "load_pubmedqa", lambda limit=None, seed=17: _pub_labeled())
    monkeypatch.setattr(G, "pubmedqa_test_keys", lambda: set())
    out = tmp_path / "run"
    G.main(_args(tiny, out, extra=["--evidence", "--pair-loss", "1"]))
    r = json.loads((out / "results.json").read_text())
    stage1 = [h for h in r["history"] if h.get("stage") == "pubmedqa" and h.get("at") == 1.0]
    assert stage1 and stage1[-1]["pairs_used"] > 0
    assert stage1[-1]["pair_loss"] is not None
    ck = torch.load(out / "last.pt", weights_only=False)
    assert ck["run_args"]["pair_loss"] == 1.0


from test_train_gen import _args, tiny  # noqa: E402,F401  (shared fixture)
