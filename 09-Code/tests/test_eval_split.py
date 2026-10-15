"""Split-half benchmark protocol: seen = validation role, unseen = test.

The seen half drives per-epoch monitoring and best.pt selection; the
unseen half is evaluated exactly once, at the end. The permutation is
seeded by --seed so both A/B arms use identical halves.
"""

from ekdant.train import merge_eval, split_half


def test_split_half_partitions_disjointly_and_deterministically():
    ds = list(range(10))
    seen = split_half(ds, True, seed=17)
    unseen = split_half(ds, False, seed=17)
    assert sorted(list(seen) + list(unseen)) == list(range(10))
    assert set(seen).isdisjoint(set(unseen))
    assert len(seen) == len(unseen) == 5
    # same seed -> identical halves (both A/B arms must match)
    assert list(split_half(ds, True, seed=17)) == list(seen)
    assert list(split_half(ds, False, seed=17)) == list(unseen)
    # a different seed yields a different split
    assert list(split_half(ds, True, seed=99)) != list(seen)


def test_split_half_handles_odd_sizes():
    ds = list(range(11))
    seen = split_half(ds, True, 17)
    unseen = split_half(ds, False, 17)
    assert len(seen) == 5
    assert len(unseen) == 6
    assert sorted(list(seen) + list(unseen)) == list(range(11))


def test_merge_eval_weights_by_n():
    a = {"accuracy": 0.5, "loss": 2.0, "n": 3}
    b = {"accuracy": 1.0, "loss": 1.0, "n": 1}
    m = merge_eval(a, b)
    assert m["n"] == 4
    assert abs(m["accuracy"] - (0.5 * 3 + 1.0 * 1) / 4) < 1e-9
    assert abs(m["loss"] - (2.0 * 3 + 1.0 * 1) / 4) < 1e-9
