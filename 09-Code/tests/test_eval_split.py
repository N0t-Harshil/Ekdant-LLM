"""Sequential 60/40 benchmark protocol.

Phase 1 trains on all of MedQA; phase 2 fine-tunes on the head partition
of MedMCQA train (MCQA_FT_FRAC = 60%); the complementary 40% tail is the
final benchmark and is never trained on. The single permutation is seeded
by --seed, so both A/B arms get the identical partition (fairness rule).
The untouched MedMCQA validation split (full, not partitioned) is the
checkpoint-selection set: best.pt vs last.pt are compared on it at the end.
"""

from ekdant.train import MCQA_FT_FRAC, split_frac


def test_split_frac_partitions_disjointly_and_deterministically():
    ds = list(range(10))
    head, tail = split_frac(ds, 0.6, seed=17)
    assert sorted(list(head) + list(tail)) == list(range(10))
    assert set(head).isdisjoint(set(tail))
    assert len(head) == 6 and len(tail) == 4
    # same seed -> identical partition (both A/B arms must match)
    assert list(split_frac(ds, 0.6, seed=17)[0]) == list(head)
    assert list(split_frac(ds, 0.6, seed=17)[1]) == list(tail)
    # a different seed yields a different partition
    assert list(split_frac(ds, 0.6, seed=99)[0]) != list(head)


def test_split_frac_sizes_truncate_down():
    for n, want_head, want_tail in ((7, 4, 3), (11, 6, 5), (1000, 600, 400)):
        head, tail = split_frac(list(range(n)), 0.6, seed=17)
        assert len(head) == want_head
        assert len(tail) == want_tail
        assert sorted(list(head) + list(tail)) == list(range(n))


def test_split_frac_fractions_and_protocol_constant():
    # the protocol is locked: 60% fine-tune / 40% benchmark
    assert MCQA_FT_FRAC == 0.6
    head, tail = split_frac(list(range(5)), MCQA_FT_FRAC, seed=17)
    assert len(head) == 3 and len(tail) == 2
