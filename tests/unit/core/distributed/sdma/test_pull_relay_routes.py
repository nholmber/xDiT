"""The pull relay's choice of relay GPU (one leg per directed link) and its cut of a copy."""

import pytest

from xfuser.core.distributed.sdma.pull_relay import helpers, split


@pytest.mark.parametrize(
    "pairs",
    [
        [(0, 1), (2, 3)],  # CFG 2 x Ulysses 2 on 4 GPUs
        [(0, 4), (1, 5), (2, 6), (3, 7)],  # TP4 x Ulysses 2 on 8 GPUs
        [(0, 1), (2, 3), (4, 5), (6, 7)],  # CFG 4 x Ulysses 2 on 8 GPUs
        [(0, 1), (2, 3), (4, 5)],
    ],
)
def test_every_directed_link_carries_one_leg(pairs):
    helper = helpers(pairs)
    partner = {a: b for p in pairs for a, b in (p, p[::-1])}
    legs = []
    for s, r in partner.items():
        h = helper[s]
        assert h not in (s, r)
        legs += [(s, r), (s, h), (h, r)]
    assert len(legs) == len(set(legs)), "two legs share a directed link"


@pytest.mark.parametrize("n", [0, 1000, 480 << 10, (480 << 10) + 1, 123_456_789])
def test_split(n):
    direct, relayed = split(n, 0.375, 480 << 10)
    assert direct + relayed == n and relayed % 256 == 0
    assert (relayed == 0) == (n < (480 << 10))
    assert abs(relayed - 0.375 * n) < 256 or n < (480 << 10)
