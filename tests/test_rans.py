"""rANS entropy coder round trips and actually compresses."""

import numpy as np
import pytest

from splatcraft.compress import rans


@pytest.mark.parametrize("n", [0, 1, 7, 2047, 2048, 100_003])
def test_roundtrip_skewed(n):
    rng = np.random.default_rng(n)
    data = np.clip(rng.normal(128, 6, n), 0, 255).astype(np.uint8)
    enc = rans.encode(data)
    assert np.array_equal(rans.decode(enc), data)
    if n > 10_000:
        assert len(enc) < 0.6 * n  # entropy is ~4.6 bits/symbol (0.58 bytes); rANS gets within ~1%


def test_roundtrip_uniform_and_single_symbol():
    rng = np.random.default_rng(0)
    u = rng.integers(0, 256, 50_000).astype(np.uint8)
    assert np.array_equal(rans.decode(rans.encode(u)), u)
    c = np.full(30_000, 42, np.uint8)
    enc = rans.encode(c)
    assert np.array_equal(rans.decode(enc), c)
    assert len(enc) < 2_000


def test_explicit_lane_counts():
    data = np.random.default_rng(3).integers(0, 40, 9_999).astype(np.uint8)
    for lanes in (1, 3, 64):
        assert np.array_equal(rans.decode(rans.encode(data, lanes=lanes)), data)


def test_normalized_frequencies_sum():
    counts = np.zeros(256)
    counts[[0, 5, 9]] = [1, 1_000_000, 3]
    f = rans.normalize_freqs(counts)
    assert f.sum() == 1 << rans.SCALE_BITS and (f[[0, 5, 9]] >= 1).all()
