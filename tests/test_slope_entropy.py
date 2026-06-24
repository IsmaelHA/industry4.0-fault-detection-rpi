import numpy as np
import pytest

from src.slope_entropy import slope_entropy


def test_constant_signal_is_low_entropy():
    x = np.ones(2048)
    assert slope_entropy(x, adaptive=False) < 0.05


def test_pure_sine_is_low_entropy():
    t = np.linspace(0, 1, 2048, endpoint=False)
    x = np.sin(2 * np.pi * 50 * t)
    assert slope_entropy(x, adaptive=False) < 0.6


def test_white_noise_is_high_entropy():
    # Adaptive thresholds are tuned for vibration (small first-differences).
    # For uncorrelated noise the difference distribution is much wider, so we
    # use fixed thresholds sized to the diff distribution (std(diff) ~= sqrt(2)).
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2048)
    assert slope_entropy(x, adaptive=False, delta=0.5, gamma=2.0) > 0.7


def test_two_noise_draws_agree():
    rng = np.random.default_rng(1)
    h1 = slope_entropy(rng.standard_normal(2048), adaptive=False, delta=0.5, gamma=2.0)
    h2 = slope_entropy(rng.standard_normal(2048), adaptive=False, delta=0.5, gamma=2.0)
    assert abs(h1 - h2) / h1 < 0.1


def test_white_noise_higher_than_sine_under_same_thresholds():
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(2048)
    t = np.linspace(0, 1, 2048, endpoint=False)
    sine = np.sin(2 * np.pi * 50 * t)
    kwargs = dict(adaptive=False, delta=0.05, gamma=0.5)
    assert slope_entropy(noise, **kwargs) > slope_entropy(sine, **kwargs)


@pytest.mark.parametrize("m", [3, 4, 5])
def test_m_sweep_returns_finite_values(m):
    rng = np.random.default_rng(2)
    x = rng.standard_normal(2048)
    h = slope_entropy(x, m=m)
    assert 0.0 < h < 1.0


def test_rejects_invalid_m():
    with pytest.raises(ValueError):
        slope_entropy(np.zeros(100), m=1)


def test_rejects_delta_ge_gamma():
    with pytest.raises(ValueError):
        slope_entropy(np.zeros(100), delta=0.1, gamma=0.1)


def test_zero_std_signal_returns_zero():
    h = slope_entropy(np.ones(2048), adaptive=True)
    assert h == 0.0
