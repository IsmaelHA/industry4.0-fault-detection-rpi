import numpy as np

from src.analysis import control_limit_change_point, rms


def test_control_limit_rarely_trips_on_stationary_noise():
    rng = np.random.default_rng(0)
    false_trips = 0
    for _ in range(200):
        noise = rng.normal(0.6, 0.02, size=240)
        cp, _ = control_limit_change_point(noise, healthy_fraction=0.25,
                                           n_sigma=3.0, min_consecutive=3)
        if cp is not None:
            false_trips += 1
    # Requiring 3 consecutive points beyond 3-sigma makes false alarms very rare.
    assert false_trips / 200 < 0.05


def test_control_limit_detects_step_increase():
    series = np.concatenate([np.full(160, 0.60), np.full(80, 0.75)])
    cp, _ = control_limit_change_point(series, 0.25, 3.0, 3)
    assert cp is not None and 155 <= cp <= 175


def test_control_limit_detects_step_decrease():
    series = np.concatenate([np.full(150, 0.60), np.full(90, 0.40)])
    cp, _ = control_limit_change_point(series, 0.25, 3.0, 3)
    assert cp is not None and 145 <= cp <= 165


def test_single_spike_does_not_count_as_turning_point():
    rng = np.random.default_rng(2)
    series = rng.normal(0.6, 0.01, size=240)
    series[180] = 5.0  # one isolated spike, not a sustained excursion
    cp, _ = control_limit_change_point(series, 0.25, 3.0, 3)
    assert cp is None


def test_band_is_returned():
    series = np.concatenate([np.full(120, 0.6), np.full(120, 0.9)])
    cp, band = control_limit_change_point(series, 0.25, 3.0, 3)
    assert cp is not None
    assert band["upper"] >= band["mu0"] >= band["lower"]


def test_rms_basic():
    assert abs(rms(np.array([3.0, 4.0])) - np.sqrt(12.5)) < 1e-9
