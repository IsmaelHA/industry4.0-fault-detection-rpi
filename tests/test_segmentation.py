import numpy as np
import pytest

from src.segmentation import n_windows, windows


def test_no_overlap_default_count():
    assert n_windows(20480, win=2048) == 10


def test_overlap_50pct_doubles_count_minus_one():
    assert n_windows(20480, win=2048, stride=1024) == 19


def test_window_shapes_are_uniform():
    sig = np.arange(20480, dtype=np.float32)
    shapes = {w.shape for w in windows(sig, win=2048)}
    assert shapes == {(2048,)}


def test_last_sample_of_last_window_is_correct():
    sig = np.arange(20480, dtype=np.int64)
    last = list(windows(sig, win=2048))[-1]
    assert last[-1] == 20479


def test_short_signal_returns_zero_windows():
    assert n_windows(1000, win=2048) == 0
    assert list(windows(np.zeros(1000), win=2048)) == []


def test_rejects_nonpositive_args():
    with pytest.raises(ValueError):
        n_windows(20480, win=0)
    with pytest.raises(ValueError):
        list(windows(np.zeros(20480), win=2048, stride=0))


@pytest.mark.parametrize("win,expected", [(512, 40), (1024, 20), (2048, 10), (4096, 5), (8192, 2)])
def test_candidate_window_sizes(win, expected):
    assert n_windows(20480, win=win) == expected
