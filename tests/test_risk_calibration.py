import numpy as np
import pytest
from scipy.optimize import linprog

from news_flywire.risk_v9 import MedianCalibration


def test_median_calibration_is_monotone_bounded_and_pools_ties():
    model = MedianCalibration().fit([0, 10, 10, 20, 30], [0, 90, 0, 20, 80])
    p = model.predict(np.linspace(-10, 110, 400))
    assert (np.diff(p) >= -1e-12).all()
    assert (p >= 0).all() and (p <= 100).all()
    assert model.predict([10, 10])[0] == model.predict([10, 10])[1]


def test_median_calibration_matches_independent_linear_program_optimum():
    # A separate LP checks the weighted L1 optimum, including tied inputs.
    rng = np.random.default_rng(45)
    for _ in range(15):
        x = rng.integers(0, 6, 12).astype(float)
        y = rng.integers(0, 101, 12).astype(float)
        w = rng.uniform(.1, 3, 12)
        model = MedianCalibration().fit(x, y, w)
        n = len(y)
        constraints, bounds = [], []
        for i in range(n):
            a = np.zeros(2*n); a[i] = 1; a[n+i] = -1
            constraints.append(a); bounds.append(y[i])
            a = np.zeros(2*n); a[i] = -1; a[n+i] = -1
            constraints.append(a); bounds.append(-y[i])
        for i in range(n):
            for j in range(n):
                if i != j and x[i] <= x[j]:
                    a = np.zeros(2*n); a[i] = 1; a[j] = -1
                    constraints.append(a); bounds.append(0.)
        optimum = linprog(np.r_[np.zeros(n), w], A_ub=constraints, b_ub=bounds,
                          bounds=[(0, 100)]*n+[(0, None)]*n, method='highs')
        assert optimum.success
        assert np.isclose(np.sum(w*np.abs(model.predict(x)-y)), optimum.fun, atol=1e-7)


@pytest.mark.parametrize('x,y,w', [([], [], None), ([1], [2, 3], None),
    ([np.nan], [0], None), ([0], [101], None), ([0], [0], [0])])
def test_calibration_rejects_invalid_data(x, y, w):
    with pytest.raises(ValueError):
        MedianCalibration().fit(x, y, w)


def test_calibration_constant_target_and_clipped_extrapolation():
    m = MedianCalibration().fit([0, 10, 20], [0, 0, 0])
    assert np.array_equal(m.predict([-10, 0, 1000]), [0, 0, 0])
