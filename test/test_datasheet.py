import pytest
import matplotlib.pyplot as plt

import sympy as sp

from fitkit.datasheet import *
from fitkit import Parametric1D

x, y, tau, alpha = sp.symbols('x y tau alpha')

@pytest.fixture
def model():
    # <symbol> : (lower limit, setpoint, upper limit)
    parameters = {
        'tau':    (0.0,   1.0,    5.0),
        'alpha':  (-10,   0,      10),
    }

    return Parametric1D(sp.exp(tau*x) + alpha, parameters)

def test_apply_metric_to(model):
    def lsq_metric(sig1d, mdata):
        shgo_opts = {'n': 1, 'iters': 1, 'sampling_method': 'sobol'}
        metadata = model.fit(sig1d, 'shgo', opts = shgo_opts)
        return model(sig1d.x)@sig1d, metadata

    x = np.linspace(0, 1, 100)
    table = apply_metric_to(sample(model, 4, x, snr = 20) , lsq_metric)

    assert len(table) == 4
    for col in table.columns:
        assert col in ['metric', 'parameters', 'fitted', 'opt_result']

def test_snr_sweep(model):
    def lsq_metric(sig1d, mdata):
        shgo_opts = {'n': 1, 'iters': 1, 'sampling_method': 'sobol'}
        metadata = model.fit(sig1d, 'shgo', opts = shgo_opts)
        return model(sig1d.x)@sig1d, metadata

    x = np.linspace(0, 1, 100)
    table = snr_sweep(model, x, lsq_metric, [15, 20], 4)

    assert len(table) == 8
    for col in table.columns:
        assert col in ['metric', 'snr', 'parameters', 'fitted', 'opt_result']

    for snr in table.snr.values:
        assert snr in [15, 20]

@pytest.mark.plot
def test_sample(model):
    for sigma, mdata in sample(model, 10, np.linspace(0, 1, 100), snr = 20):
        sigma.plot()
    plt.show()

@pytest.mark.plot
def test_snr_boxplot(model):
    def dummy_metric(sig1d, mdata):
        return np.abs(np.random.normal()), pd.Series({'fitted': sig1d})

    x = np.linspace(0, 1, 24)

    fig, axes = plt.subplots()
    plt.sca(axes)

    table = snr_sweep(model, x, dummy_metric, [15, 20], 4)
    data = snr_boxplot(table)

    fig.tight_layout()
    plt.show()
