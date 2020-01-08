import pytest
import matplotlib.pyplot as plt

import sympy as sp

from dsptypes.datasheet import *
from dsptypes import Parametric1D

x, y, tau, alpha = sp.symbols('x y tau alpha')

@pytest.fixture
def model():
    # <symbol> : (lower limit, setpoint, upper limit)
    parameters = {
        'tau':    (0.0,   1.0,    5.0),
        'alpha':  (-10,   0,      10),
    }

    return Parametric1D(sp.exp(tau*x) + alpha, parameters)

@pytest.mark.plot
def test_sample(model):
    for sigma in sample(model, 10, np.linspace(0, 1, 100), snr = 20):
        sigma.plot()
    plt.show()
