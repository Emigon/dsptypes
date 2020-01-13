import pytest
import numpy as np
import matplotlib.pyplot as plt

from fitkit import Signal1D
from fitkit.decimate import *

from pint import UnitRegistry
ureg = UnitRegistry()

@pytest.fixture
def sinusoid():
    fs = 20e3
    t = np.arange(0, 0.1, 1/fs)
    return Signal1D(np.sin(2*np.pi*1e3*t), xlims = (0*ureg('s'), 1*ureg('s')))

def test_decimate_by_derivative(sinusoid):
    N = 237
    subsample = decimate_by_derivative(sinusoid, N)

    assert len(subsample) == N
    for sample in subsample.values:
        assert sample in sinusoid.values
