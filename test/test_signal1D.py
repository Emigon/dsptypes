import pytest
import numpy as np
import matplotlib.pyplot as plt

from dsptypes import Signal1D, ureg

@pytest.fixture
def sinusoid():
    fs = 20e3
    t = np.arange(0, 0.1, 1/fs)
    return np.sin(2*np.pi*1e3*t)

@pytest.fixture
def dc():
    t = np.linspace(0, 0.1, 1000)
    return np.ones(t.shape, dtype = complex)

def test_correct_init(sinusoid):
    y = Signal1D(sinusoid)
    y = Signal1D(sinusoid, xunits = ureg('us'), xlims = (-1, 10))
    y = Signal1D(sinusoid, xunits = ureg('Hz'), xcentspan = (0, 10))

def test_bad_xdef(sinusoid):
    with pytest.raises(Exception) as e_info:
        y = Signal1D(sinusoid, xlims = (0, 50), xcentspan = (25, 10))

    with pytest.raises(Exception) as e_info:
        y = Signal1D(sinusoid, xlims = (50, 0))

def test_x_init(sinusoid):
    y = Signal1D(sinusoid, xlims = (0, 1))
    assert np.all(y.x == np.linspace(0, 1, len(y), endpoint = False) * ureg('samples'))

    y = Signal1D(sinusoid, xcentspan = (0, 2))
    assert np.all(y.x == np.linspace(-1, 1, len(y)) * ureg('samples'))

def test_len(dc):
    y = Signal1D(dc)
    assert len(y) == len(dc)

def test_pwr(dc):
    y = Signal1D(dc)
    assert y.pwr == pytest.approx(len(dc))

def test_fs(sinusoid):
    y = Signal1D(sinusoid, xunits = ureg('s'), xlims = (0, 0.1))
    assert y.fs.to('Hz').magnitude == pytest.approx(20e3)

def test_fft(sinusoid):
    y = Signal1D(sinusoid, xunits = ureg('s'), xlims = (0, 0.1))
    fft = y.fft()

    # make sure that the peak frequency is at 1kHz as specified
    assert np.abs(fft.abs().idxmax()).to('kHz').magnitude == pytest.approx(1.0)

    assert y.pwr == pytest.approx(fft.pwr)

def test_metric(sinusoid, dc):
    y1 = Signal1D(sinusoid[:len(dc)])
    y2 = Signal1D(dc)

    assert y1@y1 == pytest.approx(0.0)
    assert y1@y2 == pytest.approx(2.121320)

@pytest.mark.plot
def test_plotz(sinusoid):
    y = Signal1D(sinusoid)
    y.plotz()
    plt.show()

@pytest.mark.plot
def test_plot(sinusoid):
    y = Signal1D(sinusoid)
    y.plot()
    plt.show()
    y.plot(style = 'dBm')
    plt.show()