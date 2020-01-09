import pytest
import numpy as np
import matplotlib.pyplot as plt

from fitkit import Signal1D, ureg

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
    y = Signal1D(sinusoid, xlims = (-1*ureg('us'), 10*ureg('us')))
    y = Signal1D(sinusoid, xcentspan = (0*ureg('Hz'), 10*ureg('Hz')))

def test_bad_xdef(sinusoid):
    with pytest.raises(Exception) as e_info:
        y = Signal1D(sinusoid, xlims = (0, 50), xcentspan = (25, 10))

    with pytest.raises(Exception) as e_info:
        y = Signal1D(sinusoid, xlims = (50, 0))

def test_x_init(sinusoid):
    y = Signal1D(sinusoid, xlims = (0*ureg('s'), 1*ureg('s')))
    assert np.all(y.x == np.linspace(0, 1, len(y), endpoint = False) * ureg('s'))

    y = Signal1D(sinusoid, xcentspan = (0, 2))
    assert np.all(y.x == np.linspace(-1, 1, len(y)))

def test_len(dc):
    y = Signal1D(dc)
    assert len(y) == len(dc)

def test_pwr(dc):
    y = Signal1D(dc)
    assert y.pwr == pytest.approx(len(dc))

def test_fs(sinusoid):
    y = Signal1D(sinusoid, xlims = (0*ureg('s'), 0.1*ureg('s')))
    assert y.fs.to('Hz').magnitude == pytest.approx(20e3)

def test_fft(sinusoid):
    y = Signal1D(sinusoid, xlims = (0*ureg('s'), 0.1*ureg('s')))
    fft = y.fft()

    # make sure that the peak frequency is at 1kHz as specified
    assert np.abs(fft.abs().idxmax()).to('kHz').magnitude == pytest.approx(1.0)

    assert y.pwr == pytest.approx(fft.pwr)

def test_metric(sinusoid, dc):
    y1 = Signal1D(sinusoid[:len(dc)])
    y2 = Signal1D(dc)

    assert y1@y1 == pytest.approx(0.0)
    assert y1@y2 == pytest.approx(1500)

@pytest.mark.plot
def test_plotz(sinusoid):
    y = Signal1D(sinusoid)
    y.plotz()
    plt.show()

@pytest.mark.plot
def test_plot(sinusoid):
    y = Signal1D(sinusoid, xlims = (0*ureg('s'), 1*ureg('s')))
    y.plot()
    plt.show()
    y.plot(style = 'abs')
    plt.show()

def test_copy(sinusoid):
    y1 = Signal1D(sinusoid)
    y2 = y1.copy()

    y2._z *= 0
    assert np.all(y2._z == 0)
    assert not(np.all(y1._z == 0))

def test_arithmetic(sinusoid):
    y = Signal1D(sinusoid)

    assert np.all((y + 5)._z == y._z + 5)
    assert np.all((5 + y)._z == y._z + 5)
    assert np.all((y - 6)._z == y._z - 6)
    assert np.all((7 - y)._z == y._z - 7)
    assert np.all((y * 20)._z == y._z * 20)
    assert np.all((15 * y)._z == y._z * 15)
    assert np.all((y / .1)._z == y._z / .1)
    assert np.all((.1 / (y + 3))._z == .1 / (y._z + 3))

    # test equality operator
    assert np.all(6*y + 4 != y)

def test_samples_above_below(sinusoid):
    y = Signal1D(sinusoid)

    y1 = y.samples_above(0.5, tform = 'real')
    y2 = y.samples_below(0.5, tform = 'abs')

    assert np.all(y1.real().values > 0.5)
    assert np.all(y2.abs().values < 0.5)
    assert len(y1.samples_below(0.5)) == 0
