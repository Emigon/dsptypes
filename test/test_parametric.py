import pytest

from sympy import *
from sympy.abc import x, y, tau, alpha

from dsptypes import *

import matplotlib.pyplot as plt

@pytest.fixture
def model():
    # <symbol> : (lower limit, setpoint, upper limit)
    parameters = {
        'tau':    (0.0,   1.0,    5.0),
        'alpha':  (-10,   0,      10),
    }

    return {'expr': exp(tau*x) + alpha, 'params': parameters}

@pytest.fixture
def sinusoid():
    fs = 20e3
    t = np.arange(0, 0.1, 1/fs)
    return Signal1D(np.sin(2*np.pi*1e3*t), xunits = ureg('s'), xlims = (0, 1))

def test_correct_init(model):
    try:
        pm = Parametric1D(*model.values())
    except:
        assert False

def test_excess_free_variables(model):
    with pytest.raises(Exception) as e_info:
        pm = Parametric1D(y*model['expr'], model['params'])

def test_no_free_variables(model):
    with pytest.raises(Exception) as e_info:
        pm = Parametric1D(alpha**tau, model['params'])

def test_unused_params(model):
    with pytest.raises(Exception) as e_info:
        pm = Parametric1D(model['expr'] - alpha, model['params'])

def test_non_sympy(model):
    with pytest.raises(Exception) as e_info:
        pm = Parametric1D(lambda x : x**2, model['params'])

def test_parameter_loading(model):
    tmp_params = model['params'].copy()

    tmp_params['tau'] = (1, 0)
    with pytest.raises(Exception) as e_info:
        pm = Parametric1D(model['expr'], tmp_params)

    tmp_params['tau'] = 0
    with pytest.raises(Exception) as e_info:
        pm = Parametric1D(model['expr'], tmp_params)

    # make sure that the parameter keys are all strings
    tmp_params['tau'] = (0, 3, 6)
    tmp_params[tau] = tmp_params.pop('tau')
    with pytest.raises(Exception) as e_info:
        pm = Parametric1D(model['expr'], tmp_params)

    pm = Parametric1D(*model.values())
    for k in model['params']:
        assert pm.v[k] == model['params'][k][1]
        assert pm.v._l[k] == model['params'][k][0]
        assert pm.v._u[k] == model['params'][k][2]

def test_parameter_modification(model):
    pm = Parametric1D(*model.values())
    pm.v['alpha'] = 1
    assert pm.v['alpha'] == 1

    # try to set tau outside of the specified ranges. should raise an error
    with pytest.raises(Exception) as e_info:
        pm.v['tau'] = -1

    with pytest.raises(Exception) as e_info:
        pm.v['tau'] = 6

def test_call(model):
    pm = Parametric1D(*model.values())

    x = np.linspace(0, 1, 100)
    y_np = np.exp(model['params']['tau'][1]*x) + model['params']['alpha'][1]

    y = pm(x, xunits = ureg('s'))
    assert y_np == pytest.approx(y.values)

    np.random.seed(42)
    y_noisy = pm(x, xunits = ureg('s'), snr = 30)
    assert y_np == pytest.approx(y_noisy.values, rel = 1e-3)

@pytest.mark.plot
def test_gui(model, sinusoid):
    pm = Parametric1D(*model.values())
    pm.gui(np.linspace(0, 1, 100), xunits = ureg('s'),\
            persistent_signals = [sinusoid])
    pm.gui(np.linspace(0, 1, 100), fft = True, xunits = ureg('s'))

@pytest.mark.plot
def test_sample(model):
    pm = Parametric1D(*model.values())
    for sigma in pm.sample(10, np.linspace(0, 1, 100), snr = 20):
        sigma.plot()
    plt.show()
