import pytest

from sympy import *
from sympy.abc import x, y, tau, alpha

from dsptypes import Parametric1D

@pytest.fixture(scope = "session")
def model():
    # <symbol> : (lower limit, setpoint, upper limit)
    parameters = {
        'tau':    (0.0,   1.0,    5.0),
        'alpha':  (-10,   0,      10),
    }

    return {'expr': exp(tau*x) + alpha, 'params': parameters}

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

def test_snr_setup(model):
    pm = Parametric1D(*model.values(), minsnr = -3, maxsnr = 3)
    pm.v['snr'] = 0
    assert pm.v['alpha'] == 0

    # try to set snr outside of the specified ranges. should raise an error
    with pytest.raises(Exception) as e_info:
        pm.v['tau'] = 10

    # try to instantiate a model with impossible snr limits
    with pytest.raises(Exception) as e_info:
        pm = Parametric1D(*model.values(), minsnr = 3, maxsnr = -3)
