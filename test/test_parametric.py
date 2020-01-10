import pytest
import matplotlib.pyplot as plt

import sympy as sp

from fitkit import *

x, y, tau, alpha = sp.symbols('x y tau alpha')

@pytest.fixture
def model():
    # <symbol> : (lower limit, setpoint, upper limit)
    parameters = {
        'tau':    (0.0,   1.0,    5.0),
        'alpha':  (-10,   0,      10),
    }

    return {'expr': sp.exp(tau*x) + alpha, 'params': parameters}

@pytest.fixture
def model2():
    beta, gamma, x = sp.symbols('beta gamma x')

    # <symbol> : (lower limit, setpoint, upper limit)
    parameters = {
        'beta':   (1e2,   1e3,    1e4),
        'gamma':  (1,   2,      3),
    }

    return {'expr': gamma*sp.sin(beta*x), 'params': parameters}

@pytest.fixture
def sinusoid():
    fs = 20e3
    t = np.arange(0, 0.1, 1/fs)
    return Signal1D(np.sin(2*np.pi*1e3*t), xlims = (0*ureg('s'), 1*ureg('s')))

def test_correct_init(model):
    try:
        pm = Parametric1D(*model.values())
    except:
        assert False

def test_excess_free_variables(model):
    with pytest.raises(Exception) as e_info:
        pm = Parametric1D(y*model['expr'], model['params'])

def test_no_free_variables(model):
    # should work
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

    x = np.linspace(0, 1, 100)*ureg('s')
    y_np = np.exp(model['params']['tau'][1]*x.magnitude) + model['params']['alpha'][1]

    y = pm(x)
    assert y_np == pytest.approx(y.values)

    np.random.seed(42)
    y_noisy = pm(x, snr = 30)
    assert y_np == pytest.approx(y_noisy.values, rel = 1e-3)

def test_arithmetic_between_Pm1Ds(model, model2):
    pm1, pm2 = Parametric1D(*model.values()), Parametric1D(*model2.values())

    pm3 = pm1 + pm2
    pm4 = pm1*pm2
    pm5 = pm2/pm1

    alpha, beta, gamma, tau = sp.symbols('alpha beta gamma tau')

    for pm in [pm3, pm4, pm5]:
        for sym in [alpha, beta, gamma, tau]:
            assert sym.name in pm.v
            assert sym in pm.expr.free_symbols

    # test overloading one of the parameters with arithmetic
    model3 = {'expr': model2['expr'].subs(gamma, alpha),
              'params': {'beta': (1e2, 1e3, 1e4), 'alpha': (1, 2, 3)}}

    with pytest.warns(UserWarning):
        pm6 = pm1*Parametric1D(*model3.values())

    assert pm6.v._l['alpha'] == 1
    assert pm6.v._u['alpha'] == 3

def test_fishgo(model, sinusoid):
    pm = Parametric1D(*model.values())
    shgo_opts = {'n': 5, 'iters': 1, 'sampling_method': 'sobol'}
    table = pm.fit(sinusoid, 'shgo', opts = shgo_opts)

    assert type(table) == pd.Series
    for idx in table.index:
        assert idx in ['parameters', 'fitted', 'opt_result']
    assert type(table.parameters) == ParameterDict
    assert type(table.fitted) == Signal1D

@pytest.mark.plot
def test_gui(model, sinusoid):
    pm = Parametric1D(*model.values())
    pm.gui(np.linspace(0, 1, 100)*ureg('s'), persistent_signals = [sinusoid])
    pm.gui(np.linspace(0, 1, 100)*ureg('s'), fft = True)

@pytest.mark.plot
def test_fit(model, sinusoid):
    pm = Parametric1D(*model.values())
    shgo_opts = {'n': 5, 'iters': 1, 'sampling_method': 'sobol'}
    opt_result = pm.fit(sinusoid, 'shgo', opts = shgo_opts)
    pm.gui(sinusoid.x, persistent_signals = [sinusoid])
