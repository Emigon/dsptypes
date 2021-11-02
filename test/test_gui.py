import pytest
import matplotlib.pyplot as plt

import sympy as sp
import pandas as pd

from fitkit import *

@pytest.fixture
def models_and_data():
    x, y, tau, alpha = sp.symbols('x y tau alpha')

    parameters = {
        'tau':    (0.01, 1.0, 5.0),
        'alpha':  (-10, 1, 10),
    }
    pm1 = Parametric1D(sp.exp(tau*x) + alpha, parameters)
    parameters['tau'] = (0.001, 1.0, 6.0)
    pm2 = Parametric1D(alpha*sp.exp(2j*np.pi*tau*x), parameters)

    t = np.linspace(0, 1, 100)

    return t,\
           pm1,\
           pm2,\
           pm1(t) + 0.2*np.random.normal(size=100),\
           pm2(t) + 0.2*(1 + 1j)*np.random.normal(size=100)

@pytest.mark.plot
def test_gui_one_model(models_and_data):
    t, pm1, pm2, sigma1, sigma2 = models_and_data

    gui = Gui(parameter_resolution=100)
    ax = gui.register_model(pm1, t, color='r')
    gui.register_data(t, sigma1, axis=ax, color='b')
    ax.set_title('One model')
    gui.show()

@pytest.mark.plot
def test_transforms(models_and_data):
    t, pm1, pm2, sigma1, sigma2 = models_and_data

    gui = Gui(parameter_resolution=100)
    ax1 = gui.register_model(pm1, t, color='r')
    gui.register_data(t, sigma1, axis=ax1, color='b')

    ax2 = gui.register_model(pm2, t, color='r')
    gui.register_data(t, sigma2, axis=ax2, color='b')

    ax1.set_title('Transforms Test I')
    gui.show(transforms={'dB': lambda z : 10*np.log10(np.abs(z)), 're': np.real})

    ax1.set_title('Transforms Test II')
    gui.show(transforms=[{'sq': lambda z : z**2, 'inv': lambda z : z**-1},
                         {'dB': lambda z : 10*np.log10(np.abs(z)), 're': np.real}])

    ax1.set_title('Transforms Test III')
    gui.show(transforms=[{'sq': lambda z : z**2, 'inv': lambda z : z**-1}, None])

@pytest.mark.plot
def test_gui_two_model(models_and_data):
    t, pm1, pm2, sigma1, sigma2 = models_and_data

    gui = Gui(parameter_resolution=100, log_parameters=['tau'])
    ax1 = gui.register_model(pm1, t, color='r')
    gui.register_data(t, sigma1, axis=ax1, color='b')

    ax2 = gui.register_model(pm2, t, color='r')
    gui.register_data(t, sigma2, axis=ax2, color='b')

    ax1.set_title('Two models, two plots')
    params = gui.show()

@pytest.mark.plot
def test_gui_two_model_one_plot(models_and_data):
    t, pm1, pm2, sigma1, sigma2 = models_and_data

    gui = Gui(parameter_resolution=100)
    ax = gui.register_model(pm1, t, color='r')
    gui.register_model(pm2, t, axis=ax, color='g')
    gui.register_data(t, sigma1, axis=ax, color='b')
    gui.register_data(t, sigma2, axis=ax, color='b')

    ax.set_title('Two models, one plot')
    gui.show()

@pytest.mark.plot
def test_gui_reset(models_and_data):
    t, pm1, pm2, sigma1, sigma2 = models_and_data

    gui = Gui(parameter_resolution=100)
    ax1 = gui.register_model(pm1, t, color='r')
    gui.register_data(t, sigma1, axis=ax1, color='b')

    gui.reset()

    ax2 = gui.register_model(pm2, t, color='r')
    gui.register_data(t, sigma2, axis=ax2, color='b')

    ax2.set_title('Reset test (should be sinusoid)')
    gui.show()
