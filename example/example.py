#!/usr/bin/python

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 8, 3

from sympy import *
from fitkit import *

# let's construct a model for a dampened 2nd order system
x, tau, w = symbols('x tau omega')
pm = Parametric1D(exp(-tau*x)*cos(w*x), {'tau': (0, 5, 10), 'omega': (100, 150, 200)})

t = np.linspace(0, 1, 1000)

y1 = pm(t)

y1.plot()
plt.savefig('y1.png')
plt.show()

# we can change the parameter values within the specified bounds and re-evaluate
# the model
pm.v['tau'] = 8
pm.v['omega'] = 100
y2 = pm(t)

y1.plot()
y2.plot()
plt.savefig('y1_and_y2.png')
plt.show()

# let's use the model to generate some realistic data
import numpy as np
np.random.seed(42)

y2 += np.random.normal(scale = .01, size = len(y1))
y2.plot()
plt.savefig('y2_noisy.png')
plt.show()

# we can fit the parametric model to the data
pm.v['tau'] = 5
pm.v['omega'] = 150
result = pm.fit(y2)

y2.plot()
pm(t).plot()
plt.savefig('y2_fitted.png')
plt.show()

# if we aren't happy with the fit we can manually tune in the parameters
sl, rd = pm.gui(t, persistent_signals = [y2])

# we can even try fitting the fft
sl, rd = pm.gui(t, fft = True, persistent_signals = [y2])

# suppose that we come up with a set of optimisation parameters we like: e.g.
opts = {'iters': 50, 'n': 2, 'sampling_method': 'sobol'}
result = pm.fit(y2, method = 'shgo', opts = opts)

# let's cook up some data to test it
from fitkit.datasheet import *
test_cases = sample(pm, 3, t, snr = 8) # sample the parameter space uniformly 3 times

for sig1d, mdata in test_cases:
    sig1d.plot()

plt.savefig('sampled.png')
plt.show()

# suppose we want to see how well we fit the frequency parameter omega
def metric(sig1d, mdata):
    fit_mdata = pm.fit(sig1d, method = 'shgo', opts = opts)
    percent = np.abs(fit_mdata['parameters']['omega'] - mdata['omega'])/mdata['omega']
    return percent, fit_mdata

table = apply_metric_to(sample(pm, 3, t, snr = 8), metric)
print(table.metric)

# often we want to see how well we fit a paramter as a function of the snr (in dB).
# fitkit makes this easy
snr_boxplot(pm, t, metric, [6, 8, 10], 12)
plt.savefig('snr_boxplot.png')
plt.show()
