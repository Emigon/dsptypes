""" datasheet.py

author: daniel parker

provides methods to analyse the performance of fitting procedures based upon
Parametric1D types
"""

import numpy as np

def sample(pm1d, N, x, **callkwargs):
    """ sample pm1d(x) for N uniformly sampled points in the parameter space """
    for _ in range(N):
        for p in pm1d.v:
            pm1d.v[p] = np.random.uniform(low = pm1d.v._l[p], high = pm1d.v._u[p])
        yield pm1d(x, **callkwargs)
