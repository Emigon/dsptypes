""" parametric.py

author: daniel parker
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sympy
from sympy import *

from dsptypes import *

from collections.abc import MutableMapping

class ParameterDict(MutableMapping):
    def __init__(self, params):
        self.store = dict([(k, v[1]) for k, v in params.items()])
        self._l = {k: params[k][0] for k in params}
        self._u = {k: params[k][2] for k in params}

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        if key not in self.store:
            raise RuntimeError(f'cannot add new keys to {type(self)}')

        if not(self._l[key] <= value <= self._u[key]):
            raise ValueError("parameter value must be between " + \
                                f"{self._l[key]} and {self._u[key]}")
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return key

class Parametric1D(object):
    def __init__(self, expr, params, minsnr = -6, maxsnr = 30):
        if not(isinstance(expr, tuple(sympy.core.all_classes))):
            raise TypeError("expr must be a sympy expression")

        self.expr = expr

        all_vars = [sym.name for sym in expr.free_symbols]
        for v in params.keys():
            if v not in all_vars:
                raise KeyError(f"{v} is an unused parameter")
            all_vars.remove(v)

        if len(all_vars) != 1:
            raise RuntimeError(f"expr does not contain only 1 free variable:{all_vars}")

        if "snr" in params:
            raise KeyError("snr is a reserved parameter")

        tmp = params.copy()
        tmp["snr"] = (minsnr, maxsnr, maxsnr)

        for k in tmp:
            if not(isinstance(k, str)):
                raise TypeError(f'params key {k} must be of type str')

            if not(isinstance(tmp[k], tuple)):
                raise TypeError(f'params item {tmp[k]} must be of type tuple')

            if len(tmp[k]) != 3:
                raise TypeError(f'params item {tmp[k]} must be of length 3')

            if not(tmp[k][0] <= tmp[k][1] <= tmp[k][2]):
                raise ValueError(f'params item {tmp[k]} must be ascending')

        self.v = ParameterDict(tmp)
