""" parametric.py

author: daniel parker

defines the Parametric1D object. the user specifies some parameter space and a
sympy encoded parametric function, and this object generates signals from
selected points in the parameter space and graphical fitting tools for model
fitting or for determining realistic parameter bounds for automated fitting
"""

import warnings
from copy import deepcopy
from collections.abc import MutableMapping

import numpy as np
import sympy
import scipy.optimize as spopt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider, RadioButtons

from .gui import *

class Parametric1D(MutableMapping):
    def __init__(self, expr, params):
        """ Parametric1D
        Args:
            expr:       a sympy expression containing one free variable, and as
                        many parameters as you would like.
            params:     a dictionary of parameters of the following format
                            {key: (lower, setpoint, upper), ...}
                        key must be the symbol.name of one of the parameters in
                        expr. (lower, upper) determine the bounds of the
                        parameter space. setpoint must fall withing these bounds.
        """
        if not(isinstance(expr, sympy.Expr)):
            raise TypeError("expr must be a sympy expression")

        self._expr = expr

        all_vars = [sym.name for sym in expr.free_symbols]
        for v in params.keys():
            if v not in all_vars:
                raise KeyError(f"{v} is an unused parameter")
            all_vars.remove(v)

        if len(all_vars) > 1:
            raise RuntimeError(f"expr does not contain only 1 free variable:{all_vars}")
        elif len(all_vars) == 1:
            self._free_var = all_vars[0]
        else:
            self._free_var = '_'

        for k in params:
            if not(isinstance(k, str)):
                raise TypeError(f'params key {k} must be of type str')

            if not(isinstance(params[k], tuple)):
                raise TypeError(f'params item {params[k]} must be of type tuple')

            if len(params[k]) != 3:
                raise TypeError(f'params item {params[k]} must be of length 3')

            if not(params[k][0] <= params[k][1] <= params[k][2]):
                raise ValueError(f'params item {params[k]} must be ascending')

        self._store = dict([(k, v[1]) for k, v in params.items()])
        self._l = {k: params[k][0] for k in params}
        self._u = {k: params[k][2] for k in params}

        self.f = sympy.lambdify([k for k in self] + [self._free_var], self._expr)

        self._frozen = []

        self._parametric_traces = []

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        if key not in self._store:
            raise RuntimeError(f'cannot add new keys to {type(self)}')

        if not(self._l[key] <= value <= self._u[key]):
            raise ValueError("parameter value must be between " + \
                                f"{self._l[key]} and {self._u[key]}")
        self._store[key] = value

    def set(self, key, value, clip=False, lower=None, upper=None):
        if lower is not None:
            self._l[key] = lower
        if upper is not None:
            self._u[key] = upper

        if not(clip):
            self[key] = value
        else:
            if self._l[key] > value:
                self[key] = self._l[key]
            elif self._u[key] < value:
                self[key] = self._u[key]
            else:
                self[key] = value

    def __delitem__(self, key):
        del self._store[key]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def __keytransform__(self, key):
        return key

    @property
    def expr(self):
        """ user can see the expression but can't modify it once initialised """
        return self._expr

    @property
    def parameters_with_bounds(self):
        """ property for getting the parameters in a dictionary """
        return {k: (self._l[k], p, self._u[k]) for k, p in self.items()}
 
    @property
    def parameters(self):
        """ property for getting the parameters in a dictionary """
        return {k: p for k, p in self.items()}

    # {{{ define arithmetic with multiple Parametric1D objects
    def _combine_parameters(self, other):
        params = {k: (self._l[k], self[k], self._u[k]) for k in self}
        for k in other:
            if k in params:
                # take the intersection of the lower and upper bounds
                warnings.warn('taking intersection of common parameter bounds')
                lower = max([params[k][0], other._l[k]])
                upper = min([params[k][2], other._u[k]])
                # force the setpoint to lie halfway between the parameter range
                setpoint = (upper - lower)/2
                params[k] = setpoint
            params[k] = (other._l[k], other[k], other._u[k])

        return params

    def __add__(self, other):
        if isinstance(other, Parametric1D):
            return Parametric1D(self.expr + other.expr, self._combine_parameters(other))
        else:
            # try using sympy to apply arithmetic to expression
            params = {k: (self._l[k], self[k], self._u[k]) for k in self}
            return Parametric1D(self.expr + other, params)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Parametric1D):
            return Parametric1D(self.expr - other.expr, self._combine_parameters(other))
        else:
            # try using sympy to apply arithmetic to expression
            params = {k: (self._l[k], self[k], self._u[k]) for k in self}
            return Parametric1D(self.expr - other, params)

    def __rsub__(self, other):
        return -1*self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, Parametric1D):
            return Parametric1D(self.expr * other.expr, self._combine_parameters(other))
        else:
            # try using sympy to apply arithmetic to expression
            params = {k: (self._l[k], self[k], self._u[k]) for k in self}
            return Parametric1D(self.expr*other, params)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Parametric1D):
            return Parametric1D(self.expr / other.expr, self._combine_parameters(other))
        else:
            # try using sympy to apply arithmetic to expression
            params = {k: (self._l[k], self[k], self._u[k]) for k in self}
            return Parametric1D(self.expr / other, params)
    # }}}

    def __repr__(self):
        str = [
            'Parametric1D object',
            'expr:',
            '%s' % sympy.pretty(self.expr),
            '',
            'parameters:',
            '    name     value        bounds',
            '    ------------------------------------------------',
        ]

        for p in self:
            str += ['    {0} {1:+.4e}  ({2:+.4e}, {3:+.4e})'
                    .format(p.ljust(8), self[p], self._l[p], self._u[p])]

        return '\n'.join(str)

    def __call__(self, x, parameters={}, clip=False):
        """ evaluate the parametric expression over x for the current parameter values """
        for key, val in parameters.items():
            self.set(key, val, clip=clip)
        return self.f(*(self[k] for k in self), x)

    def default_errf(v, self, t, sigma, metric):
        try:
            for i, key in enumerate(key for key in self if key not in self._frozen):
                self.set(key, v[i], clip=False)
        except ValueError:
            warnings.warn('optimizer attempted to set parameter outside of bounds')
            return float('inf')

        return np.real(metric(self(t), sigma))

    def default_metric(sigma1, sigma2):
        return sum((y1 - y2)**2 for y1, y2 in zip(sigma1, sigma2))

    def fit(self,
            t,
            sigma,
            method='Nelder-Mead',
            opts={},
            errf=default_errf,
            metric=default_metric):
        """ fit self to signal1D using scipy.minimize with self._errf

        Args:
            sigma:      the input signal to fit.
            method:     the optimization method to use, either 'curve_fit'
                        from scipy.optimize, any of scipy's local optimization
                        methods (see scipy.optimize.minimize), and some of
                        scipy's global methods are supported 
                        ('differential_evolution', 'shgo and 'dual_annealing').
                        when specifying 'curve_fit' the default least_squares
                        metric must be used. as we use curve_fit with parameter
                        boundaries, the 'trf' optimizer is applied under the 
                        hood.
            errf:       an error function that accepts (v, self, sigma, tform) 
                        as arguments, where v is the ParameterDict associated 
                        with self. as in scipy this functions return type must 
                        be a real number.
            opts:       the keyword arguments to pass to scipy.minimize or the
                        'dual_annealing', 'shgo' or 'differential_evolution' 
                        global optimizers.
            metric:     a function of two signals (self.__call__(x), sigma) that
                        returns a the value of the cost function. A sum of 
                        squares of the residual signal (self(x) - sigma) is 
                        provided as a default.

        Returns:
            results:    a dictionary containg the ParameterDict associated
                        with the optima, a copy of the signal that was fitted 
                        and the optimisation result metadata. some of this 
                        information is superfluous but assists in logging the 
                        fits to a large set of inputs.
        """
        global_methods = {
            'differential_evolution':   spopt.differential_evolution,
            'shgo':                     spopt.shgo,
            'dual_annealing':           spopt.dual_annealing,
        }

        x0 = [self[k] for k in self if k not in self._frozen]
        b  = [(self._l[k], self._u[k]) for k in self if k not in self._frozen]

        args = (self, t, sigma, metric)
        if method in global_methods:
            opt_result = global_methods[method](errf, args=args, bounds=b, **opts)
            fit_mdata = opt_result
        elif method == 'curve_fit':
            if metric is not Parametric1D.default_metric:
                raise RuntimeError('Cannot specify metric for curve_fit')
            if np.any(np.imag(sigma)):
                raise RuntimeError('Cannot curve_fit complex data')

            def cf_func(x, *params):
                operating_pt = {k: params[i] for i, k in enumerate(self) \
                                    if k not in self._frozen}
                return np.real(self(x, parameters=operating_pt))

            x, pcov = spopt.curve_fit(cf_func,
                                      t,
                                      np.real(sigma),
                                      p0=x0,
                                      bounds=np.array(b).T,
                                      **opts)
            opt_result = type('CFResult', (object,), {'x': x, 'pcov': pcov})
            fit_mdata = {'x': x, 'pcov': pcov}
        else:
            opt_result = spopt.minimize(errf, x0, args=args, method=method, **opts)
            fit_mdata = opt_result

        if np.any(np.isnan(opt_result.x)):
            raise RuntimeError('Optimization failed to explore parameter space')

        # the last iteration isn't necessarily the minimum so we set it here
        for i, key in enumerate(key for key in self if key not in self._frozen):
            self.set(key, opt_result.x[i], clip=True)
        
        return {
            'parameters':   {k: self[k] for k in self},
            'fitted':       sigma,
            'opt_result':   fit_mdata,
        }

    def freeze(self, parameter_names):
        """ freeze parameters so they are excluded from fitting procedures
        Args:
            parameter_names: a parameter name or a list of parameter names to
                             freeze.
        """
        if isinstance(parameter_names, list):
            for p in parameter_names:
                if p not in self._frozen:
                    self._frozen.append(p)
        elif parameter_names == '*':
            self._frozen = [p for p in self] # add all parameters with glob
        else:
            if parameter_names not in self._frozen:
                self._frozen.append(parameter_names)

    def unfreeze(self, parameter_names):
        """ unfreeze parameters that have been frozen

        Args:
            parameter_names: a parameter name or a list of parameter names to
                             unfreeze.
        """
        if isinstance(parameter_names, list):
            for p in parameter_names:
                if p in self._frozen:
                    self._frozen.remove(p)
        elif parameter_names == '*':
            self._frozen = [] # remove all parameters with glob
        else:
            if parameter_names in self._frozen:
                self._frozen.remove(parameter_names)

    def gui(self,
            x,
            data=[],
            transforms=None,
            parameter_resolution=100,
            savefig_dir=None):
        """ construct a gui for the parameter space evaluated over x

        Args:
            x:          an iterable to evaluate the model over.
            data:       a list of signals that will be plotted in addtion
                        to self.
            transforms: a dictionary of named transformations that may be
                        applied to data and self. the plot will include a set of
                        check boxes allowing the user to select between
                        transforms. a default set of transforms is provided.
                        setting this to None will not transform self or data and
                        no radio buttons will be generated unless the data is 
                        complex. In this case a default set of ['real', 'imag',
                        'abs', 'ang'] are provided.
            parameter_resolution:
                        the resolution of the sliders. the finer the resolution 
                        the smoother the control you have, but this may come at 
                        the expense of performance as the function is evaluated 
                        at every point the slider crosses.
            savefig_dir:
                        the directory to save an image of the current gui state 
                        when the user presses the 'save' button. defaults to 
                        None, disabling screenshots of the current layout.
        """
        gui = Gui(parameter_resolution=parameter_resolution)
        ax = gui.register_model(self, x)
        for x, y in data:
            gui.register_data(x, y, axis=ax)
        gui.show(transforms=transforms, savefig_dir=savefig_dir)
        plt.close()
