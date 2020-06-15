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

from .call_types import _typefactory, _retrieve_x

default_transforms = {
    'real': lambda z: np.real(z),
    'imag': lambda z: np.imag(z),
    'abs':  lambda z: np.abs(z),
    'dB':   lambda z: 20*np.log10(np.abs(z)),
    'dBm':  lambda z: 10*np.log10(np.abs(z)) + 30,
    'rad':  lambda z: np.angle(z, deg=False),
    'deg':  lambda z: np.angle(z, deg=True),
}

class Parametric1D(object):
    def __init__(self, expr, params, call_type=np.complex128):
        """ Parametric1D
        Args:
            expr:       a sympy expression containing one free variable, and as
                        many parameters as you would like.
            params:     a dictionary of parameters of the following format
                            {key: (lower, setpoint, upper), ...}
                        key must be the symbol.name of one of the parameters in
                        expr. (lower, upper) determine the bounds of the
                        parameter space. setpoint must fall withing these bounds.
            call_type:  the type returned by Parametric1D.__call__. Supported
                        special types are xarray.DataArray and pd.Series that
                        can retain the x-axis the function was evaluated over.
        """
        self._call_type = call_type

        if not(isinstance(expr, tuple(sympy.core.all_classes))):
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

    def set(self, key, value, clip=False):
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
            '    %s' % self.expr,
            '',
            'parameters:',
            '    name     value        bounds',
            '    ------------------------------------------------',
        ]

        for p in self:
            str += ['    {0} {1:+.4e}  ({2:+.4e}, {3:+.4e})'
                    .format(p.ljust(8), self[p], self._l[p], self._u[p])]

        return '\n'.join(str)
    
    @_typefactory
    def __call__(self, x, parameters={}, clip=False):
        """ evaluate the parametric expression over x for the current parameter values """
        for key, val in parameters.items():
            self.set(key, val, clip=clip)
        parameter_values = [self[k] for k in self]
        if hasattr(x, '__iter__'):
            return [self.f(*parameter_values, pt) for pt in x]
        else:
            return self.f(*parameter_values, x)

    def default_errf(v, self, sigma, metric):
        try:
            for i, key in enumerate(key for key in self if key not in self._frozen):
                self.set(key, v[i], clip=False)
        except ValueError:
            warnings.warn('optimizer attempted to set parameter outside of bounds')
            return float('inf')

        return np.real(metric(self(_retrieve_x(sigma)), sigma))

    def default_metric(sigma1, sigma2):
        return sum((y1 - y2)**2 for y1, y2 in zip(sigma1, sigma2))

    def fit(self,
            sigma,
            method='Nelder-Mead',
            opts={},
            errf=default_errf,
            metric=default_metric):
        """ fit self to signal1D using scipy.minimize with self._errf

        Args:
            sigma:      the input signal to fit.
            method:     see scipy.minimize for local optimisation methods.
                        otherwise global methods 'differential_evolution', 'shgo'
                        and 'dual_annealing' are supported.
            errf:       an error function that accepts the (v, self, sigma, tform)
                        as arguments, where v is the ParameterDict associated with
                        self. as in scipy this functions return type must be a
                        real number.
            opts:       the keyword arguments to pass to scipy.minimize or the
                        'dual_annealing', 'shgo' or 'differential_evolution' global
                        optimizers.
            metric:     a function of two signals (self.__call__(x), sigma) that
                        returns a the value of the cost function. A sum of squares
                        of the residual signal (self.__call__(x) - sigma) is
                        provided as a default.
        Returns:
            results:    a dictionary containg the ParameterDict associated
                        with the optima, a copy of the signal that was fitted and
                        the optimisation result metadata. some of this information
                        is superfluous but assists in logging the fits to a large
                        set of inputs.
        """
        global_methods = {
            'differential_evolution':   spopt.differential_evolution,
            'shgo':                     spopt.shgo,
            'dual_annealing':           spopt.dual_annealing,
        }

        x0 = [self[k] for k in self if k not in self._frozen]
        b  = [(self._l[k], self._u[k]) for k in self if k not in self._frozen]

        args = (self, sigma, metric)
        if method in global_methods:
            opt_result = global_methods[method](errf, args=args, bounds=b, **opts)
        else:
            opt_result = spopt.minimize(errf, x0, args=args, method=method, **opts)

        if np.any(np.isnan(opt_result.x)):
            raise RuntimeError('Optimization failed to explore parameter space')

        # the last iteration isn't necessarily the global minimum so we set it here
        for i, key in enumerate(key for key in self if key not in self._frozen):
            self.set(key, opt_result.x[i], clip=True)

        return {
            'parameters':   deepcopy(self),
            'fitted':       sigma,
            'opt_result':   opt_result,
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
        else:
            if parameter_names in self._frozen:
                self._frozen.remove(parameter_names)

    def gui(self, x, data=[], transforms=default_transforms, **mpl_kwargs):
        """ construct a gui for the parameter space evaluated over x

        Args:
            x:          an iterable to evaluate the model over.
            data:       a list of signals that will be plotted in addtion
                        to self.
            transforms: a dictionary of named transformations that may be
                        applied to data and self. the plot will include a set of
                        check boxes allowing the user to select between
                        transforms. a default set of transforms is provided.
                        setting this to {} will not transform self or data and
                        no radio buttons will be generated.
            mpl_kwargs: **kwargs that will be passed to plt.plot.

        Returns:
            sliders:    So they don't get garbage collected and the gui doesn't
                        freeze.
            radio_btns: Again to avoid garbage collection. radio_btns are not
                        returned if transforms == {}.
        """
        self.reset_gui()

        if len(transforms) == 0:
            transforms = {'identity': lambda z : z}
        key = next(iter(transforms))

        fig, (ax1, ax2) = plt.subplots(nrows=2)

        # plot any provided data
        signals = []
        for sigma in data:
            trace = transforms[key](sigma)
            line, = ax1.plot(_retrieve_x(sigma), trace, **mpl_kwargs)
            signals += [(sigma, line)]

        # add the parametric model to the plot and construct the sliders
        self.add_to_current_axis(x, ax1, tform=transforms[key], **mpl_kwargs)
        sliders = self.construct_sliders(fig, ax2, x)

        # if there is only one defined transform, don't bother with the buttons
        if key == 'identity':
            fig.tight_layout()
            plt.show()
            return sliders

        # create radio buttons to allow user to switch between transforms
        divider = make_axes_locatable(ax1)
        rax = divider.append_axes("right", size = "15%", pad = .1)
        radio = RadioButtons(rax, transforms.keys(), active=0)

        def radio_update(key):
            # update the parametric trace and its transform
            axtop, tform, line = self._parametric_traces[0]
            axtop.set_ylabel(key)
            line.set_ydata(transforms[key](self(x)))
            self._parametric_traces[0] = (axtop, transforms[key], line)

            for sigma, line in signals:
                line.set_ydata(transforms[key](sigma))

            axtop.relim()
            axtop.autoscale_view()

            fig.canvas.draw_idle()

        radio.on_clicked(radio_update)

        fig.tight_layout()
        plt.show()

        return sliders, radio

    def reset_gui(self):
         """ resets internal state variables governing gui environments """
         self._parametric_traces = []

    def add_to_current_axis(self, x, ax, tform=lambda z : z, **mpl_kwargs):
        """ adds parameteric plot to axes 'ax' and registers update rule

        Args:
            x:          an iterable to evaluate self over (i.e. the x axis).
            ax:         the axis object to plot the data on.
            tform:      a transformation to apply to the signal before plotting.
            mpl_kwargs: **kwargs for plt.plot.
        """
        line, = ax.plot(x, tform(self(x)), **mpl_kwargs)
        self._parametric_traces.append((ax, tform, line))

    def construct_sliders(self, fig, ax, x, N=500):
        """ replace ax with parameter sliders. dynamic axes must be predefined

        Args:
            fig:        Figure object for subplots.
            ax:         Axes object to replace with Sliders.
            x:          an iterable to evaluate self over (i.e. the x axis).
            N:          the resolution of the sliders.

        Returns:
            sliders:    within a function, the user must keep a reference to the
                        sliders, or they will be garbage collected and the
                        associated gui will freeze.
        """
        divider = make_axes_locatable(ax)

        sl = {}
        for i, key in enumerate(key for key in self if key not in self._frozen):
            if i == 0:
                subax = ax
            else:
                subax = divider.append_axes("bottom", size = "100%", pad = .1)

            lo, hi = self._l[key], self._u[key]
            step = (hi - lo)/N
            sl[key] = Slider(subax, key, lo, hi, valinit=self[key], valstep=step)

        def update(event):
            for axis, tform, line in self._parametric_traces:
                trace = tform(self(x, parameters={key: sl[key].val for key in sl}))
                line.set_ydata(trace)
                axis.relim()
                axis.autoscale_view()

            fig.canvas.draw_idle()

        for slider in sl.values():
            slider.on_changed(update)

        return sl
