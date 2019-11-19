""" parametric.py

author: daniel parker

defines the Parametric1D object. the user specifies some parameter space and a
sympy encoded parametric function, and this object allows to generate signals
from selected points in the parameter space, uniformly sample the parameter space
and also provides a convenient gui for model fitting or for determining realistic
parameter bounds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider, RadioButtons

import sympy
from sympy import *

from dsptypes import *
from .signal1D import *

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
    def __init__(self, expr, params):
        """ Parametric1D
        Args:
            expr:   a sympy expression containing one free variable, and as many
                    parameters as you would like
            params: a dictionary of parameters of the following format
                        {key: (lower, setpoint, upper), ...}
                    key must be the symbol.name of one of the parameters in expr.
                    (lower, upper) determine the bounds of the parameter space.
                    setpoint must fall withing these bounds
        """
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

        self._free_var = all_vars[0]

        for k in params:
            if not(isinstance(k, str)):
                raise TypeError(f'params key {k} must be of type str')

            if not(isinstance(params[k], tuple)):
                raise TypeError(f'params item {params[k]} must be of type tuple')

            if len(params[k]) != 3:
                raise TypeError(f'params item {params[k]} must be of length 3')

            if not(params[k][0] <= params[k][1] <= params[k][2]):
                raise ValueError(f'params item {params[k]} must be ascending')

        self.v = ParameterDict(params)

        # gui variables
        self._parametric_traces = []
        self._gui_style = 'real'

    def __call__(self, x, xunits = ureg(''), snr = np.inf, dist = np.random.normal):
        """ self(x) will give a Signal1D of the parametric model evaluate at x
        Args:
            x:      the points to evaluate expr over
            xunits: the units to apply to x
            snr:    signal to noise ratio in dB
            dist:   noise distribution

        Returns:
            Signal1D:   the parametric model evaluated for parameter setpoints and
                        x, with noise added if snr is specified. noise is useful
                        for algorithm testing
        """
        z = lambdify(self._free_var, self.expr.subs(self.v), "numpy")(x)

        if snr < np.inf:
            noise = dist(size = len(z))
            noise *= 10**(-snr/10) * np.std(z)**2 / np.std(noise)**2
            z += noise

        return Signal1D(z, xunits = xunits, xraw = x)

    def gui(self, x, fft = False, persistent_signals = [], **callkwargs):
        """ construct a gui for the parameter space evaluated over x

        Args:
            x:          an iterable to evaluate self over (i.e. the x axis)
            fft:        if True, plot the fft of the signal
            persistent_signals:
                        a list of Signal1D objects that will be plotted in addtion
                        to self
            callkwargs: **kwargs for self.__call__
        """
        fig, (ax1, ax2) = plt.subplots(nrows = 2)

        # plot any persistent signals
        plt.sca(ax1)
        psigs = []
        for sigma in persistent_signals:
            if fft:
                sigma = sigma.fft()
            psigs.append((sigma, sigma.plot(style = self._gui_style)))

        # add the parametric model to the plot and construct the sliders
        self.add_to_axes(ax1, x, self._gui_style, fft = fft, **callkwargs)
        sliders = self.construct_sliders(fig, ax2, x, fft = fft, **callkwargs)

        # create radio buttons to allow user to switch between plotting styles
        divider = make_axes_locatable(ax1)
        rax = divider.append_axes("right", size = "15%", pad = .1)
        idx = list(plotting_styles.keys()).index(self._gui_style)

        radio = RadioButtons(rax, plotting_styles.keys(), active = idx)

        def radio_update(key):
            self._gui_style = key

            trace = self(x, **callkwargs)
            if fft:
                trace = trace.fft()

            axtop, line = self._parametric_traces[0]
            line.set_ydata(plotting_styles[self._gui_style](trace.values))

            for sigma, line in psigs:
                line.set_ydata(plotting_styles[self._gui_style](sigma.values))

            axtop.relim()
            axtop.autoscale_view()

            fig.canvas.draw_idle()

        radio.on_clicked(radio_update)

        fig.tight_layout()
        plt.show()

    def add_to_axes(self, ax, x, style, fft = False, **callkwargs):
        """ adds parameteric plot to axes 'ax' in 'style' and registers update rule

        Args:
            x:          an iterable to evaluate self over (i.e. the x axis)
            ax:         matplotlib Axis object to plot the data on
            style:      plotting style as defined in Signal1D
            fft:        if True, plot the fft of the signal
            callkwargs: **kwargs for self.__call__
        """
        plt.sca(ax)
        trace = self(x, **callkwargs)
        if fft:
            trace = trace.fft()
        self._parametric_traces.append((ax, trace.plot(style = style)))

    def construct_sliders(self, fig, ax, x, fft = False, **callkwargs):
        """ add parameter sliders to ax. all axes to be update must be predefined

        Args:
            fig:        Figure object for subplots
            ax:         Axes object to replace with Sliders
            x:          an iterable to evaluate self over (i.e. the x axis)
            fft:        if True, plot the fft of the signal
            callkwargs: **kwargs for self.__call__

        Returns:
            sliders:    within a function, the user must keep a reference to the
                        sliders, or they will be garbage collected and the
                        associated gui will freeze
        """
        divider = make_axes_locatable(ax)

        sl = {}
        for i, (p, y) in enumerate(self.v.items()):
            if i == 0:
                subax = ax
            else:
                subax = divider.append_axes("bottom", size = "100%", pad = .5)
            lo, hi = self.v._l[p], self.v._u[p]
            sl[p] = Slider(subax, p, lo, hi, valinit = y, valstep = (hi - lo)/100)

        def update(event):
            for p in self.v:
                self.v[p] = sl[p].val

            trace = self(x, **callkwargs)
            if fft:
                trace = trace.fft()

            for ax2, line in self._parametric_traces:
                line.set_ydata(plotting_styles[self._gui_style](trace.values))
                ax2.relim()
                ax2.autoscale_view()

            fig.canvas.draw_idle()

        for slider in sl.values():
            slider.on_changed(update)

        return sl

    def sample(self, N, x, **callkwargs):
        """ sample self(x) for N uniformly sampled points in the parameter space """
        for _ in range(N):
            for p in self.v:
                self.v[p] = np.random.uniform(low = self.v._l[p], high = self.v._u[p])
            yield self(x, **callkwargs)
