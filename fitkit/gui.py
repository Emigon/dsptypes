""" gui.py

author: daniel parker

defines the Gui object. once the user registers one or more parameteric models
and datasets to present, this class will display an interactive GUI whereby the
user can manually tune in parameter values to fit a given dataset.
"""

from time import sleep
from os import path
from functools import partial
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider, Button, RadioButtons

from .call_types import _typefactory, _retrieve_x

default_complex_transforms = {
    'real': lambda z: np.real(z),
    'imag': lambda z: np.imag(z),
    'abs':  lambda z: np.abs(z),
    'arg':  lambda z: np.unwrap(np.angle(z, deg=False)),
}

class Gui(object):
    def __init__(self,
                 parameter_resolution=100,
                 log_parameters=[],
                 figsize=(8.25, 11.75)):
        """ Gui
        Args:
            parameter_resolution:
                defines the number of steps for each parameter on the sliders
                plotted. default = 100.
            log_parameters:
                a list of parameters that will be presented with log10 sliders
                for less sensitive parameter manipulation. default = [], which
                will apply the transform to none of the parameters.
            figsize:
                the size of the gui figure (relevant for saving the plot).
                default = A4.
        """
        self._figsize = figsize
        self._parameter_resolution = parameter_resolution
        self._log_parameters = log_parameters
        self._register_history = []

        self.reset()

    def register_model(self, pm, x, axis=None, **plot_kwargs):
        """ register a Parametric1D model with the gui
        Args:
            pm:     the Parametric1D model.
            x:      the x axis to plot pm(x) against.
            axis:   the axis object to plot the data to. default = None, meaning
                    that this function will insert a new axis below the top-most
                    axis.
            plot_kwargs:
                    the kwargs to pass to matplotlib.pyplot.plot. used to 
                    customise linestyles, colors, etc...
        Returns:
            axis:   the axis object that the data is plotted on.
        """
        # keep track of function calls
        self._register_history.append(
            (Gui.register_model, (pm, x), {'axis': axis}, plot_kwargs)
        )

        if axis is None:
            if len(self._models) >= 1 or len(self._data) >= 1:
                # add a new axis to the body below the most recent body axis
                ax1 = self._body_divider.append_axes("bottom", size="100%", pad=.5)
                ax2 = self._btns_divider.append_axes("bottom", size="100%", pad=.5)
                self._axes.append(ax1)
                self._ax_btns.append(ax2)
            line, = self._axes[-1].plot(x, np.real(pm(x)), **plot_kwargs)
        else:
            line, = axis.plot(x, np.real(pm(x)), **plot_kwargs)

        for param in pm:
            all_frozen = pm._frozen + \
                         [p for _, mod, _, _ in self._models for p in mod._frozen]
            if param in all_frozen:
                continue

            # setup the axis to add (or not add) the slider to
            if len(self._sldrs) > 0 and param not in self._sldrs:
                ax = self._sldr_divider.append_axes("bottom", size="100%", pad=.1)
                self._ax_sldr[param] = ax
            elif len(self._sldrs) == 0:
                self._ax_sldr[param] = self._sldr_init

            # take the log of the parameter if the user has set it on __init__
            txt = param
            if param in self._log_parameters and pm._l[param] > 0:
                lo, hi = np.log10(pm._l[param]), np.log10(pm._u[param])
                txt = f'log10({param})'
            elif param in self._log_parameters:
                raise ValueError('Cannot apply log to non-positive parameter range.')
            else:
                lo, hi = pm._l[param], pm._u[param]

            # take the intersection of parameter bounds from existing parameters
            # registered in the gui
            if param in self._sldrs:
                lo = max([lo, self._sldrs[param][0][2]])
                hi = min([hi, self._sldrs[param][0][3]])

            step = (hi - lo)/self._parameter_resolution
            init = np.log10(pm[param]) if param in self._log_parameters else pm[param]

            # store Slider (args, kwargs) for slider generation in self.show()
            self._sldrs[param] = ((self._ax_sldr[param], txt, lo, hi),
                                  {'valinit': init, 'valstep': step, 'valfmt': '%.3e'})

        ax = self._axes[-1] if axis is None else axis
        self._models.append((x, pm, line, ax))
        return ax

    def register_data(self, sigma, axis=None, **plot_kwargs):
        """ register some data with the gui
        Args:
            sigma:  the data to plot in the gui. typically this will be a
                    pandas.Series or an xarray.DataArray, where the index will
                    form the x-axis. intended usage will look like:
                    >>> ax = gui.register_data(sigma)
                    >>> gui.register_model(pm, sigma.index, axis=ax)

            axis:   the axis object to plot the data to. default = None, meaning
                    that this function will insert a new axis below the top-most
                    axis.
            plot_kwargs:
                    the kwargs to pass to matplotlib.pyplot.plot. used to 
                    customise linestyles, colors, etc...
        Returns:
            axis:   the axis object that the data is plotted on.
        """
        # keep track of function calls
        self._register_history.append(
            (Gui.register_data, (sigma,), {'axis': axis}, plot_kwargs)
        )

        if axis is None:
            if len(self._models) >= 1 or len(self._data) >= 1:
                # add a new axis to the body below the most recent body axis
                ax1 = self._body_divider.append_axes("bottom", size="100%", pad=.5)
                ax2 = self._btns_divider.append_axes("bottom", size="100%", pad=.5)
                self._axes.append(ax1)
                self._ax_btns.append(ax2)
            line, = self._axes[-1].plot(_retrieve_x(sigma), np.real(sigma), **plot_kwargs)
        else:
            line, = axis.plot(_retrieve_x(sigma), np.real(sigma), **plot_kwargs)

        ax = self._axes[-1] if axis is None else axis
        self._data.append((sigma, line, ax))
        return self._axes[-1]

    def reset(self):
        """ removes all registered data and models, and resets the figure. """
        # reset the plot state
        plt.close()
        self._fig = plt.figure(figsize=self._figsize)
        gs = self._fig.add_gridspec(3, 2, width_ratios=[6, 1], height_ratios=[1,16,8])

        self._ax_save = self._fig.add_subplot(gs[0,1])

        self._axes = [self._fig.add_subplot(gs[1,0])]
        self._body_divider = make_axes_locatable(self._axes[0])

        self._ax_btns = [self._fig.add_subplot(gs[1,1], aspect='equal')]
        self._btns_divider = make_axes_locatable(self._ax_btns[0])

        self._sldr_init = self._fig.add_subplot(gs[2,:])
        self._sldr_divider = make_axes_locatable(self._sldr_init)
        self._ax_sldr, self._sldrs = {}, {}

        # reset internal state
        self._models, self._data = [], []
        self._register_history = []
        self._param_list = [] # parameters container to return in self.show()
        self._save_idx = 0

    def _models_on_ax(self, ax):
        for x, pm, line, axis in self._models:
            if axis == ax:
                yield (x, pm, line, axis)

    def _data_on_ax(self, ax):
        for sigma, line, axis in self._data:
            if axis == ax:
                yield (sigma, line, axis)

    def _get_radio_update(self, ax, transforms):
        def radio_update(key):
            for x, pm, line, axis in self._models_on_ax(ax):
                line.set_ydata(transforms[key](pm(x)))
            for sigma, line, axis in self._data_on_ax(ax):
                line.set_ydata(transforms[key](sigma))

            ax.relim()
            ax.autoscale_view()

            self._fig.canvas.draw_idle()

        return radio_update

    def save(self, button, savefig_dir, fmtname, sliders, event):
        self._fig.canvas.draw_idle()

        button.label.set_text('↺') # unicode loading character
        button.label.set_fontsize(24)
        self._fig.canvas.draw()

        if savefig_dir is not None:
            self._fig.savefig(path.join(savefig_dir, fmtname % self._save_idx), dpi=96)
            self._save_idx += 1
        self._param_list.append({k: sliders[k].val for k in sliders})

        button.label.set_text('Save')
        button.label.set_fontsize(12)
        button.label.draw()

    def show(self, transforms=None, savefig_dir=None, fmtname='%d.png'):
        """ display the gui
        Args:
            transforms:
                a list of dictionaries of transforms that will be independently
                to each axis object. if more than one transform is specified 
                then a set of radio buttons will be provided to allow the user 
                to select between them. e.g. for two axis objects, you could 
                have:
                    [{'real': np.real, 'imag': np.imag}, {'arg': np.angle}]
            savefig_dir:
                the directory to save an image of the current gui state when the
                user presses the 'save' button.
            fmtname:
                a format string containing one '%d' placeholder that will define
                the filenames of the saved gui states. by default, each time the
                user presses the button, files will be created at:
                    {savefig_dir}/0.pdf
                    {savefig_dir}/1.pdf
                    ...
        Returns:
            parameters:
                a list of dictionaries containing the parameters of the model.
                each element of the list corresponds to a press of the 'save'
                button, except for the last element, which represents the state
                of the gui when it is closed.
        """
        # parse the transforms input
        if isinstance(transforms, dict):
            transforms = len(self._axes)*[transforms]
        elif transforms is None:
            transforms = len(self._axes)*[None]
        else:
            if len(transforms) != len(self._axes):
                raise ValueError('Transforms list must match number of axes.')

        for i, ax in enumerate(self._axes):
            if transforms[i] is not None:
                continue
            for x, pm, _, _ in self._models_on_ax(ax):
                if np.any(np.iscomplex(pm(x))):
                    transforms[i] = default_complex_transforms
            for sigma, _, _ in self._data_on_ax(ax):
                if np.any(np.iscomplex(sigma)):
                    transforms[i] = default_complex_transforms

        # create radio buttons to allow user to switch between transforms
        buttons = []
        for i, ax in enumerate(self._ax_btns):
            if transforms[i] is None:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                buttons.append(None)
                continue

            buttons.append(RadioButtons(ax, transforms[i].keys(), active=0))
            update_function = self._get_radio_update(self._axes[i], transforms[i])
            buttons[-1].on_clicked(update_function)
            update_function(buttons[-1].value_selected)
            for label in buttons[-1].labels:
                label.set_fontsize(12) # give the radio buttons larger labels

        # build the sliders for varying the parameters
        sliders = {}
        for param, (args, kwargs) in self._sldrs.items():
            sliders[param] = Slider(*args, **kwargs)

        def update(event):
            for i, btn, ax in zip(range(len(buttons)), buttons, self._axes):
                for x, pm, line, axis in self._models_on_ax(ax):
                    inv_log = lambda p, val: 10**val if p in self._log_parameters else val
                    params = {k: inv_log(k, sliders[k].val) for k in sliders if k in pm}
                    trace = pm(x, parameters=params)
                    if btn is None:
                        line.set_ydata(trace)
                    else:
                        line.set_ydata(transforms[i][btn.value_selected](trace))

                ax.relim()
                ax.autoscale_view()

            self._fig.canvas.draw_idle()

        for slider in sliders.values():
            slider.on_changed(update)

        # add in the save button
        save_button = Button(self._ax_save, f'Save')
        save_button.label.set_fontsize(12)
        save_button.on_clicked(
            partial(Gui.save, self, save_button, savefig_dir, fmtname, sliders)
        )

        # display the gui
        self._fig.tight_layout()
        self._fig.subplots_adjust(wspace=0.01)
        plt.show(block=True)

        # reset the gui so that the user can call self.show again straight away
        hist = [vect for vect in self._register_history] # copy the register history

        # figure out which groups of data and models needed to be plotted together
        axes_hist = map(lambda vect: vect[2]['axis'], hist)
        share_ax = [self._axes.index(ax) if ax in self._axes else None for ax in axes_hist]

        self.reset()
        ax_list = []
        for share_idx, (func, args, kwargs, plot_kwargs) in zip(share_ax, hist):
            kwargs['axis'] = ax_list[share_idx] if share_idx is not None else None
            ax = func(self, *args, **kwargs, **plot_kwargs)
            if share_idx is None:
                ax_list += [ax]

        # return the last set of parameters and any parameters 'saved' by the user
        self._param_list.append({k: sliders[k].val for k in sliders})
        return deepcopy(self._param_list)
