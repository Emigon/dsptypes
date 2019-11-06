""" signal1D.py

author: daniel parker
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dsptypes import *

plotting_styles = \
    {
        'dB':   lambda z : 10*np.log10(np.abs(z)),
        'dBm':  lambda z : 10*np.log10(np.abs(z)) + 30,
        'mag':  lambda z : np.abs(z),
        'rad':  lambda z : np.angle(z),
        'deg':  lambda z : np.deg2rad(np.angle(z)),
        'real': lambda z : np.real(z),
        'imag': lambda z : np.imag(z),
    }

class Signal1D(object):
    def __init__(self,
                 z,
                 xunits = ureg('samples'),
                 xlims = (None, None),
                 xcentspan = (None, None)):

        if not(hasattr(xunits, 'units')):
            raise TypeError('xunits must be a pint datatype')

        if (xcentspan != (None, None)) and (xlims != (None, None)):
            raise RuntimeError("Cannot specify both xcentspan and xlims")
        elif xcentspan != (None, None):
            cent, span = xcentspan
            self._x = np.linspace(cent- span/2, cent + span/2, len(z))*xunits
        elif xlims != (None, None):
            if xlims[0] > xlims[1]:
                raise ValueError("xlims must have format [a, b] with b > a")
            self._x = np.linspace(xlims[0], xlims[1], len(z), endpoint = False)*xunits
        else:
            self._x = np.arange(len(z))*xunits

        self.z = pd.Series(np.complex128(z), index = self.x.magnitude)

    def fft(self):
        z = np.fft.fft(self.z)/np.sqrt(len(self))
        f = np.fft.fftfreq(len(z))
        return Signal1D(z[np.argsort(f)], xunits = self.fs, xlims = (-.5, .5))

    def plot(self, style = 'real', normalise = False, xunits = None):
        xaxis = self.x
        if xunits is not None:
            xaxis = xaxis.to(xunits)

        plt.plot(xaxis, plotting_styles[style](self.z.values))
        plt.ylabel(style)

    def plotz(self):
        plt.scatter(np.real(self.z), np.imag(self.z))
        plt.xlabel('Re')
        plt.ylabel('Im')

    @property
    def x(self):
        return self._x

    @property
    def fs(self):
        return 1.0/(self.x[1] - self.x[0])

    @property
    def pwr(self):
        return (np.abs(self.z.values)**2).sum()

    def __len__(self):
        return len(self.z)

    def __repr__(self):
        return self.z.__repr__()
