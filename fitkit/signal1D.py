""" signal1D.py

author: daniel parker

defines the Signal1D object. simply put, the Signal1D object behaves like a pandas
Series but with some extra functionality. the user has multiple (shorthand) ways
of definin the index. the index has associated units, which are automatically
added to plots, and there are a number of different ways to easily plot the signal.
additonally the fourier transform may be generated (with correct units) by calling
Signal1D.fft()
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from fitkit import *

plotting_styles = \
    {
        'dB':   lambda z: 20*np.log10(np.abs(z.values)),
        'dBm':  lambda z: 10*np.log10(np.abs(z.values)) + 30,
        'abs':  lambda z: np.abs(z.values),
        'rad':  lambda z: np.angle(z.values),
        'deg':  lambda z: np.deg2rad(np.angle(z.values)),
        'real': lambda z: np.real(z.values),
        'imag': lambda z: np.imag(z.values),
    }

def return_copy(name):
    def wrapper(obj, *args, **kwargs):
        cp = deepcopy(obj)
        if len(args) > 0 and type(args[0]) is Signal1D:
            cp._z = getattr(cp._z, name)(args[0]._z, *args[1:], **kwargs)
        else:
            cp._z = getattr(cp._z, name)(*args, **kwargs)
        return cp
    return wrapper

class Signal1D(object):
    def __init__(self,
                 z,
                 xlims = (None, None),
                 xcentspan = (None, None),
                 xraw = None):
        """ Signal1D

        Args:
            z:          the samples of the signal
            xlims:      (optional) specify the lower and upper bounds for the x
                        axis. this overrides xcentspan and xraw. e.g. (0, 1)
                        will set the xaxis to be linspace(0, 1, len(z)). pint
                        types for boundaries are compatible
            xcentspan:  (optional) specify the centre and span of the xaxis.
                        useful for resonace data. overrides xraw. e.g. (5, 10)
                        will centre x at 5 and will span 10 units. the xlims eqv.
                        would be (0, 10). pint types for boundaries are compatible
            xraw:       raw values for the x axis. use this if you want a
                        non-linearly spaced x axis. must have the same length as z.
                        pint arrays are also compatible
        """
        if (xcentspan != (None, None)) and (xlims != (None, None)):
            raise RuntimeError("Cannot specify both xcentspan and xlims")
        elif xcentspan != (None, None):
            cent, span = xcentspan
            self._x = np.linspace(cent - span/2, cent + span/2, len(z))
        elif xlims != (None, None):
            if xlims[0] > xlims[1]:
                raise ValueError("xlims must have format [a, b] with b > a")
            self._x = np.linspace(xlims[0], xlims[1], len(z), endpoint = False)
        elif xraw is not None:
            if not(hasattr(xraw, '__iter__')):
                raise TypeError('xraw must be iterable')
            if len(xraw) != len(z):
                raise ValueError('xraw must have the same len as z')
            self._x = xraw
        else:
            self._x = np.arange(len(z))

        if hasattr(z, 'units'):
            # strip them
            z = z.magnitude
            warnings.warn('Units for z are not currently supported.')

        idx = self.x.magnitude if hasattr(self.x, 'units') else self.x
        self._z = pd.Series(np.complex128(z), index = idx)

        # {{{ partially inherit from pandas Series
        setattr(Signal1D, '__len__', lambda obj: getattr(obj._z, '__len__')())
        setattr(Signal1D, '__repr__', lambda obj : getattr(obj._z, '__repr__')())
        setattr(Signal1D, 'sum', lambda obj: getattr(obj._z, 'sum')())
        setattr(Signal1D, 'min', lambda obj: getattr(obj._z, 'min')())
        setattr(Signal1D, 'max', lambda obj: getattr(obj._z, 'max')())

        setattr(Signal1D, '__add__', lambda obj, oth: return_copy('__add__')(obj, oth))
        setattr(Signal1D, '__radd__', lambda obj, oth: return_copy('__add__')(obj, oth))

        setattr(Signal1D, '__sub__', lambda obj, oth: return_copy('__sub__')(obj, oth))
        setattr(Signal1D, '__rsub__', lambda obj, oth: return_copy('__sub__')(obj, oth))

        setattr(Signal1D, '__mul__', lambda obj, oth: return_copy('__mul__')(obj, oth))
        setattr(Signal1D, '__rmul__', lambda obj, oth: return_copy('__mul__')(obj, oth))

        setattr(Signal1D, '__truediv__',
                lambda obj, oth: return_copy('__truediv__')(obj, oth))
        setattr(Signal1D, '__rtruediv__',
                lambda obj, oth: return_copy('__rtruediv__')(obj, oth))
        # }}}

    # {{{ partially inherit from pandas Series
    @property
    def values(self):
        return self._z.values

    @property
    def index(self):
        return self._z.index

    def abs(self):
        return Signal1D(np.abs(self._z.to_numpy()), xraw = self.x)

    def angle(self):
        return Signal1D(np.angle(self._z.to_numpy()), xraw = self.x)

    def real(self):
        return Signal1D(np.real(self._z.to_numpy()), xraw = self.x)

    def imag(self):
        return Signal1D(np.imag(self._z.to_numpy()), xraw = self.x)

    def __eq__(self, other):
        result = deepcopy(self)
        if type(other) is not Signal1D:
            raise TypeError('can only compare Signal1D to Signal1D')
        result._z = getattr(result._z, '__eq__')(other._z)
        return result

    def __ne__(self, other):
        result = deepcopy(self)
        if type(other) is not Signal1D:
            raise TypeError('can only compare Signal1D to Signal1D')
        result._z = getattr(result._z, '__ne__')(other._z)
        return result

    def idxmin(self):
        """ return the x value coresponding to the smallest real value in self """
        return self.x[np.argmin(np.real(self.values))]

    def idxmax(self):
        """ return the x value coresponding to the largest real value in self """
        return self.x[np.argmax(np.real(self.values))]
    # }}}

    def fft(self):
        """ returns a Signal1D of the fft of self """
        z = np.fft.fft(self.values)/np.sqrt(len(self))
        f = np.fft.fftfreq(len(z))
        return Signal1D(z[np.argsort(f)], xlims = (-.5*self.fs, .5*self.fs))

    def plot(self, style = 'real', xunits = None):
        """ adds a plot of self in specified style to the current axis

        Args:
            style:  string: 'dB' 'dBm' 'mag' 'rad' 'deg' 'real' 'imag'
            xunits: allows the user to convert the axis units (e.g. Hz to kHz)
        """
        xaxis = self.x
        if xunits is not None:
            xaxis = xaxis.to(xunits)

        if hasattr(self.x, 'magnitude'):
            unique = np.unique(self.x.magnitude)
        else:
            unique = np.unique(self.x)
           
        if len(unique) == len(self.x):
            line, = plt.plot(xaxis, plotting_styles[style](self))
        else:
            line, = plt.plot(xaxis, plotting_styles[style](self), 'o')

        plt.ylabel(style)
        plt.tight_layout()
        return line

    def plotz(self):
        """ scatter self on the complex plane of the current axis object """
        plt.scatter(np.real(self._z.to_numpy()), np.imag(self._z.to_numpy()))
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.tight_layout()

    def samples_above(self, val, tform = 'real'):
        idxs, = np.where(plotting_styles[tform](self) > val)
        return Signal1D(self._z.values[idxs], xraw = self.x[idxs])

    def samples_below(self, val, tform = 'real'):
        idxs, = np.where(plotting_styles[tform](self) < val)
        return Signal1D(self._z.values[idxs], xraw = self.x[idxs])

    @property
    def x(self):
        return self._x

    def copy(self):
        return deepcopy(self)

    @property
    def fs(self):
        return 1.0/min(np.diff(self.x))

    @property
    def pwr(self):
        return np.linalg.norm(self.values)**2

    def __matmul__(self, other):
        """ @ produces a symmetric goodness of fit metric """
        return np.linalg.norm(self.values - other.values)**2

    def normalise(self):
        return Signal1D(self._z.values/np.linalg.norm(self.values), xraw = self.x)
