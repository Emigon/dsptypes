""" call_types.py

author: daniel parker

type management for Parametric1D
"""

def _typefactory(call_method):
    def _decorator(self, x, **kwargs):
        output = call_method(self, x, **kwargs)
        if not(hasattr(output, '__iter__')):
            return output

        if self._call_type.__module__ == 'pandas.core.series':
            return self._call_type(output, index=x)
        elif self._call_type.__module__ == 'xarray.core.dataarray':
            return self._call_type(output,
                                   coords={'x': x},
                                   dims='x',
                                   attrs={k: self[k] for k in self})
        else:
            return self._call_type(output)

    return _decorator

def _retrieve_x(signal):
    if type(signal).__module__ == 'pandas.core.series':
        return signal.index
    elif type(signal).__module__ == 'xarray.core.dataarray':
        return signal[signal.dims[0]].values
    else:
        return range(len(signal))
