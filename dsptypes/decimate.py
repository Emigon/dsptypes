""" decimate.py

author: daniel parker

this file contains methods to reduce the number of points in a Signal1D so as to
preserve sufficient information to adequately fit a Parametric1D model to the data.
the methods presented here are experimental so use at your own risk, but provide
the benefit of speeding up the evaluation of a cost function by the ratio of the
number of subsampled points to the original length of the data
"""

import numpy as np
from dsptypes import Signal1D

def decimate_by_derivative(sig1d, N, tform = lambda z : np.abs(z)):
    """ [EXPERIMENTAL] draws N random samples from sig1d about points of change

    Args:
        sig1d:      the input Signal1D type to decimate
        N:          the number of points to sample
        tform:      a transformation to apply to the sig1d to generate real
                    sample values used in the decimation procedure. np.abs is
                    applied by default
    Returns:
        sig1d:      a Signal1D type of length N containing samples that occur
                    in the input sig1d
    """
    if N > len(sig1d):
        return sig1d

    real = np.array(tform(sig1d.values), dtype = np.float64)

    # the thinking here is that if we increase sparsity between points where the
    # derrivative is 0, then we can safely interpolate between them
    probs = np.abs(np.diff(real))**.5
    probs /= np.sum(probs)

    # NOTE: the last sample can never be chosen since probs is derrived from a diff
    idxs = np.random.choice(range(len(probs)), size = N, p = probs, replace = False)
    idxs.sort()
    return Signal1D(sig1d.values[idxs], xraw = sig1d.x[idxs])
