""" datasheet.py

author: daniel parker

provides methods to analyse the performance of fitting procedures based upon
Parametric1D types
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy

def sample(pm1d, N, x, **callkwargs):
    """ sample pm1d(x) for N uniformly sampled points in the parameter space """
    v_init = deepcopy(pm1d.v)
    for _ in range(N):
        for p in pm1d.v:
            pm1d.v[p] = np.random.uniform(pm1d.v._l[p], pm1d.v._u[p])
        samples, mdata = pm1d(x, **callkwargs), deepcopy(pm1d.v)
        pm1d.v = v_init # reset the model back to its original state
        yield samples, mdata

def apply_metric_to(sampler, metric):
    """ evalute metric on samples drawn from sampler

    Args:
        sampler:        a python generator that yields Signal1Ds and any associated
                        metadata
        metric:         a function that accepts a (Signal1D, metadata) pair and
                        returns a (metric_value, metadata) pair, whereby
                        metric_value is some positive number and the metadata object
                        is a consistent pd.Series object containing additional
                        information to log
    Returns:
        table:          pd.DataFrame with columns containing the metadata labels +
                        the metric_value for each sample. indices are integers
    """
    rows = []
    for sig1d, sig1d_mdata in sampler:
        val, metadata = metric(sig1d, sig1d_mdata)
        metadata['metric'] = val
        rows.append(metadata)

    return pd.DataFrame(rows)

def snr_sweep(pm1d, x, metric, snrs, N, **callkwargs):
    """ apply_metric_to samples drawn from pm1d at the specified signal to noises

    Args:
        pm1d:       The Parametric1D model to draw samples from
        x:          The x-axis to evaluate the model over
        metric:     The metric to apply to each of the samples
        snrs:       A list of signal to noise ratios to draw samples with
        N:          The number of samples to draw for each signal to noise ratio
        callkwargs: The keyword arguments to pass to Parametric1D.__call__

    Returns:
        table:      Same as the table returned by apply_metric_to but with an
                    additional 'snr' column
    """
    tables = []
    for snr in snrs:
        callkwargs['snr'] = snr
        sampler = sample(pm1d, N, x, **callkwargs)
        table = apply_metric_to(sampler, metric)
        table['snr'] = N*[snr]
        tables.append(table)

    return pd.concat(tables, ignore_index = True)

def snr_boxplot(pm1d, x, metric, snrs, N, **callkwargs):
    """ plot the distribution of metric values vs snr as a series of box plots

    Args:   same as datasheet.snr_sweep
    """
    dataset = snr_sweep(pm1d, x, metric, snrs, N)

    # subdivide the data
    folded = np.array([dataset[dataset.snr == snr].metric.values for snr in snrs])

    w = np.diff(snrs).min()/2
    plt.boxplot(folded.T, positions = snrs, widths = len(snrs)*[w])
    plt.xlabel('SNR (dB)')
    plt.ylabel('Metric')
    plt.ylim(-1e-2*dataset.metric.max())
    plt.xlim(np.min(snrs) - .6*w, np.max(snrs) + .6*w)
    plt.tight_layout()

    return dataset

def percentage_error_metric_creator(parameter, fitter, fitter_kwargs = {}):
    """ returns a metric that measures the percentage error in a parameter fit

    Args:
        parameter:      the name of the parameter to measure the error on
        fitter:         the fitting method to fit the test case with. should
                        accept a Signal1D as it's first input followed by any
                        desired keyword arguments
        fitter_kwargs:  keyword arguments to pass to the fitter function

    Returns:
        metric_function:a metric that is approriate to pass to apply_metric_to
                        or snr_boxplot, that measures the percentage error in
                        the fit to parameter using the specified fitter
    """
    def metric(sig1d, mdata):
        fit_mdata = fitter(sig1d, **fitter_kwargs)
        estimated = fit_mdata.parameters[parameter]
        percent = 100*np.abs(estimated - mdata[parameter])/mdata[parameter]
        return percent, fit_mdata
    return metric
