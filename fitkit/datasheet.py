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
            pm1d.v[p] = pint_safe_uniform(pm1d.v._l[p], pm1d.v._u[p])
        samples, mdata = pm1d(x, **callkwargs), deepcopy(pm1d.v)
        pm1d.v = v_init # reset the model back to its original state
        yield samples, mdata

def pint_safe_uniform(lo, hi, **kwargs):
    if hasattr(lo, 'units') and not(hasattr(hi, 'units')):
        raise TypeError('lo and hi must both be numeric or pint types')
    if hasattr(hi, 'units') and not(hasattr(hi, 'units')):
        raise TypeError('lo and hi must both be numeric or pint types')

    if hasattr(lo, 'units'):
        base_units = lo.to_base_units().units
        lo_base = lo.to_base_units().magnitude
        hi_base = hi.to_base_units().magnitude
        samples = np.random.uniform(lo_base, hi_base, **kwargs)

        return (samples*base_units).to(min([lo.units, hi.units]))
    else:
        return np.random.uniform(lo, hi, **kwargs)

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
        metric_tbl:     pd.DataFrame containing the metrics measured output by
                        metric
        metadata:       a list of metadata collected from the output of metric for
                        each of the test cases
    """
    metric_table, mdata_table = [], []
    for sig1d, sig1d_mdata in sampler:
        metric_dict, metadata = metric(sig1d, sig1d_mdata)
        metric_table.append(metric_dict)
        mdata_table.append(metadata)

    return pd.DataFrame(metric_table), mdata_table

def snr_sweep(pm1d, x, metric, snrs, N, sampler = sample, callkwargs = {}):
    """ apply_metric_to samples drawn from pm1d at the specified signal to noises

    Args:
        pm1d:       The Parametric1D model to draw samples from
        x:          The x-axis to evaluate the model over
        metric:     The metric to apply to each of the samples
        snrs:       A list of signal to noise ratios to draw samples with
        N:          The number of samples to draw for each signal to noise ratio
        sampler:    The sampler to use in the sweep. Must have the same interface
                    as fitkit.datasheet.sample
        callkwargs: The keyword arguments to pass to Parametric1D.__call__

    Returns:
        metric_tbl: Same as the table returned by apply_metric_to but with an
                    additional 'snr' column
        metadata:   A list of metadata collected for each of the SNRs sampled.
                    This metadata corresponds to metadata returned by
                    apply_metric_to.
    """
    dataset = []
    metadata = []
    for snr in snrs:
        callkwargs['snr'] = snr
        metrics, mdata = apply_metric_to(sampler(pm1d, N, x, **callkwargs), metric)
        metrics['snr'] = N*[snr]

        dataset.append(metrics)
        metadata.append(mdata)

    return pd.concat(dataset, ignore_index = True), metadata

def snr_boxplot(dataset, **boxkwargs):
    """ plot the distribution of metric values vs snr as a series of box plots

    Args:
        dataset:    The metric table. Must have an snr column and at least one
                    other numeric column.
        boxkwargs:  keyword arguments for the boxplot. 'positions', 'widths',
                    and 'whis' will be overwritten.

    Returns:
        fig, axes:  The matplotlib objects used to generate the figures. See
                    source code for the figure format.
    """
    fig, axes = plt.subplots(nrows = len(dataset.columns) - 1,
                             sharex = True,
                             figsize = (8, 2.5*(len(dataset.columns) - 1)))

    if len(dataset.columns) == 2:
        # then axes is not iterable. make it iterable
        axes = [axes]

    snrs = dataset.snr.unique()

    i = 0
    for col in dataset:
        if col == 'snr':
            continue
        # subdivide the data
        folded = np.array([dataset[dataset.snr == snr][col].values for snr in snrs])

        if len(snrs) > 1:
            w = np.diff(snrs).min()/2
        else:
            w = 1 # arbitrary because the plot will be pretty boring
        boxkwargs['positions'] = snrs
        boxkwargs['widths'] = len(snrs)*[w]
        boxkwargs['whis'] = [5, 95]
        axes[i].boxplot(folded.T, **boxkwargs)
        axes[i].set_yscale('log')

        axes[i].set_xlabel('SNR (dB)')
        axes[i].set_ylabel(f'Error in {col}')
        axes[i].set_xlim(np.min(snrs) - .6*w, np.max(snrs) + .6*w)
        i += 1

    fig.tight_layout()

    return fig, axes

def percentage_error_metric_creator(fitter, fitter_kwargs = {}):
    """ returns a metric for the percentage error in all estimated parameters

    Args:
        fitter:         the fitting method to fit the test case with. should
                        accept a Signal1D as it's first input followed by any
                        desired keyword arguments. must return the parameter
                        dictionary followed by any additional metadata you want
                        to log
        fitter_kwargs:  keyword arguments to pass to the fitter function

    Returns:
        metric_func:    a metric that is approriate to pass to apply_metric_to
                        or snr_boxplot, that measures the percentage error in
                        the fit to parameter using the specified fitter
    """
    def metric(sig1d, mdata):
        parameters, fit_mdata = fitter(sig1d, **fitter_kwargs)
        percentages = {}
        for p in parameters:
            percentages[p] = 100*np.abs((parameters[p] - mdata[p])/mdata[p])
            if hasattr(percentages[p], 'units'):
                percentages[p] = percentages[p].to_reduced_units()
        return percentages, fit_mdata
    return metric
