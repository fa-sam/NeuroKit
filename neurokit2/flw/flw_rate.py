# -*- coding: utf-8 -*-

import pandas as pd
from .flw_peaks import flw_peaks
from ..signal.signal_rate import signal_rate
from ..rsp.rsp_rate import _rsp_rate_xcorr

def flw_rate(
    flw_cleaned,
    troughs=None,
    sampling_rate=1000,
    window=10,
    hop_size=1,
    method="trough",
    peak_method="khodadad2018",
    interpolation_method="monotone_cubic",
):
    """**Find respiration rate**

    Parameters
    ----------
    flw_cleaned : Union[list, np.array, pd.Series]
        The cleaned airflow respiration channel as returned by :func:`.flw_clean`.
    troughs : Union[list, dict, np.array, pd.Series, pd.DataFrame]
        The airflow respiration troughs as returned by :func:`.flw_peaks`.
        If None (default), troughs  will be automatically identified from the :func:`.flw_clean` signal.
    sampling_rate : int
        The sampling frequency of :func:`.flw_cleaned` (in Hz, i.e., samples/second).
    window : int
        The duration of the sliding window (in second). Default to 10 seconds.
    hop_size : int
        The number of samples between each successive window. Default to 1 sample.
    method : str
        Method can either be ``"trough"`` or ``"xcorr"``. In ``"trough"`` method, respiratory rate
        is calculated from the periods between successive inspirations (i.e., troughs).
        In ``"xcorr"`` method, cross-correlations between the changes in respiration with
        a bank of sinusoids of different frequencies are calculated to identify the principal
        frequency of oscillation.
    peak_method : str
        Method to identify successive respiratory inspirations, only relevant if method is
        ``"trough"``. Can be one of ``"khodadad2018"`` (default) or ``"biosppy"``.
    interpolation_method : str
        Method used to interpolate the rate between inhalation onsets.
        See :func:`.signal_interpolate`. ``"monotone_cubic"`` is chosen as the default
        interpolation method since it ensures monotone interpolation between data points (i.e., it
        prevents physiologically implausible "overshoots" or "undershoots" in the
        y-direction). In contrast, the widely used cubic spline interpolation does not ensure
        monotonicity.

    Return
    ------
    rsp_rate : np.ndarray
        Instantenous respiration rate.

    Example
    -------


    """

    if method.lower() in ["period", "peak", "peaks", "trough", "troughs", "signal_rate"]:
        if troughs is None:
            _, troughs = flw_peaks(flw_cleaned, sampling_rate=sampling_rate, method=peak_method)
        if isinstance(troughs, (pd.DataFrame, dict)):
            troughs = troughs["FLW_Troughs"]
        rate = signal_rate(
            troughs,
            sampling_rate=sampling_rate,
            desired_length=len(flw_cleaned),
            interpolation_method=interpolation_method,
        )

    elif method.lower() in ["cross-correlation", "xcorr"]:
        rate = _rsp_rate_xcorr(
            flw_cleaned,
            sampling_rate=sampling_rate,
            window=window,
            hop_size=hop_size,
            interpolation_method=interpolation_method,
        )

    else:
        raise ValueError("NeuroKit error: rsp_rate(): 'method' should be" " one of 'trough', or 'cross-correlation'.")

    return rate
