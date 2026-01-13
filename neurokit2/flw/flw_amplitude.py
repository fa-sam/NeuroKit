# -*- coding: utf-8 -*-

import numpy as np

from ..signal import signal_interpolate
from ..rsp.rsp_fixpeaks import _rsp_fixpeaks_retrieve

def flw_amplitude(flw_cleaned, peaks, troughs=None, inspiration_onsets=None, method="standard", interpolation_method="monotone_cubic"):
    """**Compute respiratory amplitude from airflow**

    Compute respiratory amplitude given the raw respiration signal and its extrema. The
    **standard** method computes the amplitude as the difference between a peak and its preceding
    trough, while the **prepost** method computes the amplitude as the average of the differences
    of peak to its preceding and succeeding troughs (Cole, 2019).

    Parameters
    ----------
    flw_cleaned : Union[list, np.array, pd.Series]
        The cleaned respiration channel as returned by :func:`.flw_clean`.
    peaks : list or array or DataFrame or Series or dict
        The samples at which the respiration peaks occur. If a dict or a
        DataFrame is passed, it is assumed that these containers were obtained
        with :func:`flw_findpeaks`.
    troughs : list or array or DataFrame or Series or dict
        The samples at which the respiration troughs occur. If a dict or a
        is passed, it is assumed that these containers were obtained with :func:`.flw_findpeaks`.
    inspiration_onsets : list or array
        The samples at which the inspiration onsets occur. It is needed if the method is ``"max-min"``, otherwise
        it is not required.
    method : str
        The method to use to compute the amplitude. Can be ``"standard"`` or ``"prepost"`` or ``"max-min"``.
    interpolation_method : str
        Method used to interpolate the amplitude between peaks. See :func:`.signal_interpolate`.
        ``"monotone_cubic"`` is chosen as the default interpolation method since it ensures monotone
        interpolation between data point (i.e., it prevents physiologically implausible "overshoots"
        or "undershoots" in the y-direction). In contrast, the widely used cubic spline
        'interpolation does not ensure monotonicity.

    Returns
    -------
    array
        A vector containing the respiratory amplitude.

    See Also
    --------
    flw_clean, flw_peaks, signal_rate, flw_process, flw_plot, flw_symmetry

    Examples
    --------



    References
    ----------
    * Cole, S., & Voytek, B. (2019). Cycle-by-cycle analysis of neural oscillations. Journal of
      neurophysiology, 122(2), 849-861.

    """
    # Make sure `flw_cleaned` is a np.array
    flw_cleaned = np.array(flw_cleaned)

    if method == 'max-min':
        amplitude_per_breath = _max_min_amplitude(flw_cleaned, inspiration_onsets)
        # Interpolate amplitude to length of flw_cleaned.
        if len(inspiration_onsets) == 1:
            amplitude = np.full(flw_cleaned.shape, amplitude_per_breath[0])
        else:
            amplitude = signal_interpolate(inspiration_onsets[:-1], amplitude_per_breath, x_new=np.arange(len(flw_cleaned)),
                                           method=interpolation_method)
    else:
        amplitude_per_breath = _calculate_amplitude(flw_cleaned, peaks, troughs, method)
        # Interpolate amplitude to length of flw_cleaned.
        if len(peaks) == 1:
            amplitude = np.full(flw_cleaned.shape, amplitude_per_breath[0])
        else:
            amplitude = signal_interpolate(peaks, amplitude_per_breath, x_new=np.arange(len(flw_cleaned)),
                                           method=interpolation_method)

    info = {'FLW_Amplitude': amplitude_per_breath}

    return amplitude, info

def _max_min_amplitude(flw_cleaned, inspiration_onsets):
    if inspiration_onsets is None:
        raise ValueError('When using method="max-min", the inspiration_onsets cannot be None.')

    amplitudes = []
    for i in range(len(inspiration_onsets)-1):
        onset1 = inspiration_onsets[i]
        onset2 = inspiration_onsets[i+1]
        max_flow = max(flw_cleaned[onset1:onset2])
        min_flow = min(flw_cleaned[onset1:onset2])
        amplitude = max_flow - min_flow
        amplitudes.append(amplitude)
    return amplitudes

def _calculate_amplitude(flw_cleaned, peaks, troughs, method):

    # Format input.
    peaks, troughs = _rsp_fixpeaks_retrieve(peaks, troughs)

    peaks, troughs = _flw_fixpeaks_retrieve(peaks, troughs)

    # To consistently calculate amplitude, peaks and troughs must have the same
    # number of elements, and the first trough must precede the first peak.
    if (peaks.size != troughs.size):
        raise TypeError(
            "NeuroKit error: Please provide one of the containers returned by `flw_findpeaks()` "
            "as `extrema` argument and do not modify its content.",
        )

    # Calculate amplitude in units of the raw signal, based on vertical
    # difference of each peak to the preceding trough.
    amplitude = flw_cleaned[peaks] - flw_cleaned[troughs]

    # The above is the standard amplitude (each peak height to the preceding trough).
    if method in ["prepost"]:
        # Alternative amplitude calculation that corresponds to the average of
        # the peak height to the preceding and following troughs.
        # https://twitter.com/bradleyvoytek/status/1591495571269124096/photo/1
        # (Note that it cannot be done for the last peak)
        amplitude[0:-1] += flw_cleaned[peaks[0:-1]] - flw_cleaned[troughs[1::]]
        amplitude[0:-1] /= 2

    return amplitude

def _flw_fixpeaks_retrieve(peaks, troughs, sequence='peak-first'):
    if sequence == 'peak-first':
        if troughs[0] < peaks[0]:   # we expect first peak and then trough
            troughs = np.delete(troughs, 0)
    if sequence == 'trough-first':
        if peaks[0] < troughs[0]:
            peaks = np.delete(peaks, 0)

    min_len = min(len(peaks), len(troughs))
    peaks = peaks[:min_len]
    troughs = troughs[:min_len]
    return peaks, troughs
