# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from ..rsp.rsp_findpeaks import _rsp_findpeaks_khodadad, _rsp_findpeaks_scipy, _rsp_findpeaks_biosppy

def flw_findpeaks(
    flw_cleaned,
    sampling_rate=100,
    method="fsa",
    amplitude_diff=0.3,
    peak_distance=0.8,
    peak_prominence=0.5,
    min_amplitude=1.5,
):
    """**Extract extrema in an airflow respiration (FLW) signal**

    Low-level function used by :func:`.flw_peaks` to identify troughs and peaks in a
    preprocessed airflow respiration signal using different sets of
    parameters. See :func:`.flw_peaks` for details.

    Parameters
    ----------
    flw_cleaned : Union[list, np.array, pd.Series]
        The cleaned airflow respiration channel as returned by :func:`.flw_clean`.
    sampling_rate : int
        The sampling frequency of :func:`.flw_cleaned` (in Hz, i.e., samples/second).
    method : str
        The processing pipeline to apply. Can be one of ``"fsa"`` (default),  ``"khodadad2018"``, ``"scipy"`` or
        ``"biosppy"``.
    amplitude_diff : float
        Only applies if method is ``"fsa"`` or ``"khodadad2018"``. Extrema that have a vertical distance smaller
        than(outlier_threshold * average vertical distance) to any direct neighbour are removed as
        false positive outliers. I.e., outlier_threshold should be a float with positive sign (the
        default is 0.3). Larger values of outlier_threshold correspond to more conservative
        thresholds (i.e., more extrema removed as outliers).
    peak_distance: float
        Only applies if method is ``"scipy"``. Minimal distance between peaks. Default is 0.8
        seconds.
    peak_prominence: float
        Only applies if method is ``"scipy"``. Minimal prominence between peaks. Default is 0.5.
    min_amplitude: float
        Only applies if method is ``"fsa"``. Minimum value for the absolute amplitude of peaks and troughs. Default
        value is 1.5 L/min. It should be set accordingly, if the flw_cleaned values are in another unit, e.g. L/s.

    Returns
    -------
    info : dict
        A dictionary containing additional information, in this case the samples at which peaks and troughs occur,
        accessible with the keys ``"FLW_Troughs"`` and ``"FLW_Peaks"``, respectively.

    See Also
    --------
    flw_clean, flw_peaks, signal_rate, flw_amplitude, flw_process, flw_plot

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      flw = nk.flw_simulate(duration=30, respiratory_rate=15)
      cleaned = nk.flw_clean(flw, sampling_rate=1000)
      info = nk.flw_findpeaks(cleaned)
      @savefig p_flw_findpeaks1.png scale=100%
      nk.events_plot([info["RSP_Peaks"], info["RSP_Troughs"]], cleaned)
      @suppress
      plt.close()

    """
    # Try retrieving correct column
    if isinstance(flw_cleaned, pd.DataFrame):
        try:
            flw_cleaned = flw_cleaned["FLW_Clean"]
        except NameError:
            try:
                flw_cleaned = flw_cleaned["FLW_Raw"]
            except NameError:
                flw_cleaned = flw_cleaned["FLW"]

    cleaned = np.array(flw_cleaned)

    # Find peaks
    method = method.lower()  # remove capitalised letters
    if method == 'fsa':
        info = _flw_findpeaks_fsa(cleaned, min_amplitude=min_amplitude, amplidute_diff=amplitude_diff)
    elif method in ["khodadad", "khodadad2018"]:
        info = _rsp_findpeaks_khodadad(cleaned, amplitude_min=amplitude_diff)
    elif method == "biosppy":
        info = _rsp_findpeaks_biosppy(cleaned, sampling_rate=sampling_rate)
    elif method == "scipy":
        info = _rsp_findpeaks_scipy(
            cleaned,
            sampling_rate=sampling_rate,
            peak_distance=peak_distance,
            peak_prominence=peak_prominence,
        )
    else:
        raise ValueError(
            "NeuroKit error: flw_findpeaks(): 'method' should be one of 'fsa', 'khodadad2018', 'scipy' or 'biosppy'."
        )

    return info

def _flw_findpeaks_fsa(flw_cleaned, min_amplitude=1.5, amplidute_diff=0.3):
    """
    The method is an adapted version of khodadad method, where the absolute amplitude of peaks and troughs
    are expected to be larger than a threshold (min_amplitude).
    The default value for the threshold is 1.5 L/min.
    https://iopscience.iop.org/article/10.1088/1361-6579/aad7e6/meta
    """

    extrema = _flw_findpeaks_extrema(flw_cleaned)
    extrema, amplitudes = _flw_findpeaks_outliers(flw_cleaned, extrema, min_amplitude=min_amplitude, amplitude_diff=amplidute_diff)
    peaks, troughs = _flw_findpeaks_sanitize(extrema, amplitudes)

    info = {"FLW_Peaks": peaks, "FLW_Troughs": troughs}
    return info


# =============================================================================
# Internals
# =============================================================================


def _flw_findpeaks_extrema(flw_cleaned):
    # Detect zero crossings (note that these are zero crossings in the raw
    # signal, not in its gradient).
    greater = flw_cleaned > 0
    smaller = flw_cleaned < 0
    risex = np.where(np.bitwise_and(smaller[:-1], greater[1:]))[0]
    fallx = np.where(np.bitwise_and(greater[:-1], smaller[1:]))[0]

    if risex[0] < fallx[0]:
        startx = "rise"
    elif fallx[0] < risex[0]:
        startx = "fall"

    allx = np.concatenate((risex, fallx))
    allx.sort(kind="mergesort")

    # Find extrema by searching minima between falling zero crossing and
    # rising zero crossing, and searching maxima between rising zero
    # crossing and falling zero crossing.
    extrema = []
    for i in range(len(allx) - 1):

        # Determine whether to search for minimum or maximum.
        if startx == "rise":
            if (i + 1) % 2 != 0:
                argextreme = np.argmax
            else:
                argextreme = np.argmin
        elif startx == "fall":
            if (i + 1) % 2 != 0:
                argextreme = np.argmin
            else:
                argextreme = np.argmax

        # Get the two zero crossings between which the extreme will be
        # searched.
        beg = allx[i]
        end = allx[i + 1]

        extreme = argextreme(flw_cleaned[beg:end])
        extrema.append(beg + extreme)

    extrema = np.asarray(extrema)
    return extrema


def _flw_findpeaks_outliers(flw_cleaned, extrema, min_amplitude=1.5, amplitude_diff=0.3):

    # Different from _rsp_findpeaks_outliers, we only consider those extrema whose absolute amplitude
    # is greater than a threshold given by min_amplitude parameter (default=1.5 liter/min).

    # first remove those extrema whose amplitude doesn't reach the threshold
    vertical_values = np.abs(flw_cleaned[extrema])
    min_amp = np.where(vertical_values > min_amplitude)[0]
    extrema = extrema[min_amp]

    # Then consider those extrema that have a minimum vertical distance to
    # their direct neighbor, i.e., define outliers in absolute amplitude
    # difference between neighboring extrema.
    vertical_diff = np.abs(np.diff(flw_cleaned[extrema]))
    median_diff = np.median(vertical_diff)
    min_diff = np.where(vertical_diff > (median_diff * amplitude_diff))[0]
    ind = np.append(min_diff, -1)   # keep the last extrema too. Otherwise, it will be removed without a reason
    extrema = extrema[ind]

    # Make sure that the alternation of peaks and troughs is unbroken. If
    # alternation of sign in extdiffs is broken, remove the extrema that
    # cause the breaks.
    amplitudes = flw_cleaned[extrema]
    extdiffs = np.sign(np.diff(amplitudes))
    extdiffs = np.add(extdiffs[0:-1], extdiffs[1:])
    removeext = np.where(extdiffs != 0)[0] + 1
    extrema = np.delete(extrema, removeext)
    amplitudes = np.delete(amplitudes, removeext)

    return extrema, amplitudes


def _flw_findpeaks_sanitize(extrema, amplitudes):
    # # To be able to consistently calculate breathing amplitude, make sure that
    # # the extrema always start with a trough and end with a peak, since
    # # breathing amplitude will be defined as vertical distance between each
    # # peak and the preceding trough. Note that this also ensures that the
    # # number of peaks and troughs is equal.
    # if amplitudes[0] > amplitudes[1]:
    #     extrema = np.delete(extrema, 0)
    # if amplitudes[-1] < amplitudes[-2]:
    #     extrema = np.delete(extrema, -1)
    # peaks = extrema[1::2]
    # troughs = extrema[0:-1:2]
    if amplitudes[0] > amplitudes[1]:
        #starts with peak
        peaks = extrema[0::2]
        troughs = extrema[1::2]
    else:
        peaks = extrema[1::2]
        troughs = extrema[0::2]

    return peaks, troughs
