# -*- coding: utf-8 -*-
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning, find_closest
from ..stats import rescale
from ..rsp.rsp_fixpeaks import _rsp_fixpeaks_retrieve
from .flw_amplitude import _flw_fixpeaks_retrieve
from .flw_onsets import find_onsets, _flw_fix_onsets

def flw_symmetry(
    flw_cleaned,
    peaks,
    troughs=None,
    sampling_rate=100,
    show=False,
):
    """**Respiration Cycle Symmetry Features**

    Compute symmetry features of the respiration cycle, such as the Peak-Trough symmetry and the
    Rise-Decay symmetry (see Cole, 2019). Note that the values for each cycle are interpolated to
    the same length as the signal (and the first and last cycles, for which one cannot compute the
    symmetry characteristics, are padded).

    .. figure:: ../img/cole2019.png
       :alt: Figure from Cole and Voytek (2019).
       :target: https://journals.physiology.org/doi/full/10.1152/jn.00273.2019

    Parameters
    ----------
    flw_cleaned : Union[list, np.array, pd.Series]
        The cleaned respiration channel as returned by :func:`.flw_clean`.
    peaks : list or array or DataFrame or Series or dict
        The samples at which the inhalation peaks occur. If a dict or a DataFrame is passed, it is
        assumed that these containers were obtained with :func:`.flw_findpeaks`.
    troughs : list or array or DataFrame or Series or dict
        The samples at which the inhalation troughs occur. If a dict or a DataFrame is passed, it is
        assumed that these containers were obtained with :func:`.flw_findpeaks`.
    sampling_rate : int, optional (default=100) the sampling frequency of signal flw_cleaned in Hz,
    show : bool
        If True, show a plot of the symmetry features.

    Returns
    -------
    pd.DataFrame
        A DataFrame of same length as :func:`.flw_signal` containing the following columns:

        * ``"RSP_Symmetry_PeakTrough"``
        * ``"RSP_Symmetry_RiseDecay"``

    See Also
    --------
    flw_clean, flw_peaks, flw_amplitude, rsp_phase

    Examples
    --------
    .. ipython:: python


    References
    ----------
    * Cole, S., & Voytek, B. (2019). Cycle-by-cycle analysis of neural oscillations. Journal of
      neurophysiology, 122(2), 849-861.

    """
    # Format input.
    peaks, troughs = _rsp_fixpeaks_retrieve(peaks, troughs)
    # peaks, troughs = _flw_fixpeaks_retrieve(peaks, troughs)
    # Sanity checks -----------------------------------------------------------
    failed_checks = False
    if len(peaks) <= 2 or len(troughs) <= 2:
        warn(
            "Not enough peaks and troughs (signal too short?) to compute symmetry"
            " returning nan for symmetry.",
            category=NeuroKitWarning,
        )
        failed_checks = True

    if failed_checks:
        return {
            "FLW_Symmetry_RiseDecay": np.array([]),
            "FLW_Expiration_Symmetry": np.array([]),
            "FLW_Inspiration_Symmetry": np.array([]),
            "FLW_InspExp_Ratio": np.array([]),
        }

    # Compute symmetry features -----------------------------------------------
    # See https://twitter.com/bradleyvoytek/status/1591495571269124096/photo/1

    # Inspiration, Expiration symmetry
    onsets = find_onsets(flw_cleaned, sampling_rate)
    insp_onsets = onsets["FLW_InspirationOnsets"]
    exsp_onsets = onsets["FLW_ExpirationOnsets"]
    insp_onsets, exsp_onsets = _flw_fix_onsets(peaks, troughs, insp_onsets, exsp_onsets)
    exsp_onsets_backup = exsp_onsets.copy()

    exsp_symmetry = []
    insp_symmetry = []

    if insp_onsets[0] > exsp_onsets[0]:
        exsp_onsets = exsp_onsets[1:]
    for insp, exsp in zip(insp_onsets, exsp_onsets):
        # the ratio of inspiration-to-peak interval to the peak-to-expiration interval

        pk = peaks[(insp<peaks) & (peaks<exsp)]  # peak must be between insp and exsp onset
        if len(pk) >1:
            raise ValueError('Found more than one peak between inspiration onset and expiration onset.')
        elif len(pk)==1:
            pk = pk[0]
            insp_symmetry.append((pk-insp)/(exsp-pk))

    insp_exsp_ratios = []
    for i in range(len(insp_onsets)-1):
        insp1 = insp_onsets[i]
        exsp1 = exsp_onsets[i]
        insp2 = insp_onsets[i+1]
        insp_duration = exsp1 - insp1
        exsp_duration = insp2 - exsp1
        insp_exsp_ratio = insp_duration/exsp_duration
        insp_exsp_ratios.append(insp_exsp_ratio)


    exsp_onsets = exsp_onsets_backup  # in case it has been changed
    if insp_onsets[0] < exsp_onsets[0]:
        insp_onsets = insp_onsets[1:]
    for exsp, insp in zip(exsp_onsets, insp_onsets):
        # the ratio of expiration-to-trough interval to the trough-to-inspiration interval

        tr = troughs[(exsp<troughs) & (troughs<insp)]
        if len(tr) > 1:
            raise ValueError('Found more than one trough between inspiration onset and expiration onset.')
        elif len(tr)==1:
            tr = tr[0]
            exsp_symmetry.append((tr-exsp)/(insp-tr))

    if np.any(np.array(insp_symmetry) < 0):
        raise ValueError("Inspiration symmetry values must be positive.")

    if np.any(np.array(exsp_symmetry) < 0):
        raise ValueError("Expiration symmetry values must be positive.")

    # Rise-decay symmetry
    risedecay_symmetry = _compute_rise_decay_symmetry(peaks, troughs)

    # peaks, troughs = _flw_fixpeaks_retrieve(peaks, troughs, sequence='trough-first')
    # through_to_peak = peaks - troughs
    # peak_to_through = troughs[1:] - peaks[:-1]
    # risedecay_symmetry = through_to_peak[:-1] / (through_to_peak[:-1] + peak_to_through)

    if show is True:
        normalized = rescale(flw_cleaned)  # Rescale to 0-1
        plt.plot(normalized, color="grey", label="Respiration (normalized)")
        plt.scatter(peaks, normalized[peaks], color="red")
        plt.scatter(troughs, normalized[troughs], color="blue")
        plt.plot(peaks[1:], risedecay_symmetry , color="purple", label="Rise-decay symmetry")
        plt.legend()

    info = {
        "FLW_Symmetry_RiseDecay": risedecay_symmetry,
        "FLW_Expiration_Symmetry": np.asarray(exsp_symmetry),
        "FLW_Inspiration_Symmetry": np.asarray(insp_symmetry),
        "FLW_InspExp_Ratio": np.array(insp_exsp_ratios)
    }
    return info



def _compute_rise_decay_symmetry(peaks, troughs):
    """
    Compute rise-decay symmetry even if peaks and troughs have different lengths.

    Parameters:
        peaks (array-like): Indices of peaks
        troughs (array-like): Indices of troughs

    Returns:
        np.ndarray: Array of rise-decay symmetry values
    """

    through_to_peak = []
    peak_to_through = []

    # Compute trough → next peak distances
    for t in troughs:
        next_peak = peaks[peaks > t]
        if len(next_peak) > 0:
            through_to_peak.append(next_peak[0] - t)

    # Compute peak → next trough distances
    for p in peaks:
        next_trough = troughs[troughs > p]
        if len(next_trough) > 0:
            peak_to_through.append(next_trough[0] - p)

    # Convert to arrays
    through_to_peak = np.array(through_to_peak)
    peak_to_through = np.array(peak_to_through)

    # Match lengths for symmetry calculation
    min_len = min(len(through_to_peak), len(peak_to_through))
    risedecay_symmetry = through_to_peak[:min_len] / (through_to_peak[:min_len] + peak_to_through[:min_len])

    return risedecay_symmetry

def _compute_peak_trough_symmetry(halfway_locations, halfway_locations2):
    """
    Compute peak-trough symmetry without assuming equal array lengths.

    Parameters:
        halfway_locations (array-like): First set of halfway points
        halfway_locations2 (array-like): Second set of halfway points

    Returns:
        np.ndarray: Array of peak-trough symmetry values
    """

    asc_to_desc = []
    desc_to_asc = []

    # Iterate through consecutive pairs dynamically
    for i in range(len(halfway_locations) - 1):
        # Find next element in halfway_locations2 after halfway_locations[i]
        next_desc = halfway_locations2[halfway_locations2 > halfway_locations[i]]
        if len(next_desc) > 0:
            asc_to_desc.append(next_desc[0] - halfway_locations[i + 1])

    for i in range(len(halfway_locations2) - 1):
        # Find next element in halfway_locations after halfway_locations2[i]
        next_asc = halfway_locations[halfway_locations > halfway_locations2[i]]
        if len(next_asc) > 0:
            desc_to_asc.append(next_asc[0] - halfway_locations2[i])

    # Convert to arrays and match lengths
    asc_to_desc = np.array(asc_to_desc)
    desc_to_asc = np.array(desc_to_asc)
    min_len = min(len(asc_to_desc), len(desc_to_asc))

    if min_len == 0:
        return np.array([])  # No valid pairs

    peaktrough_symmetry = desc_to_asc[:min_len] / (asc_to_desc[:min_len] + desc_to_asc[:min_len])
    return peaktrough_symmetry

def _find_halfway_points(flw_cleaned, peaks, troughs):
    extrema = [(p, 'peak') for p in peaks] + [(t, 'trough') for t in troughs]
    points_sorted = sorted(extrema, key=lambda x: x[0])

    positions = [p[0] for p in points_sorted]
    types = [p[1] for p in points_sorted]
    for i in range(len(points_sorted)-1):
        point1 = positions[i]
        point2 = positions[i+1]
        halfway_value = (flw_cleaned[point1] + flw_cleaned[point2])/2
        segment = flw_cleaned[point1: point2]
        halfway_location = find_closest(halfway_value, segment, return_index=True) + point1

        if types[i] == 'peak' and types[i+1] == 'trough':
            pass
        elif types[i] == 'trough' and types[i+1] == 'peak':
            pass
        else:
            pass
