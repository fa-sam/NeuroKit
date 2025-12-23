# -*- coding: utf-8 -*-
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..misc import NeuroKitWarning, find_closest
from ..signal import signal_interpolate
from ..stats import rescale
from ..rsp.rsp_fixpeaks import _rsp_fixpeaks_retrieve
from .flw_onsets import find_onsets

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

    # Sanity checks -----------------------------------------------------------
    failed_checks = False
    if len(peaks) <= 4 or len(troughs) <= 4:
        warn(
            "Not enough peaks and troughs (signal too short?) to compute symmetry"
            + ", returning nan for symmetry.",
            category=NeuroKitWarning,
        )
        failed_checks = True

    if np.any(peaks - troughs < 0):
        warn(
            "Peaks and troughs are not correctly aligned (i.e., not consecutive)"
            + ", returning nan for symmetry.",
            category=NeuroKitWarning,
        )
        failed_checks = True

    if failed_checks:
        return {
                "FLW_Symmetry_PeakTrough": np.array([]),
                "FLW_Symmetry_RiseDecay":  np.array([]),
                "FLW_Expiration_Symmetry": np.array([]),
                "FLW_Inspiration_Symmetry": np.array([]),
            }

    # Compute symmetry features -----------------------------------------------
    # See https://twitter.com/bradleyvoytek/status/1591495571269124096/photo/1

    # Inspiration, Expiration symmetry
    onsets = find_onsets(flw_cleaned, sampling_rate)
    insp_onsets = onsets["FLW_InspirationOnsets"]
    exsp_onsets = onsets["FLW_ExpirationOnsets"]

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

    exsp_onsets = onsets["FLW_ExpirationOnsets"]  # in case it has been changed
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
    through_to_peak = peaks - troughs
    peak_to_through = troughs[1:] - peaks[:-1]
    risedecay_symmetry = through_to_peak[:-1] / (through_to_peak[:-1] + peak_to_through)

    # Find half-way points (trough to peak)
    halfway_values = (flw_cleaned[peaks] - flw_cleaned[troughs]) / 2
    halfway_values += flw_cleaned[troughs]
    halfway_locations = np.zeros(len(halfway_values))
    for i in range(len(peaks)):
        segment = flw_cleaned[troughs[i] : peaks[i]]
        halfway_locations[i] = (
            find_closest(halfway_values[i], segment, return_index=True) + troughs[i]
        )

    # Find half-way points (peak to next through)
    halfway_values2 = (flw_cleaned[peaks[:-1]] - flw_cleaned[troughs[1::]]) / 2
    halfway_values2 += flw_cleaned[troughs[1::]]
    halfway_locations2 = np.zeros(len(halfway_values2))
    for i in range(len(peaks[:-1])):
        segment = flw_cleaned[peaks[i] : troughs[i + 1]]
        halfway_locations2[i] = (
            find_closest(halfway_values2[i], segment, return_index=True) + peaks[i]
        )

    # Peak-trough symmetry
    asc_to_desc = halfway_locations2[1:] - halfway_locations[1:-1]
    desc_to_asc = halfway_locations[1:-1] - halfway_locations2[:-1]
    peaktrough_symmetry = desc_to_asc / (asc_to_desc + desc_to_asc)


    if show is True:
        normalized = rescale(flw_cleaned)  # Rescale to 0-1
        plt.plot(normalized, color="grey", label="Respiration (normalized)")
        plt.scatter(peaks, normalized[peaks], color="red")
        plt.scatter(troughs, normalized[troughs], color="blue")
        plt.scatter(halfway_locations, normalized[halfway_locations.astype(int)], color="orange")
        plt.scatter(
            halfway_locations2, normalized[halfway_locations2.astype(int)], color="darkgreen"
        )

        plt.plot(peaks[1:], risedecay_symmetry , color="purple", label="Rise-decay symmetry")
        plt.plot(peaks[1:], peaktrough_symmetry, color="green", label="Peak-trough symmetry")
        plt.legend()

    info = {
        "FLW_Symmetry_PeakTrough": peaktrough_symmetry,
        "FLW_Symmetry_RiseDecay": risedecay_symmetry,
        "FLW_Expiration_Symmetry": np.asarray(exsp_symmetry),
        "FLW_Inspiration_Symmetry": np.asarray(insp_symmetry),
    }
    return info

