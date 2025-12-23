# -*- coding: utf-8 -*-
import numpy as np

from .flw_peaks import flw_peaks


def flw_time(flw_cleaned, sampling_rate=100, method="khodadad2018"):
    """
        It calculates the inspiration time, expiration time and the respiratory intervals using the airflow signal.
        It uses the ``.flw_peaks`` function to detect the troughs, inspiration onsets and expiration onsets.

        Parameters
        ----------
        flw_cleaned : np.ndarray
            Airflow signal (L/s or mL/s). Ideally, inspiration is positive and expiration negative.
        sampling_rate : float
            Sampling rate in Hz.

        Returns
        -------
        dict
            {
              "FLW_Inspiration_Times":np.ndarray,
              "FLW_Expiration_Times": np.ndarray,
              "FLW_RRI": np.ndarray
            }
        """
    # Extract, fix and format peaks
    peak_signal, peaks_info = flw_peaks(
        flw_cleaned,
        sampling_rate=sampling_rate,
        method=method
    )

    insp_onsets = peaks_info["FLW_InspirationOnsets"]
    exsp_onsets = peaks_info["FLW_ExpirationOnsets"]

    troughs = peaks_info["FLW_Troughs"]

    insp_times = []
    exsp_times = []

    if insp_onsets[0] > exsp_onsets[0]:
        exsp_onsets = exsp_onsets[1:]
    for i, e in zip(insp_onsets, exsp_onsets):
        insp_time = (e-i)/sampling_rate
        insp_times.append(insp_time)

    exsp_onsets = peaks_info["FLW_ExpirationOnsets"]  # in case it has been changed
    if insp_onsets[0] < exsp_onsets[0]:
        insp_onsets = insp_onsets[1:]
    for e, i in zip(exsp_onsets, insp_onsets):
        exsp_time = (i-e)/sampling_rate
        exsp_times.append(exsp_time)

    flw_rri = np.diff(troughs)/sampling_rate

    insp_times = np.array(insp_times)
    exsp_times = np.array(exsp_times)
    flw_rri = np.array(flw_rri)

    info = {
        "FLW_Inspiration_Times": insp_times,
        "FLW_Expiration_Times": exsp_times,
        "FLW_RRI": flw_rri
    }

    return info
