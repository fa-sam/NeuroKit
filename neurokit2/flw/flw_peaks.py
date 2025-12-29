# -*- coding: utf-8 -*-
import numpy as np

from ..rsp.rsp_peaks import rsp_peaks
from .flw_onsets import find_onsets

def flw_peaks(flw_cleaned, sampling_rate=100, method="khodadad2018", **kwargs):
    flw_info = find_onsets(flw_cleaned, sampling_rate=sampling_rate)
    peak_signal, rsp_info = rsp_peaks(flw_cleaned, sampling_rate, method, **kwargs)

    peaks = rsp_info['RSP_Peaks']
    troughs = rsp_info['RSP_Troughs']
    insp_onsets = flw_info["FLW_InspirationOnsets"]
    exsp_onsets = flw_info["FLW_ExpirationOnsets"]
    insp_onsets, exsp_onsets = _flw_fix_onsets(peaks, troughs, insp_onsets, exsp_onsets)

    flw_info = {
        "FLW_InspirationOnsets": insp_onsets,
        "FLW_ExpirationOnsets": exsp_onsets,
        "FLW_Peaks": peaks,
        "FLW_Troughs": troughs,
    }

    return peak_signal, flw_info


def _flw_fix_onsets(peaks, troughs, insp_onsets, exsp_onsets):
    events = np.concatenate([peaks, troughs])
    labels = np.array(['peak'] * len(peaks) + ['trough'] * len(troughs))
    order = np.argsort(events)
    events = events[order]
    labels = labels[order]

    # Iterate through consecutive pairs
    for i in range(len(events) - 1):
        start = events[i]
        end = events[i + 1]

        if labels[i] == 'trough' and labels[i + 1] == 'peak':
            # Trough → Peak: keep one inspiration onset
            del_candidates = exsp_onsets[(exsp_onsets > start) & (exsp_onsets < end)]
            insp_candidate = insp_onsets[(insp_onsets > start) & (insp_onsets < end)]
            del_candidates = np.concatenate((del_candidates, insp_candidate[1:]))
            for j in del_candidates:
                index = np.where(exsp_onsets == j)[0]
                exsp_onsets = np.delete(exsp_onsets, index)

        elif labels[i] == 'peak' and labels[i + 1] == 'trough':
            # Peak → Trough: keep one expiration onset
            del_candidates = insp_onsets[(insp_onsets > start) & (insp_onsets < end)]  # remove all inspiration onsets
            exsp_candidate = exsp_onsets[(exsp_onsets > start) & (exsp_onsets < end)]  # keep only one expiration onset
            del_candidates = np.concatenate((del_candidates, exsp_candidate[1:]))
            for j in del_candidates:
                index = np.where(insp_onsets == j)[0]
                insp_onsets = np.delete(insp_onsets, index)
                print('removing ', j)

    return insp_onsets, exsp_onsets
