# -*- coding: utf-8 -*-

from ..rsp.rsp_peaks import rsp_peaks
from .flw_onsets import find_onsets, _flw_fix_onsets

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


