# -*- coding: utf-8 -*-

from ..rsp.rsp_peaks import rsp_peaks
from .flw_onsets import find_onsets, _flw_fix_onsets

def flw_peaks(flw_cleaned, sampling_rate=100, pad_length=0, method="khodadad2018", **kwargs):
    flw_info = find_onsets(flw_cleaned, sampling_rate=sampling_rate)
    peak_signal, rsp_info = rsp_peaks(flw_cleaned, sampling_rate, method, **kwargs)

    peaks = rsp_info['RSP_Peaks']
    troughs = rsp_info['RSP_Troughs']
    insp_onsets = flw_info["FLW_InspirationOnsets"]
    exsp_onsets = flw_info["FLW_ExpirationOnsets"]

    insp_onsets, exsp_onsets = _flw_fix_onsets(peaks, troughs, insp_onsets, exsp_onsets)

    peaks_info = {
        "FLW_InspirationOnsets": insp_onsets,
        "FLW_ExpirationOnsets": exsp_onsets,
        "FLW_Peaks": peaks,
        "FLW_Troughs": troughs,
    }
    if pad_length >0:
        low_ind = int(sampling_rate * pad_length)
        high_ind = len(flw_cleaned) - low_ind
        peak_signal = peak_signal.iloc[low_ind:high_ind,:].reset_index(drop=True)
        flw_cleaned, peaks_info = _fix_padded_params(flw_cleaned, peaks_info, sampling_rate, pad_length)

    peak_signal['FLW_Clean'] = flw_cleaned
    return peak_signal, peaks_info


def _fix_padded_params(flw_cleaned, peaks_info, sampling_rate, pad_length):

    low_ind = int(sampling_rate * pad_length)
    high_ind = len(flw_cleaned) - low_ind

    peaks = peaks_info['FLW_Peaks']
    troughs = peaks_info['FLW_Troughs']
    insp_onsets = peaks_info['FLW_InspirationOnsets']
    exsp_onsets = peaks_info['FLW_ExpirationOnsets']

    peaks = peaks[(low_ind < peaks) & (peaks < high_ind)] - low_ind
    troughs = troughs[(low_ind < troughs) & (troughs < high_ind)] - low_ind
    insp_onsets = insp_onsets[(low_ind < insp_onsets) & (insp_onsets < high_ind)] - low_ind
    exsp_onsets = exsp_onsets[(low_ind < exsp_onsets) & (exsp_onsets < high_ind)] - low_ind

    peaks_info["FLW_Peaks"] = peaks
    peaks_info["FLW_Troughs"] = troughs
    peaks_info["FLW_InspirationOnsets"] = insp_onsets
    peaks_info["FLW_ExpirationOnsets"] = exsp_onsets

    flw_cleaned = flw_cleaned[low_ind:high_ind]
    return flw_cleaned, peaks_info
