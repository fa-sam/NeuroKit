# -*- coding: utf-8 -*-

from ..rsp.rsp_peaks import rsp_peaks
from .flw_onsets import find_onsets

def flw_peaks(flw_cleaned, sampling_rate=100, method="khodadad2018", **kwargs):
    flw_info = find_onsets(flw_cleaned, sampling_rate=sampling_rate)
    peak_signal, rsp_info = rsp_peaks(flw_cleaned, sampling_rate, method, **kwargs)

    flw_info['FLW_Peaks'] = rsp_info['RSP_Peaks']
    flw_info['FLW_Troughs'] = rsp_info['RSP_Troughs']

    return peak_signal, flw_info
