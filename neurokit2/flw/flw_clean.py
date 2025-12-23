# -*- coding: utf-8 -*-

from ..rsp.rsp_clean import rsp_clean


def flw_clean(flw_signal, sampling_rate=100, method="khodadad2018", **kwargs):
    return rsp_clean(flw_signal, sampling_rate=sampling_rate, method=method, **kwargs)
