# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from dataclasses import dataclass

import neurokit2 as nk

sampling_rate = 100

@dataclass
class TestCase:
    file_id: int
    pad_length: int
    peaks: list
    troughs: list
    insp_onsets: list
    exsp_onsets: list


cases = [
    TestCase(file_id=1, pad_length=0, peaks=[], troughs=[], insp_onsets=[], exsp_onsets=[]),
    TestCase(file_id=1, pad_length=5, peaks=[], troughs=[], insp_onsets=[], exsp_onsets=[]),
    TestCase(file_id=2, pad_length=0, peaks=[], troughs=[], insp_onsets=[], exsp_onsets=[]),
    TestCase(file_id=2, pad_length=5, peaks=[], troughs=[], insp_onsets=[], exsp_onsets=[]),
    TestCase(file_id=3, pad_length=0, peaks=[], troughs=[], insp_onsets=[], exsp_onsets=[]),
    TestCase(file_id=3, pad_length=5, peaks=[], troughs=[], insp_onsets=[], exsp_onsets=[]),
    # Add more cases as needed
]

def load_flow(id=1):
    flow = np.loadtxt(fr'Y:\Proj\Neurokit2\data\flow_sample{id}.txt')
    return flow


@pytest.mark.parametrize("file_id", [1,2,3])
@pytest.mark.parametrize("method", ["khodadad2018", "hampel", "biosppy"])
def test_flw_clean(file_id, method):
    flw = load_flow(file_id)

    flw_cleaned = nk.flw_clean(flw, sampling_rate=sampling_rate, method=method)

    assert len(flw) == len(flw_cleaned)



@pytest.mark.parametrize("file_id", [1,2,3])
@pytest.mark.parametrize("pad_length", [0,3,6])
def test_flw_peaks(file_id, pad_length):
    flw = load_flow(file_id)
    flw_cleaned = nk.flw_clean(flw, sampling_rate)

    signals, info = nk.flw_peaks(flw_cleaned, sampling_rate, pad_length=pad_length)
    flw_cleaned = signals['FLW_Clean']
    assert len(flw_cleaned) == len(flw)-2*pad_length*sampling_rate

