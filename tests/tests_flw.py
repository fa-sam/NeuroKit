# -*- coding: utf-8 -*-
import json

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

flow_sample_files = ["003_943477", "003_1037669", "003_1327737", "003_1352827", "003_1390320", "003_1439755", "003_1698227", "003_1833370", "003_2363468", "003_2551420", "003_2975805"]
cases = [
    TestCase(file_id=1, pad_length=0, peaks=[], troughs=[], insp_onsets=[], exsp_onsets=[]),
    TestCase(file_id=1, pad_length=5, peaks=[], troughs=[], insp_onsets=[], exsp_onsets=[]),
    TestCase(file_id=2, pad_length=0, peaks=[], troughs=[], insp_onsets=[], exsp_onsets=[]),
    TestCase(file_id=2, pad_length=5, peaks=[], troughs=[], insp_onsets=[], exsp_onsets=[]),
    TestCase(file_id=3, pad_length=0, peaks=[], troughs=[], insp_onsets=[], exsp_onsets=[]),
    TestCase(file_id=3, pad_length=5, peaks=[], troughs=[], insp_onsets=[], exsp_onsets=[]),
    # Add more cases as needed
]

def load_test_results():
    file_path = r'tests/flw_test_results.json'
    with open(file_path, 'r') as f:
        test_results = json.load(f)
        return test_results

def load_flow(id=None):
    flow = np.loadtxt(fr'data/flow_sample_{id}.txt')
    return flow


@pytest.mark.parametrize("file_id", flow_sample_files)
@pytest.mark.parametrize("method", ["khodadad2018", "hampel", "biosppy"])
def test_flw_clean(file_id, method):
    flw = load_flow(file_id)

    flw_cleaned = nk.flw_clean(flw, sampling_rate=sampling_rate, method=method)

    assert len(flw) == len(flw_cleaned)



@pytest.mark.parametrize("file_id", flow_sample_files[0:1])
@pytest.mark.parametrize("pad_length", [0,5])
def test_flw_peaks(file_id, pad_length):
    flw = load_flow(id=file_id)
    test_results_json = load_test_results()
    test_info = test_results_json[f"flow_sample_{file_id}"][f"pad_len={pad_length}"]
    flw_cleaned = nk.flw_clean(flw, sampling_rate)

    signals, info = nk.flw_peaks(flw_cleaned, sampling_rate, pad_length=pad_length)
    flw_cleaned = signals['FLW_Clean']
    assert len(flw_cleaned) == len(flw)-2*pad_length*sampling_rate

    assert test_info['FLW_Peaks'] == info["FLW_Peaks"].tolist()
    assert test_info['FLW_Troughs'] == info["FLW_Troughs"].tolist()
    assert test_info['FLW_InspirationOnsets'] == info["FLW_InspirationOnsets"].tolist()
    assert test_info['FLW_ExpirationOnsets'] == info["FLW_ExpirationOnsets"].tolist()

@pytest.mark.parametrize("file_id", ["003_943477"])
@pytest.mark.parametrize("pad_length", [0,5])
def test_flw_time(file_id, pad_length):
    flw = load_flow(file_id)
    test_results_json = load_test_results()
    test_info = test_results_json[f'flow_sample_{file_id}'][f"pad_len={pad_length}"]
    flw_cleaned = nk.flw_clean(flw, sampling_rate)

    signals, peaks_info = nk.flw_peaks(flw_cleaned, sampling_rate, pad_length=pad_length)
    flw_cleaned = signals['FLW_Clean']
    time_info = nk.flw_time(flw_cleaned, peaks_info=peaks_info, sampling_rate=sampling_rate)

    assert len(time_info['FLW_Inspiration_Times']) == len(test_info['FLW_Inspiration_Times'])
    assert len(time_info['FLW_Expiration_Times']) == len(test_info['FLW_Expiration_Times'])
    assert len(time_info['FLW_RRI']) == len(test_info['FLW_RRI'])

    assert time_info['FLW_Inspiration_Times'].tolist() == test_info['FLW_Inspiration_Times']
    assert time_info['FLW_Expiration_Times'].tolist() == test_info['FLW_Expiration_Times']
    assert time_info['FLW_RRI'].tolist() == test_info['FLW_RRI']


@pytest.mark.parametrize("file_id", ["003_943477"])
@pytest.mark.parametrize("pad_length", [0,5])
def test_flw_amplitude(file_id, pad_length):
    flw = load_flow(file_id)
    test_results_json = load_test_results()
    test_info = test_results_json[f'flow_sample_{file_id}'][f"pad_len={pad_length}"]
    flw_cleaned = nk.flw_clean(flw, sampling_rate)

    signals, peaks_info = nk.flw_peaks(flw_cleaned, sampling_rate, pad_length=pad_length)
    flw_cleaned = signals['FLW_Clean']
    peaks = peaks_info["FLW_Peaks"]
    troughs = peaks_info["FLW_Troughs"]
    insp_onsets = peaks_info["FLW_InspirationOnsets"]

    _, amp_info = nk.flw_amplitude(flw_cleaned, peaks={'FLW_Peaks':peaks, 'FLW_Troughs':troughs}, inspiration_onsets=insp_onsets, method='max-min')

    assert len(test_info['FLW_Amplitude']) == len(amp_info['FLW_Amplitude'])
    assert list(map(lambda x: round(x, 2), amp_info['FLW_Amplitude'].tolist())) == list(map(lambda x: round(x,2), test_info['FLW_Amplitude']))

@pytest.mark.parametrize("file_id", ["003_943477"])
@pytest.mark.parametrize("pad_length", [0,5])
def test_flw_symmetry(file_id, pad_length):
    flw = load_flow(file_id)
    test_results_json = load_test_results()
    test_info = test_results_json[f'flow_sample_{file_id}'][f"pad_len={pad_length}"]
    flw_cleaned = nk.flw_clean(flw, sampling_rate)

    signals, peaks_info = nk.flw_peaks(flw_cleaned, sampling_rate, pad_length=pad_length)
    flw_cleaned = signals['FLW_Clean']
    peaks = peaks_info["FLW_Peaks"]
    troughs = peaks_info["FLW_Troughs"]

    symm_info = nk.flw_symmetry(flw_cleaned, peaks={'FLW_Peaks':peaks, 'FLW_Troughs':troughs})

    for key, array_values in symm_info.items():
        assert len(test_info[key]) == len(array_values)
        assert list(map(lambda x: round(x, 2), array_values.tolist())) == list(map(lambda x: round(x,2), test_info[key]))
