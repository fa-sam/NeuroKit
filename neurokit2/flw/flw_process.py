# -*- coding: utf-8 -*-
import pandas as pd
from ..misc import as_vector
from .flw_amplitude import flw_amplitude
from .flw_clean import flw_clean
from .flw_peaks import flw_peaks
from .flw_rvt import flw_rvt
from .flw_symmetry import flw_symmetry
from .flw_time import flw_time
from ..signal import signal_rate

def flw_process(
    flw_signal,
    sampling_rate=100,
    method="khodadad2018",
    method_rvt="cycle",
    pad_length=0,
    **kwargs
):
    """**Process a respiration airflow signal**

    Convenience function that automatically processes an airflow signal with one of the
    following methods:

    * `Khodadad et al. (2018) <https://iopscience.iop.org/article/10.1088/1361-6579/aad7e6/meta>`_

    * `BioSPPy <https://github.com/PIA-Group/BioSPPy/blob/master/biosppy/signals/resp.py>`_

    Parameters
    ----------
    flw_signal : Union[list, np.array, pd.Series]
        The raw respiration airflow channel.
    sampling_rate : int
        The sampling frequency of airflow signal (in Hz, i.e., samples/second).
    method : str
        The processing pipeline to apply. Can be one of ``"khodadad2018"`` (default)
        or ``"biosppy"``.
    method_rvt : str
        The rvt method to apply. Can be one of ``"cycle"`` (default), ``"continuous"``,
        or ``"cycle"``.
    pad_length : float
        The length of padding in seconds from both sides of the signal. If the signal is padded from both sides, we use the whole
        signal for filtering (flw_clean) and peak detection, but only consider the part without padding for the
        feature analysis.
    **kwargs
        Other arguments to be passed to specific methods. For more information,
        see :func:`.rsp_methods`.

    Returns
    -------
    signals : DataFrame
        A DataFrame of same length as :func:`.rsp_signal` containing the following columns:

        .. codebookadd::
            FLW_Raw|The raw flow signal.
            FLW_Clean|The filtered and cleaned flow signal.
            FLW_Peaks|The respiratory flow peaks  marked as "1" in a list of zeros.
            FLW_Troughs|The respiratory flow troughs  marked as "1" in a list of zeros.
            FLW_Rate|The breathing rate interpolated between inhalation peaks.
            RSP_Amplitude|The breathing amplitude interpolated between inhalation peaks.
            RSP_Phase|The breathing phase, marked by "1" for inspiration and "0" for expiration.
            RSP_Phase_Completion|The breathing phase completion, expressed in percentage \
                (from 0 to 1), representing the stage of the current respiratory phase.
            RSP_RVT|Respiratory volume per time (RVT).

    info : dict
        A dictionary containing:
            "FLW_Peaks" | the samples at which inhalation peaks occur,
            "FLW_Troughs" | the samples at which exhalation troughs occur,
            "FLW_InspirationOnsets" | the samples at which inspiration onsets occur,
            "FLW_ExpirationOnsets" | the samples at which expiration onsets occur,
            "FLW_Amplitude"|The breathing amplitude in each breath,
            "FLW_RVT"|Respiratory volume per time (RVT) for each breath.

    .. note::

      The pad_length parameter is used when we already padded the input flw_signal from both side, either with
      constant value or be real flow values from previous and next epoch. The padding helps the peak detection
      algorithm to detect more peaks and troughs in the real interval. Otherwise, one or more peaks/troughs
      from beginning or end of signal might be missed by the find_peaks function.

    See Also
    --------
    flw_clean, flw_findpeaks, flw_amplitude, flw_rvt, flw_symmetry

    Examples
    --------

    """
    # Sanitize input
    flw_signal = as_vector(flw_signal)


    # Clean signal
    flw_cleaned = flw_clean(
        flw_signal,
        sampling_rate=sampling_rate,
        method=method
    )

    # Extract, fix and format peaks
    peaks_signal, peaks_info = flw_peaks(
        flw_cleaned,
        sampling_rate=sampling_rate,
        pad_length=pad_length,
        method=method,
        amplitude_min=0.3,
    )
    flw_cleaned = peaks_signal['FLW_Clean'] # the original flw_cleaned might be with padding, we get the updated one here.

    if pad_length >0:
        low_ind = int(sampling_rate * pad_length)
        high_ind = len(flw_signal) - low_ind
        flw_signal = flw_signal[low_ind:high_ind]
        # flw_cleaned, flw_signal, peaks_info = _fix_padded_params(flw_cleaned, flw_signal, peaks_info, sampling_rate, pad_length)

    peaks = peaks_info["FLW_Peaks"]
    troughs = peaks_info["FLW_Troughs"]
    insp_onsets = peaks_info["FLW_InspirationOnsets"]

    # Get additional parameters
    # _, amp_info = flw_amplitude(flw_cleaned, {'FLW_Peaks':peaks, 'FLW_Troughs':troughs})
    _, amp_info = flw_amplitude(flw_cleaned, peaks={'FLW_Peaks':peaks, 'FLW_Troughs':troughs}, inspiration_onsets=insp_onsets, method='max-min')

    symmetry_info = flw_symmetry(flw_cleaned, {'FLW_Peaks':peaks, 'FLW_Troughs':troughs})

    _, rvt_info = flw_rvt(
        flw_cleaned,
        peaks_info,
        method=method_rvt,
        sampling_rate=sampling_rate
    )
    time_info = flw_time(flw_cleaned, peaks_info, sampling_rate)

    process_info = {
        "FLW_MVF": rvt_info["FLW_MVF"],
        "FLW_Tidal_Volume": rvt_info["FLW_Tidal_Volume"],
    }
    process_info.update(peaks_info)
    process_info.update(amp_info)
    process_info.update(symmetry_info)
    process_info.update(time_info)

    rate = signal_rate(
        peaks_info["FLW_Troughs"], sampling_rate=sampling_rate, desired_length=len(flw_signal)
    )
    signals = pd.DataFrame(
        {
            "FLW_Raw": flw_signal,
            "FLW_Clean": flw_cleaned,
            "FLW_Rate": rate,
        }
    )
    return signals, process_info

# def _fix_padded_params(flw_cleaned, flw_signal, peaks_info, sampling_rate, pad_length):
#
#     low_ind = int(sampling_rate * pad_length)
#     high_ind = len(flw_cleaned) - low_ind
#
#     peaks = peaks_info['FLW_Peaks']
#     troughs = peaks_info['FLW_Troughs']
#     insp_onsets = peaks_info['FLW_InspirationOnsets']
#     exsp_onsets = peaks_info['FLW_ExpirationOnsets']
#
#     peaks = peaks[(low_ind < peaks) & (peaks < high_ind)] - low_ind
#     troughs = troughs[(low_ind < troughs) & (troughs < high_ind)] - low_ind
#     insp_onsets = insp_onsets[(low_ind < insp_onsets) & (insp_onsets < high_ind)] - low_ind
#     exsp_onsets = exsp_onsets[(low_ind < exsp_onsets) & (exsp_onsets < high_ind)] - low_ind
#
#     peaks_info["FLW_Peaks"] = peaks
#     peaks_info["FLW_Troughs"] = troughs
#     peaks_info["FLW_InspirationOnsets"] = insp_onsets
#     peaks_info["FLW_ExpirationOnsets"] = exsp_onsets
#
#     flw_cleaned = flw_cleaned[low_ind:high_ind]
#     flw_signal = flw_signal[low_ind:high_ind]
#     return flw_cleaned, flw_signal, peaks_info
