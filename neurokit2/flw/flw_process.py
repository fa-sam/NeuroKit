# -*- coding: utf-8 -*-

from ..misc import as_vector
from .flw_amplitude import flw_amplitude
from .flw_clean import flw_clean
from .flw_peaks import flw_peaks
from .flw_rvt import flw_rvt
from .flw_symmetry import flw_symmetry
from .flw_time import flw_time

def flw_process(
    flw_signal,
    sampling_rate=100,
    method="khodadad2018",
    method_rvt="cycle",
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



    See Also
    --------
    flw_clean, flw_findpeaks, flw_amplitude, flw_rvt, flw_symmetry

    Examples
    --------

    """
    # Sanitize input
    rsp_signal = as_vector(flw_signal)


    # Clean signal
    flw_cleaned = flw_clean(
        rsp_signal,
        sampling_rate=sampling_rate,
        method=method
    )

    # Extract, fix and format peaks
    peak_signal, peaks_info = flw_peaks(
        flw_cleaned,
        sampling_rate=sampling_rate,
        method=method,
        amplitude_min=0.3,
    )

    # Get additional parameters
    _, amp_info = flw_amplitude(flw_cleaned, peak_signal)

    symmetry_info = flw_symmetry(flw_cleaned, peak_signal)

    _, rvt_info = flw_rvt(
        flw_cleaned,
        method=method_rvt,
        sampling_rate=sampling_rate
    )
    time_info = flw_time(flw_cleaned, sampling_rate)

    process_info = {
        "FLW_Amplitude": amp_info["FLW_Amplitude"],
        "FLW_MVF": rvt_info["FLW_MVF"],
        "FLW_Tidal_Volume": rvt_info["FLW_Tidal_Volume"],
    }
    process_info.update(peaks_info)
    process_info.update(symmetry_info)
    process_info.update(time_info)

    return process_info

