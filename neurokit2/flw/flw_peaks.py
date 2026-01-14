# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..rsp.rsp_peaks import rsp_peaks
from .flw_onsets import find_onsets, _flw_fix_onsets
from .flw_findpeaks import flw_findpeaks
from ..signal import signal_formatpeaks

def flw_peaks(flw_cleaned, sampling_rate=100, pad_length=0, method="fsa", **kwargs):
    """**Identify crucial points in an airflow respiration (FLW) signal**

        This function runs :func:`.find_onsets` and :func:`.rsp_peaks` to identify and process
        peaks, troughs, inhalation onsets and exhalation onsets in a preprocessed respiration airflow signal
        using different sets of parameters, such as:

        * **khodad2018**: Uses the parameters in Khodadad et al. (2018).
        * **biosppy**: Uses the parameters in `BioSPPy's <https://github.com/PIA-Group/BioSPPy>`_
          ``resp()`` function.
        * **scipy** Uses the `scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_
          peak-detection function.

        Parameters
        ----------
        flw_cleaned : Union[list, np.array, pd.Series]
            The cleaned respiration airflow channel as returned by :func:`.flw_clean`.
        sampling_rate : int
            The sampling frequency of airflow signal (in Hz, i.e., samples/second).
        pad_length : int, optional
            The length of the padded signal (in seconds). All crucial points that are located in the padded part,
            will be omitted.
        method : str
            The processing pipeline to apply. Can be one of ``"fsa"`` (default), ``"khodadad2018"``, ``"biosppy"``
            or ``"scipy"``.
        **kwargs
            Other arguments to be passed to the different peak finding methods. See
            :func:`.rsp_findpeaks`.

        Returns
        -------
        info : dict
            A dictionary containing additional information, in this case the samples at which peaks
            ,troughs, inhalation onsets, and exhalation onsets occur, accessible with the keys
            ``"FLW_Peaks"``, ``"FLW_Troughs"``, ``"FLW_InspirationOnsets"``, and ``"FLW_ExpirationOnsets"`` respectively, as well as the signals' sampling rate.
        peak_signal : DataFrame
            A DataFrame of same length as the input signal (without padding) in which occurrences of peaks, troughs,
            inhalation onsets, and exhalation onsets are marked as "1" in lists of zeros with the same
            length as :func:`.flw_cleaned`. Accessible with the keys ``"FLW_Peaks"``, ``"FLW_Troughs"``,
            ``"FLW_InspirationOnsets"``, and ``"FLW_ExpirationOnsets"`` respectively.
            In addition, the cleaned flow signal without padding is accessible with the keys ``"FLW_Clean"``.


        See Also
        --------
        flw_clean, signal_rate, flw_amplitude, flw_process

        Examples
        --------
        .. ipython:: python

          import neurokit2 as nk
          import pandas as pd

          flw = nk.rsp_simulate(duration=30, respiratory_rate=15, sampling_rate=100)
          cleaned = nk.flw_clean(flw, sampling_rate=100)
          peak_signal, info = nk.flw_peaks(cleaned, sampling_rate=100)
          peak_signal.drop(columns=['FLW_Clean'], inplace=True)

          data = pd.concat([pd.DataFrame({"FLW": flw}), peak_signal], axis=1)
          @savefig p_flw_peaks1.png scale=100%
          fig = nk.signal_plot(data)
          @suppress
          plt.close()

        References
        ----------
        * Khodadad, D., Nordebo, S., Müller, B., Waldmann, A., Yerworth, R., Becher, T., ... & Bayford,
          R. (2018). Optimized breath detection algorithm in electrical impedance tomography.
          Physiological measurement, 39(9), 094001.

        """

    # _, rsp_info = rsp_peaks(flw_cleaned, sampling_rate, method, **kwargs)
    info = flw_findpeaks(flw_cleaned, sampling_rate=sampling_rate, method=method, **kwargs)

    peaks = info['FLW_Peaks']
    troughs = info['FLW_Troughs']


    flw_info = find_onsets(flw_cleaned, sampling_rate=sampling_rate)
    insp_onsets = flw_info["FLW_InspirationOnsets"]
    exsp_onsets = flw_info["FLW_ExpirationOnsets"]

    peaks, troughs = _fix_peaks(flw_cleaned, peaks, troughs)
    insp_onsets, exsp_onsets = _flw_fix_onsets(peaks, troughs, insp_onsets, exsp_onsets)

    peaks_info = {
        "FLW_InspirationOnsets": insp_onsets,
        "FLW_ExpirationOnsets": exsp_onsets,
        "FLW_Peaks": peaks,
        "FLW_Troughs": troughs,
    }
    if pad_length >0:
        flw_cleaned, peaks_info = _fix_padded_params(flw_cleaned, peaks_info, sampling_rate, pad_length)

    onset_signal = signal_formatpeaks(peaks_info, desired_length=len(flw_cleaned))
    onset_signal['FLW_Clean'] = flw_cleaned
    return onset_signal, peaks_info

def _fix_peaks(flw_cleaned, peaks, troughs):
    """
    It removes the peak if the peak is not the local maxima, or removes the trough if the trough is not a local minima.
    :param flw_cleaned:
    :param peaks:
    :param troughs:
    :return:
    """
    new_peaks = []
    new_troughs = []
    for p in peaks:
        low_ind = max(0, p-10)
        high_ind = min(len(flw_cleaned), p+10)
        if max(flw_cleaned[low_ind:high_ind]) <= flw_cleaned[p]:
            new_peaks.append(int(p))

    for t in troughs:
        low_ind = max(0, t - 10)
        high_ind = min(len(flw_cleaned), t + 10)
        if min(flw_cleaned[low_ind:high_ind]) >= flw_cleaned[t]:
            new_troughs.append(int(t))

    return np.array(new_peaks, dtype=object), np.array(new_troughs, dtype=object)

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
