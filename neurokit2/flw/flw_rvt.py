# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from ..signal import signal_interpolate
from ..stats import rescale
from .flw_peaks import flw_peaks


def flw_rvt(
    flw_cleaned,
    peaks_info=None,
    sampling_rate=100,
    method="cycle",
    use_mean_insp_exp=False,
    show=False,
    **kwargs
):
    """**Respiratory Volume per Time (RVT) from Airflow**

    Computes Respiratory Volume per Time (RVT) from the airflow. RVT is the product of respiratory volume and
    breathing rate. RVT can be used to identify the global fMRI confounds of breathing, which is
    often considered noise.

    Parameters
    ----------
    flw_cleaned : array
        Array containing the flow rate, produced by :func:`.signal_rate`.
    sampling_rate : int, optional
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    method: str, optional
        The rvt method to apply. Can be one of  ``"cycle"`` (default), ``"continuous"`` or
        ``"harrison2021"``.
    use_mean_insp_exp: bool, optional
        If True, compute VT as mean(inspiration_integral, expiration_integral_abs).
    show : bool, optional
        If ``True``, will return a simple plot of the RVT (with the re-scaled original FLW signal).
    **kwargs
        Arguments to be passed to the underlying peak detection algorithm.

    Returns
    -------
    array

        rvt : array, RVT in L/s, length == len(flow). Array containing the current RVT at every breath.
        info : dict with keys
            'FLW_Tidal_Volume'|nparray, tidal volume for each breath
            'cycle_s'|nparray, inspiration time for each breath
            'FLW_MVF'|nparray, mean ventilatory flow for each breath.

    See Also
    --------
    signal_rate, flw_peaks, flw_clean

    Examples
    --------



    References
    ----------
    * Harrison, S. J., Bianchi, S., Heinzle, J., Stephan, K. E., Iglesias, S., & Kasper, L. (2021).
      A Hilbert-based method for processing respiratory timeseries. Neuroimage, 230, 117787.
    """
    method = method.lower()  # remove capitalised letters
    if method in ["cycle"]:
        rvt, info = _flw_rvt_cycle(flw_cleaned, peaks_info, sampling_rate=sampling_rate, use_mean_insp_exp=use_mean_insp_exp, **kwargs)
    elif method in ["continuous", "ventilation"]:
        rvt, info = _flw_rvt_continuous(flw_cleaned, sampling_rate=sampling_rate ** kwargs)
    elif method in ["harrison", "harrison2021"]:
        rvt, info = _flw_rvt_harrison(
            flw_cleaned,
            sampling_rate=sampling_rate
        )
    else:
        raise ValueError("NeuroKit error: flw_rvt(): 'method' should be one of 'cycle', 'continuous' or 'harrison'.")
    if show:
        _flw_rvt_plot(rvt, flw_cleaned, sampling_rate)
    return rvt, info

def _flw_rvt_cycle(
    flw_cleaned,
    peaks_info=None,
    sampling_rate=100,
    use_mean_insp_exp=False,
    peak_distance=0.8,
    peak_prominence=0.5,
    interpolation_method="monotone_cubic",
):
    """
        Compute RVT (L/s) from airflow by integrating per inspiration to get tidal volume.

        Steps:
        - Detect inspiration onsets and expiration onsets.
        - Integrate airflow over the inspiration phase to get VT_insp (L).
        - RVT per breath = VT_insp / (time between consecutive inspiration starts).
        - Interpolate per-breath RVT to a continuous per-sample series.

        Parameters
        ----------
        flw_cleaned : array, cleaned airflow in L/s (positive = inspiration).
        sampling_rate   : sampling rate (Hz).
        use_mean_insp_exp : if True, compute VT as mean(inspiration_integral, expiration_integral_abs).
                            Otherwise, use inspiration integral.
        **kwargs
        Arguments to be passed to the underlying peak detection algorithm.

        Returns
        -------
        rvt : array, RVT in L/s, length == len(flow).
        info : dict with keys
            'FLW_Tidal_Volume'|nparray, tidal volume for each breath
            'cycle_s'|nparray, inspiration time for each breath
            'FLW_MVF'|nparray, mean ventilatory flow for each breath.
        """

    if peaks_info is None or 'FLW_InspirationOnsets' not in peaks_info or 'FLW_ExpirationOnsets' not in peaks_info:
        # find peaks and troughs
        _, peaks_info = flw_peaks(
            flw_cleaned,
            sampling_rate=sampling_rate,
            method="scipy",
            peak_distance=peak_distance,
            peak_prominence=peak_prominence,
        )
    insp_onsets = peaks_info["FLW_InspirationOnsets"]
    exsp_onsets = peaks_info["FLW_ExpirationOnsets"]

    empty_info = {
            "FLW_Tidal_Volume": np.asarray([]),
            "FLW_cycle_s": np.asarray([]),
            "FLW_MVF": np.asarray([]),
        }
    if len(insp_onsets) < 2 or len(exsp_onsets) < 1:

        return np.full_like(flw_cleaned, np.nan), empty_info

    if insp_onsets[0] > exsp_onsets[0]:
        exsp_onsets = exsp_onsets[1:]

    # 3) Per-breath tidal volume by integrating positive flow during inspiration
    VT = []
    t_breath = []  # time reference per breath (midpoint between consecutive inspiration starts)
    rvt_breath = []
    cycle_s = []

    # go over loop
    for ind in range(len(insp_onsets)-1):
        insp_index = insp_onsets[ind]
        exsp_index = exsp_onsets[ind]
        next_insp = insp_onsets[ind+1]

        # get inspiration segment and then integral (positive flow only)
        insp_seg = np.clip(flw_cleaned[insp_index:exsp_index], 0.0, None)
        vt_insp = np.trapezoid(insp_seg, dx=1.0 / sampling_rate)

        if use_mean_insp_exp:
            # Expiration integral (absolute value of negative flow) until next inspiration
            exp_seg = np.clip(-flw_cleaned[exsp_index:next_insp], 0.0, None)
            vt_exp = np.trapezoid(exp_seg, dx=1.0 / sampling_rate)
            vt = 0.5 * (vt_insp + vt_exp)
        else:
            vt = vt_insp

        # Cycle duration: time between inspiration starts
        cyc = (next_insp - insp_index) / sampling_rate
        if cyc <= 0:
            continue

        VT.append(vt)
        cycle_s.append(cyc)
        rvt_breath.append(vt / cyc)  # L/s
        t_breath.append((insp_index + next_insp) * 0.5)  # index midpoint

    VT = np.asarray(VT)
    cycle_s = np.asarray(cycle_s)
    rvt_breath = np.asarray(rvt_breath)
    t_breath = np.asarray(t_breath, dtype=float)

    if len(rvt_breath) == 0:
        return np.full_like(flw_cleaned, np.nan), empty_info

    # 4) Interpolate to full time base
    rvt = signal_interpolate(t_breath, rvt_breath, range(len(flw_cleaned)), method=interpolation_method)
    info = {
            "FLW_Tidal_Volume": VT,
            "FLW_cycle_s": cycle_s,
            "FLW_MVF": rvt_breath,
        }
    return rvt, info



def _flw_rvt_continuous(
    flw_cleaned,
    sampling_rate=100,
    unit="L/min"
):

    """
    Breath-aware minute ventilation:
    - Detect inspirations
    - Integrate positive flow per inspiration to compute VT for each breath
    - For each time t, sum VTs of breaths whose inspiration start lies in [t-window_s, t]
      and divide by window length to get VE (L/s), then convert to L/min.

    Parameters
    ----------
    flow : array (L/s)
    fs   : Hz
    window_s : seconds (default 60)
    lp_hz : low-pass cutoff for airflow preprocessing
    thresh : positive threshold for inspiration detection (L/s)
    min_breath_s : minimum allowed duration between starts (s)
    unit : "L/min" (default) or "L/s"

    Returns
    -------
    VE : array, continuous ventilation (L/min or L/s)
    """
    raise NotImplementedError("_flw_rvt_continuous is not implemented yet. Try _flw_rvt_cycle")





def _flw_rvt_harrison(
    flw_cleaned,
    sampling_rate=1000,
    boundaries=[2.0, 1 / 30],
    iterations=10,
    silent=False,
):
    raise NotImplementedError("_flw_rvt_continuous is not implemented yet. Try _flw_rvt_cycle")



def _flw_rvt_plot(rvt, flw_cleaned, sampling_rate):
    plt.figure()
    plt.title("Respiratory Volume per Time (RVT) from airflow")
    plt.xlabel("Time [s]")
    plt.plot(
        rescale(flw_cleaned, to=[np.nanmin(rvt), np.nanmax(rvt)]),
        label="Cleaned Flow",
        color="#CFD8DC",
    )
    plt.plot(rvt, label="RVT", color="#00BCD4")
    plt.legend()
    tickpositions = plt.gca().get_xticks()[1:-1]
    plt.xticks(tickpositions, [tickposition / sampling_rate for tickposition in tickpositions])


