# -*- coding: utf-8 -*-
from warnings import warn

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..complexity import entropy_approximate, entropy_sample, fractal_dfa
from ..misc import NeuroKitWarning
from ..signal import signal_power, signal_rate
from ..signal.signal_formatpeaks import _signal_formatpeaks_sanitize
from ..stats import mad

def flw_rrv(flw_rate, troughs=None, sampling_rate=100, show=False, silent=True):
    """**Respiratory Rate Variability (RRV)**

        Computes time domain and frequency domain features for Respiratory Rate Variability (RRV)
        analysis.

        Parameters
        ----------
        flw_rate : array
            Array containing the respiratory rate, produced by :func:`.signal_rate`.
        troughs : dict
            The samples at which the inhalation onsets occur.
            Dict returned by :func:`flw_peaks` (Accessible with the key, ``"FLW_Troughs"``).
            Defaults to ``None``.
        sampling_rate : int
            The sampling frequency of the signal (in Hz, i.e., samples/second).
        show : bool
            If ``True``, will return a Poincaré plot, a scattergram, which plots each breath-to-breath
            interval against the next successive one. The ellipse centers around the average
            breath-to-breath interval. Defaults to ``False``.
        silent : bool
            If ``False``, warnings will be printed. Default to ``True``.

        Returns
        -------
        DataFrame
            A DataFrame consisting of the computed RRV metrics, which includes:
            "RRV_RMSSD", "RRV_MeanBB", "RRV_SDBB", "RRV_SDSD", "RRV_CVBB", "RRV_CVSD", "RRV_MedianBB",
            "RRV_MadBB", "RRV_MCVBB", "RRV_VLF", "RRV_LF", "RRV_HF", "RRV_LFHF", "RRV_LFn", "RRV_HFn",
            "RRV_SD1", "RRV_SD2", "RRV_SD2SD1", "RRV_ApEn", "RRV_SampEn",


        See Also
        --------
        signal_rate, flw_peaks, signal_power, entropy_sample, entropy_approximate

        Examples
        --------


        References
        ----------
        * Soni, R., & Muniyandi, M. (2019). Breath rate variability: a novel measure to study the
          meditation effects. International Journal of Yoga, 12(1), 45.

        """
    # Sanitize input
    flw_rate, troughs = _flw_rrv_formatinput(flw_rate, troughs, sampling_rate)

    # Get raw and interpolated R-R intervals
    bbi = np.diff(troughs) / sampling_rate * 1000
    flw_period = 60 * sampling_rate / flw_rate

    # Get indices
    rrv = {}  # Initialize empty dict
    rrv.update(_flw_rrv_time(bbi))
    rrv.update(
        _flw_rrv_frequency(flw_period, sampling_rate=sampling_rate, show=show, silent=silent)
    )
    rrv.update(_flw_rrv_nonlinear(bbi))

    rrv = pd.DataFrame.from_dict(rrv, orient="index").T.add_prefix("RRV_")

    if show:
        _flw_rrv_plot(bbi)

    return rrv


# =============================================================================
# Methods (Domains)
# =============================================================================


def _flw_rrv_time(bbi):
    diff_bbi = np.diff(bbi)
    out = {}  # Initialize empty dict

    # Mean based
    out["RMSSD"] = np.sqrt(np.mean(diff_bbi ** 2))

    out["MeanBB"] = np.nanmean(bbi)
    out["SDBB"] = np.nanstd(bbi, ddof=1)
    out["SDSD"] = np.nanstd(diff_bbi, ddof=1)

    out["CVBB"] = out["SDBB"] / out["MeanBB"]
    out["CVSD"] = out["RMSSD"] / out["MeanBB"]

    # Robust
    out["MedianBB"] = np.nanmedian(bbi)
    out["MadBB"] = mad(bbi)
    out["MCVBB"] = out["MadBB"] / out["MedianBB"]

    #    # Extreme-based
    #    nn50 = np.sum(np.abs(diff_rri) > 50)
    #    nn20 = np.sum(np.abs(diff_rri) > 20)
    #    out["pNN50"] = nn50 / len(rri) * 100
    #    out["pNN20"] = nn20 / len(rri) * 100
    #
    #    # Geometrical domain
    #    bar_y, bar_x = np.histogram(rri, bins=range(300, 2000, 8))
    #    bar_y, bar_x = np.histogram(rri, bins="auto")
    #    out["TINN"] = np.max(bar_x) - np.min(bar_x)  # Triangular Interpolation of the NN Interval Histogram
    #    out["HTI"] = len(rri) / np.max(bar_y)  # HRV Triangular Index

    return out


def _flw_rrv_frequency(
    flw_period,
    vlf=(0, 0.04),
    lf=(0.04, 0.15),
    hf=(0.15, 0.4),
    sampling_rate=1000,
    method="welch",
    show=False,
    silent=True,
):
    power = signal_power(
        flw_period,
        frequency_band=[vlf, lf, hf],
        sampling_rate=sampling_rate,
        method=method,
        max_frequency=0.5,
        show=show,
    )
    power.columns = ["VLF", "LF", "HF"]
    out = power.to_dict(orient="index")[0]

    if silent is False:
        for frequency in out.keys():
            if out[frequency] == 0.0:
                warn(
                    "The duration of recording is too short to allow"
                    " reliable computation of signal power in frequency band " + frequency + "."
                                                                                             " Its power is returned as zero.",
                    category=NeuroKitWarning,
                )

    # Normalized
    total_power = np.sum(power.values)
    out["LFHF"] = out["LF"] / out["HF"]
    out["LFn"] = out["LF"] / total_power
    out["HFn"] = out["HF"] / total_power

    return out


def _flw_rrv_nonlinear(bbi):
    diff_bbi = np.diff(bbi)
    out = {}

    # Poincaré plot
    out["SD1"] = np.sqrt(np.std(diff_bbi, ddof=1) ** 2 * 0.5)
    out["SD2"] = np.sqrt(2 * np.std(bbi, ddof=1) ** 2 - 0.5 * np.std(diff_bbi, ddof=1) ** 2)
    out["SD2SD1"] = out["SD2"] / out["SD1"]

    # CSI / CVI
    #    T = 4 * out["SD1"]
    #    L = 4 * out["SD2"]
    #    out["CSI"] = L / T
    #    out["CVI"] = np.log10(L * T)
    #    out["CSI_Modified"] = L ** 2 / T

    # Entropy
    out["ApEn"] = entropy_approximate(bbi, dimension=2)[0]
    out["SampEn"] = entropy_sample(bbi, dimension=2, tolerance=0.2 * np.std(bbi, ddof=1))[0]

    # DFA
    if len(bbi) / 10 > 16:
        out["DFA_alpha1"] = fractal_dfa(bbi, scale=np.arange(4, 17), multifractal=False)[0]
        # For multifractal
        mdfa_alpha1, _ = fractal_dfa(
            bbi, multifractal=True, q=np.arange(-5, 6), scale=np.arange(4, 17)
        )
        for k in mdfa_alpha1.columns:
            out["MFDFA_alpha1_" + k] = mdfa_alpha1[k].values[0]

    if len(bbi) > 65:
        out["DFA_alpha2"] = fractal_dfa(bbi, scale=np.arange(16, 65), multifractal=False)[0]
        # For multifractal
        mdfa_alpha2, _ = fractal_dfa(
            bbi, multifractal=True, q=np.arange(-5, 6), scale=np.arange(16, 65)
        )
        for k in mdfa_alpha2.columns:
            out["MFDFA_alpha2_" + k] = mdfa_alpha2[k].values[0]
    return out


# =============================================================================
# Internals
# =============================================================================


def _flw_rrv_formatinput(flw_rate, troughs, sampling_rate=1000):
    if isinstance(flw_rate, tuple):
        flw_rate = flw_rate[0]
        troughs = None

    if isinstance(flw_rate, pd.DataFrame):
        df = flw_rate.copy()
        cols = [col for col in df.columns if "FLW_Rate" in col]
        if len(cols) == 0:
            cols = [col for col in df.columns if "FLW_Troughs" in col]
            if len(cols) == 0:
                raise ValueError(
                    "NeuroKit error: _flw_rrv_formatinput(): Wrong input, "
                    "we couldn't extract flw_rate and respiratory troughs indices."
                )
            else:
                flw_rate = signal_rate(
                    df[cols], sampling_rate=sampling_rate, desired_length=len(df)
                )
        else:
            flw_rate = df[cols[0]].values

    if troughs is None:
        try:
            troughs = _signal_formatpeaks_sanitize(df, key="FLW_Troughs")
        except NameError as e:
            raise ValueError(
                "NeuroKit error: _flw_rrv_formatinput(): "
                "Wrong input, we couldn't extract "
                "respiratory troughs indices."
            ) from e
    else:
        troughs = _signal_formatpeaks_sanitize(troughs, key="FLW_Troughs")

    return flw_rate, troughs


def _flw_rrv_plot(bbi):
    # Axes
    ax1 = bbi[:-1]
    ax2 = bbi[1:]

    # Compute features
    poincare_features = _flw_rrv_nonlinear(bbi)
    sd1 = poincare_features["SD1"]
    sd2 = poincare_features["SD2"]
    mean_bbi = np.mean(bbi)

    # Plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    plt.title("Poincaré Plot", fontsize=20)
    plt.xlabel("BB_n (s)", fontsize=15)
    plt.ylabel("BB_n+1 (s)", fontsize=15)
    plt.xlim(min(bbi) - 10, max(bbi) + 10)
    plt.ylim(min(bbi) - 10, max(bbi) + 10)
    ax.scatter(ax1, ax2, c="b", s=4)

    # Ellipse plot feature
    ellipse = matplotlib.patches.Ellipse(
        xy=(mean_bbi, mean_bbi),
        width=2 * sd2 + 1,
        height=2 * sd1 + 1,
        angle=45,
        linewidth=2,
        fill=False,
    )
    ax.add_patch(ellipse)
    ellipse = matplotlib.patches.Ellipse(
        xy=(mean_bbi, mean_bbi), width=2 * sd2, height=2 * sd1, angle=45
    )
    ellipse.set_alpha(0.02)
    ellipse.set_facecolor("blue")
    ax.add_patch(ellipse)

    # Arrow plot feature
    sd1_arrow = ax.arrow(
        mean_bbi,
        mean_bbi,
        -sd1 * np.sqrt(2) / 2,
        sd1 * np.sqrt(2) / 2,
        linewidth=3,
        ec="r",
        fc="r",
        label="SD1",
    )
    sd2_arrow = ax.arrow(
        mean_bbi,
        mean_bbi,
        sd2 * np.sqrt(2) / 2,
        sd2 * np.sqrt(2) / 2,
        linewidth=3,
        ec="y",
        fc="y",
        label="SD2",
    )

    plt.legend(handles=[sd1_arrow, sd2_arrow], fontsize=12, loc="best")

    return fig
