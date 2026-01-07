# -*- coding: utf-8 -*-
import numpy as np

def find_onsets(
    flw_cleaned: np.ndarray,
    sampling_rate: float,
    cluster_gap_sec: float = 0.20,  # max gap between zero-crossings inside one cluster
    min_breath_sec: float = 0.30,  # min time between consecutive onsets
    probe_offset_sec: float = 0.05,  # how far from cluster edge to probe sign
    inspiration_positive: bool = True  # set False if your sensor uses opposite sign
):
    """
    Detect inspiration and expiration onset in a noisy airflow signal by taking
    the midpoint between the first and last zero-crossings inside each cluster.

    Parameters
    ----------
    flw_cleaned : np.ndarray
        Airflow signal (L/s or mL/s). Ideally, inspiration is positive and expiration negative.
    sampling_rate : float
        Sampling rate in Hz.
    cluster_gap_sec : float
        Max separation (seconds) between adjacent zero-crossings to consider them
        part of the same cluster.
    min_breath_sec : float
        Minimum time between accepted onsets (seconds) to avoid duplicates.
    probe_offset_sec : float
        Offset (seconds) from the cluster boundary to evaluate the sign before/after.
    inspiration_positive : bool
        If False, the function will flip the signal so that inspiration becomes positive.

    Returns
    -------
    dict
        {
          "FLW_InspirationOnsets":np.ndarray,
          "FLW_ExpirationOnsets": np.ndarray
        }
    """
    flw_cleaned = np.asarray(flw_cleaned).astype(float)
    N = len(flw_cleaned)
    if N == 0:
        raise ValueError("Empty input signal.")

    # Normalize sign orientation if needed
    if not inspiration_positive:
        flw_cleaned = -flw_cleaned


    # --- 1) Find all zero-crossing indices (sign changes between consecutive samples)
    # Use product rule: sign change if consecutive samples have opposite sign
    # We ignore exact zeros by nudging them slightly using previous non-zero sample.
    f = flw_cleaned.copy()
    # Replace exact zeros with tiny values preserving local sign tendency
    zero_mask = (f == 0)
    if zero_mask.any():
        # forward-fill then back-fill
        ff = f.copy()
        for i in range(1, N):
            if ff[i] == 0:
                ff[i] = ff[i - 1]
        bf = f.copy()
        for i in range(N - 2, -1, -1):
            if bf[i] == 0:
                bf[i] = bf[i + 1]
        f[zero_mask] = np.where(np.abs(ff[zero_mask]) >= np.abs(bf[zero_mask]), ff[zero_mask], bf[zero_mask])

    # Indices i where f[i] and f[i+1] have different signs
    sign_prod = f[:-1] * f[1:]
    zc_idx = np.where(sign_prod < 0)[0]  # crossing is between i and i+1
    if zc_idx.size == 0:
        # No zero-crossings found
        return {
            "FLW_InspirationOnsets": np.array([], dtype=int),
            "FLW_ExpirationOnsets": np.array([], dtype=int),

        }

    # --- 2) Cluster zero-crossings that are close in time
    cluster_gap = int(round(cluster_gap_sec * sampling_rate))
    clusters = []  # list of arrays of indices within each cluster
    current = [zc_idx[0]]
    for k in zc_idx[1:]:
        if k - current[-1] <= cluster_gap:
            current.append(k)
        else:
            clusters.append(np.array(current, dtype=int))
            current = [k]
    clusters.append(np.array(current, dtype=int))

    # --- 3) For each cluster, take the midpoint between first and last crossing
    probe_offset = int(round(probe_offset_sec * sampling_rate))
    min_breath = int(round(min_breath_sec * sampling_rate))

    insp_onsets = []
    exp_onsets = []

    last_accepted = -np.inf

    def robust_sign_at(idx):
        """Get sign at idx; if near zero, look a few samples around."""
        idx = int(np.clip(idx, 0, N - 1))
        val = f[idx]
        if val == 0:
            # search small neighborhood for non-zero
            w = min(5, N - 1)  # small window
            s = np.sign(f[max(0, idx - w):min(N, idx + w + 1)])
            nonzeros = s[s != 0]
            return int(np.sign(np.sum(nonzeros))) if nonzeros.size > 0 else 0
        return int(np.sign(val))

    for cl in clusters:
        first = int(cl[0])
        last = int(cl[-1])
        midpoint = int((first + last) // 2)
        midpoint = last

        # probe sign before and after the cluster (with an offset to escape the near-zero zone)
        before_idx = max(0, first - probe_offset)
        after_idx = min(N - 1, last + probe_offset)
        sign_before = robust_sign_at(before_idx)
        sign_after = robust_sign_at(after_idx)

        # Accept only if there's an overall sign change across the cluster
        if sign_before == 0 or sign_after == 0 or sign_before == sign_after:
            continue

        # Enforce minimal spacing between onsets to avoid duplicates
        if midpoint - last_accepted < min_breath:
            continue

        # Classify onset by overall sign change
        if sign_before < 0 and sign_after > 0:
            # expiration -> inspiration (start of inspiration)
            insp_onsets.append(midpoint)
            last_accepted = midpoint
        elif sign_before > 0 and sign_after < 0:
            # inspiration -> expiration (start of expiration)
            exp_onsets.append(midpoint)
            last_accepted = midpoint


    insp_onsets = np.array(insp_onsets, dtype=int)
    exp_onsets = np.array(exp_onsets, dtype=int)

    return {
        "FLW_InspirationOnsets": insp_onsets,
        "FLW_ExpirationOnsets": exp_onsets
    }

def _flw_fix_onsets(peaks, troughs, insp_onsets, exsp_onsets):
    events = np.concatenate([peaks, troughs])
    labels = np.array(['peak'] * len(peaks) + ['trough'] * len(troughs))
    order = np.argsort(events)
    events = events[order]
    labels = labels[order]

    # Iterate through consecutive pairs
    for i in range(len(events) - 1):
        start = events[i]
        end = events[i + 1]

        if labels[i] == 'trough' and labels[i + 1] == 'peak':
            # Trough → Peak: keep one inspiration onset
            del_candidates = exsp_onsets[(exsp_onsets > start) & (exsp_onsets < end)]
            for j in del_candidates:                # remove all inspiration onsets
                index = np.where(exsp_onsets == j)[0]
                exsp_onsets = np.delete(exsp_onsets, index)

            insp_candidate = insp_onsets[(insp_onsets > start) & (insp_onsets < end)]
            del_candidates = insp_candidate[1:]          # keep only one inspiration onset
            for j in del_candidates:
                index = np.where(insp_onsets == j)[0]
                insp_onsets = np.delete(insp_onsets, index)

        elif labels[i] == 'peak' and labels[i + 1] == 'trough':
            # Peak → Trough: keep one expiration onset
            del_candidates = insp_onsets[(insp_onsets > start) & (insp_onsets < end)]  # remove all inspiration onsets
            for j in del_candidates:
                index = np.where(insp_onsets == j)[0]
                insp_onsets = np.delete(insp_onsets, index)

            exsp_candidate = exsp_onsets[(exsp_onsets > start) & (exsp_onsets < end)]  # keep only one expiration onset
            del_candidates = exsp_candidate[1:]
            for j in del_candidates:
                index = np.where(exsp_onsets == j)[0]
                exsp_onsets = np.delete(exsp_onsets, index)

        last_peak_trough = events[-1]
        insp_onsets, exsp_onsets = _keep_only_first_onset_after_peakstroughs(insp_onsets, exsp_onsets, last_peak_trough)
    return insp_onsets, exsp_onsets



def _keep_only_first_onset_after_peakstroughs(insp, exsp, last_peak_trough):
    # Find first index in each array where value > X
    exsp_idx = np.searchsorted(exsp, last_peak_trough + 1)  # first greater than X
    insp_idx = np.searchsorted(insp, last_peak_trough + 1)

    # Determine which comes first in original order
    first_source = None
    first_value = None

    if exsp_idx < len(exsp) and insp_idx < len(insp):
        # Compare actual positions in original arrays
        if exsp[exsp_idx] < insp[insp_idx]:
            first_source = 'exsp'
            first_value = exsp[exsp_idx]
        else:
            first_source = 'insp'
            first_value = insp[insp_idx]
    elif exsp_idx < len(exsp):
        first_source = 'exsp'
        first_value = exsp[exsp_idx]
    elif insp_idx < len(insp):
        first_source = 'insp'
        first_value = insp[insp_idx]

    # Remove all other values greater than X
    if first_source == 'exsp':
        exsp = np.concatenate([exsp[:exsp_idx], [first_value]])
        insp = insp[:insp_idx]
    elif first_source == 'insp':
        insp = np.concatenate([insp[:insp_idx], [first_value]])
        exsp = exsp[:exsp_idx]

    return insp, exsp
