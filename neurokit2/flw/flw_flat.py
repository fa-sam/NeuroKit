import numpy as np

def flw_flat(flw_cleaned, sampling_rate=100, min_dur=2, tolerance=2):
    """
        It finds the flat intervals of the signal, where the change of consecutive values remain below a
        given threshold for at least a specified minimum duration.

        Parameters
        ----------
        flw_cleaned : np.ndarray
            Airflow signal (L/min). Ideally, inspiration is positive and expiration negative.
        sampling_rate : float
            Sampling rate in Hz.
        min_dur : float
            Minimum duration in seconds.
        tolerance: float
            Maximum allowed change between consecutive signal values for them to be
            considered part of a flat interval.

        Returns
        -------
        intervals : list of tuple of int
            A list of (start, end) index tuples, representing the flat region whose duration is at
            least `min_dur` seconds. The `start` index is inclusive and the `end`
            index is exclusive, matching Python slicing conventions.
    """

    min_len = int(min_dur *sampling_rate)
    # diff_array = np.abs(np.diff(flw_cleaned))
    # intervals = find_subthreshold_intervals(diff_array, tolerance, min_len)
    intervals = _find_intervals(flw_cleaned, min_len, tolerance)
    return intervals



def _find_intervals(arr, min_length, tolerance):
    intervals = []
    n = len(arr)

    start = 0
    end = 0
    while end < len(arr)-1:

        for end in range(start+1, n):

            current_min = min(arr[start:end])
            current_max = max(arr[start:end])

            # Check if the difference is within tolerance
            if current_max - current_min > tolerance:
                if (end - start-1 ) >= min_length:
                    intervals.append((start, end-1))
                break
            elif end == n-1:
                if (end - start+1) >= min_length:
                    intervals.append((start, end+1))
        start = end-1


    return intervals
