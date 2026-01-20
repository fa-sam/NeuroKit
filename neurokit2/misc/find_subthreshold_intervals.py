
def find_subthreshold_intervals(array, value_threshold, min_len):
    """

    Identify consecutive intervals in an array where values remain below a
    given threshold for at least a specified minimum length.

    :param array:
    :param value_threshold: float
        Upper bound for values to qualify as part of a subthreshold interval
        (i.e., values strictly less than this threshold are considered).

    :param min_len: int
        Minimum required length of a consecutive run for it to be included in the output.

    :return:
        intervals : list of tuple of int
        A list of (start, end) index tuples, where each tuple represents a
        contiguous interval of values below `threshold` whose length is at
        least `min_len`. The `start` index is inclusive and the `end`
        index is exclusive, matching Python slicing conventions.

    """
    groups = []
    current = []
    intervals = []
    for ind, x in enumerate(array):
        if x < value_threshold:
            current.append(ind)
        else:
            if len(current) > min_len:
                groups.append(current)
                intervals.append((current[0], current[-1]))
            current = []
    if len(current) > min_len:
        groups.append(current)
        intervals.append((current[0], current[-1]))

    return intervals

