# -*- coding: utf-8 -*-
import pandas as pd
from ..rsp.rsp_rav import rsp_rav

def flw_rav(
    amplitude,
    peaks,
    troughs=None,
):
    """**Respiratory Amplitude Variability (RAV)**

    Computes indices of amplitude variability, such as the mean and SD of the amplitude, and the
    RMSSD of the successive differences.

    .. note::

      This is an exploratory feature. If you manage to find studies and literature on RAV, please
      let us know by opening an issue on GitHub. Adding more indices (similar to HRV) would be
      trivial, but having some evidence as for its usefulness would be prerequisite.

    Parameters
    ----------
    amplitude : Union[list, np.array, pd.Series]
        The amplitude signal as returned by :func:`.flw_amplitude`.
    peaks : list or array or DataFrame or Series or dict
        The samples at which the inhalation peaks occur. If a dict or a DataFrame is passed, it is
        assumed that these containers were obtained with :func:`.flw_findpeaks`.
    troughs : list or array or DataFrame or Series or dict
        The samples at which the troughs occur. If a dict or a DataFrame is passed, it is
        assumed that these containers were obtained with :func:`.flw_findpeaks`. This argument can
        be inferred from the ``peaks`` argument if the information.

    Returns
    -------
    pd.DataFrame
        A DataFrame of containing the following columns with RAV indices.

    See Also
    --------
    rsp_amplitude, rsp_rrv

    Examples
    --------


    """

    return rsp_rav(amplitude, peaks, troughs)
