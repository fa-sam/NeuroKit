from ..rsp.rsp_simulate import rsp_simulate

def flw_simulate(
    duration=10,
    length=None,
    sampling_rate=1000,
    noise=0.01,
    respiratory_rate=15,
    random_state=None,
    random_state_distort="spawn",
):
    """**Simulate an airflow respiratory signal**

    Generate an artificial (synthetic) respiratory signal of a given duration
    and rate.

    Parameters
    ----------
    duration : int
        Desired length of duration (s).
    sampling_rate : int
        The desired sampling rate (in Hz, i.e., samples/second).
    length : int
        The desired length of the signal (in samples).
    noise : float
        Noise level (amplitude of the laplace noise).
    respiratory_rate : float
        Desired number of breath cycles in one minute.
    random_state : None, int, numpy.random.RandomState or numpy.random.Generator
        Seed for the random number generator. See for ``misc.check_random_state`` for further information.
    random_state_distort : {'legacy', 'spawn'}, None, int, numpy.random.RandomState or numpy.random.Generator
        Random state to be used to distort the signal. If ``"legacy"``, use the same random state used to
        generate the signal (discouraged as it creates dependent random streams). If ``"spawn"``, spawn
        independent children random number generators from the random_state argument. If any of the other types,
        generate independent children random number generators from the random_state_distort provided (this
        allows generating multiple version of the same signal distorted by different random noise realizations).

    See Also
    --------
    flw_clean, flw_findpeaks, signal_rate, flw_process, flw_plot

    Returns
    -------
    array
        Vector containing the respiratory signal.

    Examples
    --------
    .. ipython:: python

      import pandas as pd
      import neurokit2 as nk

      flw1 = nk.flw_simulate(duration=30)

      @savefig p_flw_simulate1.png scale=100%
      pd.DataFrame({"FLW_Simple": flw1}).plot(subplots=True)
      @suppress
      plt.close()

    References
    ----------
    * Noto, T., Zhou, G., Schuele, S., Templer, J., & Zelano, C. (2018). Automated analysis of
      breathing waveforms using BreathMetrics: A respiratory signal processing toolbox. Chemical Senses, 43(8), 583-597.

    """

    return rsp_simulate(duration, length, sampling_rate, noise, respiratory_rate, method="sinusoidal", random_state=random_state, random_state_distort=random_state_distort)
