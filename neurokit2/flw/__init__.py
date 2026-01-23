"""Submodule for NeuroKit."""

# Aliases
from .flw_amplitude import flw_amplitude
from .flw_clean import flw_clean
from .flw_peaks import flw_peaks
from .flw_process import flw_process
from .flw_rate import flw_rate
from .flw_rrv import flw_rrv
from .flw_rvt import flw_rvt
from .flw_symmetry import flw_symmetry
from .flw_time import flw_time
from .flw_rav import flw_rav
from .flw_onsets import find_onsets
from .flw_simulate import flw_simulate
from .flw_flat import flw_flat

__all__ = [
    "flw_clean",
    "flw_peaks",
    "flw_amplitude",
    "flw_process",
    "flw_rrv",
    "flw_rvt",
    "flw_rate",
    "flw_symmetry",
    "flw_time",
    "flw_rav",
    "find_onsets",
    "flw_simulate",
    "flw_flat"
]
