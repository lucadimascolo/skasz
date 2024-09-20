import numpy as np
from astropy import units as u

from skasz.senscalc.common.model import CalculatorInputPSS

# Correction factor for quantisation loss in beamformed data
BETA = 1.05
# Correction factor for FFT-based search loss in beamformed data
EPSILON = 0.77


def _find_scatter_broadening(dm: float, frequency_ghz: float) -> float:
    """
    Calculate scatter broadening for a given dispersion measure and observing frequency
    using the relation from Bhat et al (2004).

    :param dm: Dispersion measure in pc/cm^3
    :param frequency_ghz: Frequency in GHz

    :return: scatter broadening in milliseconds
    """
    if dm > 0.0:
        return_value = 10 ** (
            -6.46
            + (0.154 * np.log10(dm))
            + (1.07 * np.log10(dm) ** 2)
            - (3.86 * np.log10(frequency_ghz))
        )
    else:
        return_value = 0.0

    return return_value


def _find_dispersion_broadening(
    dm: float, frequency_mhz: float, chan_width_mhz: float
) -> float:
    """
    Calculate pulse broadening due to interstellar dispersion.

    :param dm: Dispersion measure in pc/cm^3
    :param frequency_mhz: Representative freqency in MHz
    :param chan_width_mhz: Channel frequency resolution in MHz

    :return: Pulse broadening due to dispersion in milliseconds
    """
    return 8.3e6 * dm * chan_width_mhz / frequency_mhz**3


def _find_observed_pulse_width(
    intrinsic_width: float,
    dm: float,
    frequency_mhz: float,
    chan_width_mhz: float,
) -> float:
    """
    For a pulse with intrinsic_width in milliseconds, calculate
    the observed pulse width taking dispersive and scattering
    broadening into account.

    :param intrinsic_width: Intrinsic pulse width in milliseconds
    :param dm: Dispersion measure in pc/cm^3
    :param frequency_mhz: Representative freqency in MHz
    :param chan_width_mhz: Channel frequency resolution in MHz

    :return: Observed pulse width in milliseconds
    """
    return np.sqrt(
        intrinsic_width**2
        + _find_dispersion_broadening(dm, frequency_mhz, chan_width_mhz) ** 2
        + _find_scatter_broadening(dm, frequency_mhz / 1e3)
    )


def convert_continuum_to_bf_sensitivity(
    continuum_sensitivity: u.Quantity,
    calculator_input: CalculatorInputPSS,
) -> u.Quantity:
    """
    Convert a given continuum sensitivity to beamformed senstivity for an observation
    defined by calculator_input.

    :param continuum_sensitivity: Continuum sensitivity
    :param calculator_input: an instance of BeamformedInput with validated user parameters
    :return: Folded pulse sensitivity in the same units as continuum_sensitivity
    """
    # Work out the effective pulse width
    effective_width = _find_observed_pulse_width(
        intrinsic_width=calculator_input.intrinsic_pulse_width,
        dm=calculator_input.dm,
        frequency_mhz=calculator_input.freq_centre_mhz,
        chan_width_mhz=calculator_input.chan_width * 1e-6,  # convert to MHz
    )

    if calculator_input.pulse_period > effective_width:
        sensitivity = (
            (BETA / EPSILON)
            * continuum_sensitivity.value
            * np.sqrt(
                (calculator_input.num_stations - 1) / calculator_input.num_stations
            )
            * np.sqrt(
                effective_width / (calculator_input.pulse_period - effective_width)
            )
        )
    else:
        raise ValueError(
            "The effective pulse width (due to interstellar dispersion and"
            " scattering) is larger than the pulse period. Check your inputs."
        )
    return u.Quantity(sensitivity, continuum_sensitivity.unit * u.beam)
