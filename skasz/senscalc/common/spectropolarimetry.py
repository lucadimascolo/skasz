from math import pi, sqrt
from typing import TypedDict

from astropy import units as u
from astropy.constants import c as speed_of_light


class SpectropolarimetryInput(TypedDict):
    bandwidth: u.Quantity
    frequency: u.Quantity
    effective_channel_width: u.Quantity


class SpectropolarimetryResults(TypedDict):
    fwhm_of_the_rmsf: u.Quantity
    max_faraday_depth_extent: u.Quantity
    max_faraday_depth: u.Quantity


def get_spectropolarimetry_results(
    spectropolarimetry_input: SpectropolarimetryInput,
) -> SpectropolarimetryResults:
    """
    :param spectropolarimetry_input: the input parameters required to calculate the spectropolarimetry results
    :return: A SpectropolarimetryResults with the values calculated
    """
    return SpectropolarimetryResults(
        fwhm_of_the_rmsf=calcuate_fwhm_of_the_rmsf(**spectropolarimetry_input),
        max_faraday_depth_extent=calculate_max_faraday_depth_extent(
            **spectropolarimetry_input
        ),
        max_faraday_depth=calculate_max_faraday_depth(**spectropolarimetry_input),
    )


def calcuate_fwhm_of_the_rmsf(
    bandwidth: u.Quantity, frequency: u.Quantity, effective_channel_width: u.Quantity
) -> u.Quantity:
    """
    See ReadTheDocs for more details on the astronomy and the equation.

    :param bandwidth: the total bandwidth of the spectral window
    :param frequency: the central frequency of the spectral window
    :param effective_channel_width: the effective channel width of the observation. ie the channel width multiplied by the spectral averaging
    :return: The rotation measure spread function of the full width at half maximum
    """
    first_channel_central_frequency = _get_first_channel_central_frequency(
        bandwidth, frequency, effective_channel_width
    )
    last_channel_central_frequency = _get_last_channel_central_frequency(
        bandwidth, frequency, effective_channel_width
    )

    result = (
        2
        * sqrt(3)
        / (
            (speed_of_light / first_channel_central_frequency) ** 2
            - (speed_of_light / last_channel_central_frequency) ** 2
        )
    )
    return result * u.rad


def calculate_max_faraday_depth_extent(
    bandwidth: u.Quantity, frequency: u.Quantity, effective_channel_width: u.Quantity
) -> u.Quantity:
    """
    See ReadTheDocs for more details on the astronomy and the equation.

    :param bandwidth: the total bandwidth of the spectral window
    :param frequency: the central frequency of the spectral window
    :param effective_channel_width: the effective channel width of the observation. ie the channel width multiplied by the spectral averaging
    :return: the maximum Faraday depth extent
    """
    last_channel_central_frequency = _get_last_channel_central_frequency(
        bandwidth, frequency, effective_channel_width
    )
    result = pi * (last_channel_central_frequency / speed_of_light) ** 2

    return result * u.rad


def calculate_max_faraday_depth(
    bandwidth: u.Quantity, frequency: u.Quantity, effective_channel_width: u.Quantity
) -> u.Quantity:
    """
    See ReadTheDocs for more details on the astronomy and the equation.

    :param bandwidth: the total bandwidth of the spectral window
    :param frequency: the central frequency of the spectral window
    :param effective_channel_width: the effective channel width of the observation. ie the channel width multiplied by the spectral averaging
    :return: the maximum Faraday depth
    """
    effective_channel_width = effective_channel_width.to("s^-1")
    first_channel_central_frequency = _get_first_channel_central_frequency(
        bandwidth, frequency, effective_channel_width
    )
    x = (
        2
        * speed_of_light**2
        * effective_channel_width
        / first_channel_central_frequency**3
    )
    y = 1 + 0.5 * (effective_channel_width / first_channel_central_frequency) ** 2

    result = sqrt(3) / (x * y)

    return result * u.rad


def _get_first_channel_central_frequency(
    bandwidth: u.Quantity, frequency: u.Quantity, effective_channel_width: u.Quantity
) -> u.Quantity:
    """
    :param bandwidth: the total bandwidth of the spectral window
    :param frequency: the central frequency of the spectral window
    :param effective_channel_width: the effective channel width of the observation. ie the channel width multiplied by the spectral averaging
    :return: the central frequency of the first channel, calculated by adding half the channel width to the start of the spectral window
    """
    lower_limit = frequency - 0.5 * bandwidth
    return (lower_limit + 0.5 * effective_channel_width).to("s^-1")


def _get_last_channel_central_frequency(
    bandwidth: u.Quantity, frequency: u.Quantity, effective_channel_width: u.Quantity
) -> u.Quantity:
    """
    :param bandwidth: the total bandwidth of the spectral window
    :param frequency: the central frequency of the spectral window
    :param effective_channel_width: the effective channel width of the observation. ie the channel width multiplied by the spectral averaging
    :return: the central frequency of the last channel, calculated by removing half the channel width from the end of the spectral window
    """
    upper_limit = frequency + 0.5 * bandwidth
    return (upper_limit - 0.5 * effective_channel_width).to("s^-1")
