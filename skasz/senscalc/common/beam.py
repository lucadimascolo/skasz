"""
Module holding functions useful to the MidCalculator
"""
import logging

import astropy.units as u
import numpy as np
from astropy.constants import c, k_B

from skasz.senscalc.common.confusion_noise_lookup import ConfusionNoiseTable
from skasz.senscalc.common.model import (
    ConfusionNoise,
    Weighting,
    WeightingInput,
    WeightingMultiInput,
    WeightingMultiResultElement,
    WeightingResult,
)
from skasz.senscalc.common.weighting_lookup import (
    BeamSize,
    WeightingTable,
    WeightingTableFactory,
)

logger = logging.getLogger("senscalc")


table_factory = WeightingTableFactory()
confusion_noise_table = ConfusionNoiseTable()


def calculate_weighting(weighting_input: WeightingInput) -> WeightingResult:
    """
    A function to calculate weighting factor, surface brightness sensitivity conversion factor,
    beam size and confusion noise based on a weighting and confusion noise lookup tables for a single frequency.

    :param weighting_input: WeightingInput structure containing all the parameters required to filter
        the lookup tables with and the declination and frequency used in the calculations.
    :return: WeightingResult structure containing the weighting factor, SBS conversion factor(s), confusion
        noise(s) (for Mid and Low) and beam size(s)
    """
    # Filter the weighting table based on given inputs
    weighting_table = table_factory.get_table(
        telescope=weighting_input.telescope,
        array_config=weighting_input.subarray_configuration,
        spectral_mode=weighting_input.spectral_mode,
        weighting_mode=weighting_input.weighting_mode,
        taper=weighting_input.taper,
        robustness=weighting_input.robustness,
    )
    if (
        weighting_input.weighting_mode is Weighting.NATURAL
        and weighting_input.taper == 0.0 * u.arcsec
    ):
        weighting_factor = 1.0
    else:
        weighting_factor = weighting_table.get_weighting_factor(weighting_input.dec)
    beam_size_response = weighting_table.get_beam_size(
        weighting_input.freq_centre, weighting_input.dec
    )
    sbs_conversion_factor = _get_surface_brightness_conversion_factor(
        weighting_input.freq_centre, beam_size_response
    )
    confusion_noise_response = confusion_noise_table.get_confusion_noise(
        weighting_input.freq_centre, beam_size_response
    )

    return WeightingResult(
        weighting_factor=weighting_factor,
        surface_brightness_conversion_factor=sbs_conversion_factor,
        beam_size=beam_size_response,
        confusion_noise=confusion_noise_response,
    )


def calculate_multi_weighting(
    weighting_input: WeightingMultiInput,
) -> list[WeightingMultiResultElement]:
    """
    A function to calculate weighting factor, surface brightness sensitivity conversion factor,
    beam size and confusion noise based on a weighting and confusion noise lookup tables. For the multiple
    frequencies specified in the input, then SBS conversion factor, beam size and confusion noise are returned
    for each of the frequencies.

    :param weighting_input: WeightingInput structure containing all the parameters required to filter
        the lookup tables with and the declination and frequencies used in the calculations.
    :return: WeightingResult structure containing the weighting factor, SBS conversion factor(s), confusion
        noise(s) (for Mid and Low) and beam size(s)
    """
    # Filter the weighting table based on given inputs
    weighting_table = table_factory.get_table(
        telescope=weighting_input.telescope,
        array_config=weighting_input.subarray_configuration,
        spectral_mode=weighting_input.spectral_mode,
        weighting_mode=weighting_input.weighting_mode,
        taper=weighting_input.taper,
        robustness=weighting_input.robustness,
    )
    if (
        weighting_input.weighting_mode is Weighting.NATURAL
        and weighting_input.taper == 0.0 * u.arcsec
    ):
        weighting_factor = 1.0
    else:
        weighting_factor = weighting_table.get_weighting_factor(weighting_input.dec)

    beam_size_responses = _calculate_beam_size(
        weighting_table, weighting_input.freq_centres, weighting_input.dec
    )
    sbs_conversion_factors = _calculate_sbs_conv_factor(
        weighting_input.freq_centres, beam_size_responses
    )
    confusion_noise_responses = _calculate_confusion_noise(
        weighting_input.freq_centres, beam_size_responses
    )

    return [
        WeightingMultiResultElement(
            freq_centre=freq_centre,
            weighting_factor=weighting_factor,
            surface_brightness_conversion_factor=sbs_conversion_factor,
            beam_size=beam_size_response,
            confusion_noise=confusion_noise_response,
        )
        for freq_centre, beam_size_response, sbs_conversion_factor, confusion_noise_response in zip(
            weighting_input.freq_centres,
            beam_size_responses,
            sbs_conversion_factors,
            confusion_noise_responses,
        )
    ]


def _calculate_beam_size(
    weighting_table: WeightingTable, frequencies: list[u.Quantity], dec: u.Quantity
) -> list[BeamSize]:
    beam_size_response = [
        weighting_table.get_beam_size(frequency, dec) for frequency in frequencies
    ]
    return beam_size_response


def _calculate_confusion_noise(
    frequencies: list[u.Quantity], beam_size: list[BeamSize]
) -> list[ConfusionNoise]:
    cn_responses = [
        confusion_noise_table.get_confusion_noise(f, bs)
        for (f, bs) in zip(frequencies, beam_size)
    ]
    return cn_responses


def _calculate_sbs_conv_factor(
    frequencies: list[u.Quantity], beam_size: list[BeamSize]
) -> list[u.Quantity]:
    sbs_conf_factor_response = [
        _get_surface_brightness_conversion_factor(frequency, beam_size[i])
        for i, frequency in enumerate(frequencies)
    ]
    return sbs_conf_factor_response


def _get_surface_brightness_conversion_factor(
    frequency: u.Quantity, beam: BeamSize
) -> u.Quantity:
    wavelength = c / frequency.to(u.s**-1)
    factor = (
        wavelength**2 / (2.0 * k_B * _get_beam_size_area(beam)) * (1 * u.sr)
    )  # fudge factor to remove the /sr

    return factor.to(u.K / u.Jy)


def _get_beam_size_area(beam: BeamSize) -> u.Quantity:
    theta = np.sqrt(beam.beam_maj * beam.beam_min).to(u.rad)
    return (np.pi * theta**2 / (4 * np.log(2))).to(u.sr)
