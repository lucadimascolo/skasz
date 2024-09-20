"""
The service layer is responsible for turning validated inputs into the relevant calculation inputs,
calling any calculation functions and collating the results.
"""

from typing import List, Optional, Tuple, TypedDict

from astropy.units import Quantity

from skasz.senscalc.common.model import CalculatorInputPSS
from skasz.senscalc.common.pss import convert_continuum_to_bf_sensitivity
from skasz.senscalc.common.service import get_subbands
from skasz.senscalc.common.spectropolarimetry import (
    SpectropolarimetryInput,
    SpectropolarimetryResults,
    get_spectropolarimetry_results,
)
from skasz.senscalc.low.bright_source_lookup import BrightSourceCatalog
from skasz.senscalc.low.calculator import calculate_sensitivity
from skasz.senscalc.low.model import CalculatorInput
from skasz.senscalc.low.validation import LOW_CONTINUUM_CHANNEL_WIDTH_KHZ
from skasz.senscalc.subarray import SubarrayStorage
from skasz.senscalc.utilities import Telescope

subarray_storage = SubarrayStorage(Telescope.LOW)

FLUX_DENSITY_THRESHOLD_JY = 10.0


class SubbandResponse(TypedDict):
    subband_freq_centre: Quantity
    sensitivity: Quantity


class ContinuumSensitivityResponse(TypedDict):
    """
    Typed dictionary constrained to match the OpenAPI schema for the
    response body of a single continuum sensitivity calculation.
    """

    continuum_sensitivity: Quantity
    continuum_subband_sensitivities: Optional[List[SubbandResponse]]
    spectral_sensitivity: Quantity

    spectropolarimetry_results: SpectropolarimetryResults

    warning: Optional[str]


class SingleZoomSensitivityResponse(TypedDict):
    """
    Typed dictionary constrained to match the OpenAPI schema for the
    response body of a single zoom sensitivity calculation.
    """

    freq_centre: Quantity

    spectral_sensitivity: Quantity

    spectropolarimetry_results: SpectropolarimetryResults

    warning: Optional[str]


class PSSSensitivityResponse(TypedDict):
    """
    Typed dictionary constrained to match the OpenAPI schema for the
    response body of a single PSS sensitivity calculation.
    """

    # For pulsar search:
    folded_pulse_sensitivity: Quantity

    warning: Optional[str]


def convert_continuum_input_and_calculate(
    user_input: dict,
) -> ContinuumSensitivityResponse:
    """
    :param user_input: A kwarg dict of the HTTP parameters sent by the user
    :return: a SensitivityResponse containing the calculated sensitivity and its units
    """

    num_stations = _num_stations_from_input(user_input)

    continuum_calculator_input = CalculatorInput(
        freq_centre_mhz=user_input["freq_centre_mhz"],
        bandwidth_mhz=user_input["bandwidth_mhz"],
        num_stations=num_stations,
        pointing_centre=user_input["pointing_centre"],
        integration_time_h=user_input["integration_time_h"],
        elevation_limit=user_input["elevation_limit"],
    )
    effective_resolution_mhz = (LOW_CONTINUUM_CHANNEL_WIDTH_KHZ / 1e3) * user_input.get(
        "spectral_averaging_factor", 1
    )
    spectral_calculator_input = CalculatorInput(
        freq_centre_mhz=user_input["freq_centre_mhz"],
        bandwidth_mhz=effective_resolution_mhz,
        num_stations=num_stations,
        pointing_centre=user_input["pointing_centre"],
        integration_time_h=user_input["integration_time_h"],
        elevation_limit=user_input["elevation_limit"],
    )

    spectropolarimetry_input = SpectropolarimetryInput(
        bandwidth=Quantity(user_input["bandwidth_mhz"], "MHz"),
        frequency=Quantity(user_input["freq_centre_mhz"], "MHz"),
        effective_channel_width=Quantity(effective_resolution_mhz, "MHz"),
    )

    continuum_result, warning = _get_calculation_value(continuum_calculator_input)
    # Warnings will be identical, so we don't need this one:
    spectral_result, _ = _get_calculation_value(spectral_calculator_input)

    subband_sensitivities = _get_subband_sensitivities(user_input, num_stations)

    spectropolarimetry_results = get_spectropolarimetry_results(
        spectropolarimetry_input
    )

    return ContinuumSensitivityResponse(
        continuum_sensitivity=continuum_result,
        continuum_subband_sensitivities=subband_sensitivities,
        spectral_sensitivity=spectral_result,
        warning=warning,
        spectropolarimetry_results=spectropolarimetry_results,
    )


def convert_zoom_input_and_calculate(
    user_input: dict,
) -> [SingleZoomSensitivityResponse]:
    """
    :param user_input: A kwarg dict of the HTTP parameters sent by the user
    :return: a dict containing the calculated sensitivity and its units
    """
    num_stations = _num_stations_from_input(user_input)

    result = []
    for freq_centre_mhz, spectral_resolution_hz, total_bandwidth_khz in zip(
        user_input["freq_centres_mhz"],
        user_input["spectral_resolutions_hz"],
        user_input["total_bandwidths_khz"],
    ):
        effective_resolution_hz = spectral_resolution_hz * user_input.get(
            "spectral_averaging_factor", 1
        )
        calculator_input = CalculatorInput(
            freq_centre_mhz=freq_centre_mhz,
            bandwidth_mhz=effective_resolution_hz * 1e-6,  # Convert to MHz
            num_stations=num_stations,
            pointing_centre=user_input["pointing_centre"],
            integration_time_h=user_input["integration_time_h"],
            elevation_limit=user_input["elevation_limit"],
        )

        spectropolarimetry_input = SpectropolarimetryInput(
            bandwidth=Quantity(total_bandwidth_khz, "kHz"),
            frequency=Quantity(freq_centre_mhz, "MHz"),
            effective_channel_width=Quantity(effective_resolution_hz, "Hz"),
        )

        sensitivity, warning = _get_calculation_value(calculator_input)

        spectropolarimetry_results = get_spectropolarimetry_results(
            spectropolarimetry_input
        )

        result.append(
            SingleZoomSensitivityResponse(
                freq_centre=Quantity(freq_centre_mhz, "MHz"),
                spectral_sensitivity=sensitivity,
                warning=warning,
                spectropolarimetry_results=spectropolarimetry_results,
            )
        )

    return result


def convert_pss_input_and_calculate(user_input: dict) -> PSSSensitivityResponse:
    """
    :param user_input: A kwarg dict of the HTTP parameters sent by the user
    :return: a dict containing the calculated sensitivity and its units
    """
    num_stations = _num_stations_from_input(user_input)

    calculator_input_pss = CalculatorInputPSS(
        freq_centre_mhz=user_input["freq_centre_mhz"],
        bandwidth_mhz=user_input["bandwidth_mhz"],
        chan_width=user_input["spectral_resolution_hz"],
        num_stations=num_stations,
        pointing_centre=user_input["pointing_centre"],
        integration_time_h=user_input["integration_time_h"],
        elevation_limit=user_input["elevation_limit"],
        dm=user_input["dm"],
        pulse_period=user_input["pulse_period"],
        intrinsic_pulse_width=user_input["intrinsic_pulse_width"],
    )

    # First, estimate the corresponding continuum sensitivity
    continuum_input = CalculatorInput(
        freq_centre_mhz=calculator_input_pss.freq_centre_mhz,
        bandwidth_mhz=calculator_input_pss.chan_width * 1e-6,  # Convert to MHz
        num_stations=calculator_input_pss.num_stations,
        pointing_centre=calculator_input_pss.pointing_centre,
        integration_time_h=calculator_input_pss.integration_time_h,
        elevation_limit=calculator_input_pss.elevation_limit,
    )

    continuum_sensitivity, warning = _get_calculation_value(continuum_input)

    # Convert the continuum sensitivity to folded-pulse sensitivity
    folded_pulse_sensitivity = convert_continuum_to_bf_sensitivity(
        continuum_sensitivity,
        calculator_input_pss,
    )

    return PSSSensitivityResponse(
        folded_pulse_sensitivity=folded_pulse_sensitivity, warning=warning
    )


def get_subarray_response():
    """
    return the appropriate subarray objects
    """
    return [
        {
            "name": subarray.name,
            "label": subarray.label,
            "n_stations": subarray.n_stations,
        }
        for subarray in subarray_storage.list()
    ]


def _get_calculation_value(
    calculator_input: CalculatorInput,
) -> Tuple[Quantity, Optional[str]]:
    result = calculate_sensitivity(calculator_input)
    sensitivity = Quantity(result.sensitivity, result.units)
    warning = _check_for_warning(calculator_input)
    return sensitivity, warning


def _check_for_warning(calculator_input: CalculatorInput) -> Optional[str]:
    mwa_cat = BrightSourceCatalog(threshold_jy=FLUX_DENSITY_THRESHOLD_JY)
    if mwa_cat.check_for_bright_sources(
        calculator_input.pointing_centre, calculator_input.freq_centre_mhz
    ):
        return (
            "The specified pointing contains at least one source brighter "
            + f"than {FLUX_DENSITY_THRESHOLD_JY} Jy. Your observation may be "
            + "dynamic range limited."
        )
    return None


def _num_stations_from_input(user_input: dict) -> int:
    """
    If the user has given a subarray_configuration, extract the num_stations from that.
    Otherwise, use the value given by the user.

    Validation has checked that one and only on of these fields is present in the input.

    :param user_input: a dict of the parameters given by the user
    :return: the num_stations to use in the calculation
    """
    if "subarray_configuration" in user_input:
        subarray = subarray_storage.load_by_name(user_input["subarray_configuration"])
        return subarray.n_stations

    return user_input["num_stations"]


def _get_subband_sensitivities(
    user_input: dict, num_stations: int
) -> List[SubbandResponse]:
    if user_input.get("n_subbands", 1) == 1:
        return []

    subband_freq_centres_hz, subband_bandwidth = get_subbands(
        user_input["n_subbands"],
        user_input["freq_centre_mhz"],
        user_input["bandwidth_mhz"],
    )
    return [
        SubbandResponse(
            subband_freq_centre=Quantity(subband_frequency, "MHz"),
            sensitivity=_get_calculation_value(
                CalculatorInput(
                    freq_centre_mhz=subband_frequency,
                    bandwidth_mhz=subband_bandwidth,
                    num_stations=num_stations,
                    pointing_centre=user_input["pointing_centre"],
                    integration_time_h=user_input["integration_time_h"],
                    elevation_limit=user_input["elevation_limit"],
                )
            )[0],
        )
        for subband_frequency in subband_freq_centres_hz
    ]
