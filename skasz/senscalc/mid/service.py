"""
The service layer is responsible for turning validated inputs into the relevant calculation inputs,
calling any calculation functions and collating the results.
"""

from inspect import signature
from typing import Optional, TypedDict

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.units import Quantity

from skasz.senscalc.common.service import get_subbands
from skasz.senscalc.common.spectropolarimetry import (
    SpectropolarimetryInput,
    SpectropolarimetryResults,
    get_spectropolarimetry_results,
)
from skasz.senscalc.mid.calculator import Calculator
from skasz.senscalc.mid.validation import MID_CONTINUUM_CHANNEL_WIDTH_KHZ
from skasz.senscalc.subarray import SubarrayStorage
from skasz.senscalc.utilities import Telescope

subarray_storage = SubarrayStorage(Telescope.MID)


class SingleSubbandResponse(TypedDict):
    subband_freq_centre: Quantity
    sensitivity: Optional[Quantity]
    integration_time: Optional[Quantity]


class ContinuumSensitivityResponse(TypedDict):
    """
    Typed dictionary constrained to match the OpenAPI schema for the
    response body of a single continuum sensitivity calculation.
    """

    continuum_sensitivity: Optional[Quantity]
    continuum_integration_time: Optional[Quantity]

    spectral_sensitivity: Optional[Quantity]
    spectral_integration_time: Optional[Quantity]

    continuum_subband_sensitivities: Optional[list[SingleSubbandResponse]]
    continuum_subband_integration_times: Optional[list[SingleSubbandResponse]]

    spectropolarimetry_results: SpectropolarimetryResults


class SingleZoomSensitivityResponse(TypedDict):
    """
    Typed dictionary constrained to match the OpenAPI schema for the
    response body of a single zoom sensitivity calculation.
    """

    freq_centre: Quantity
    spectral_sensitivity: Optional[Quantity]
    spectral_integration_time: Optional[Quantity]

    spectropolarimetry_results: SpectropolarimetryResults


# Get the keywords of the Calculator constructor
KWARGS_CALCULATOR = [
    p.name
    for p in signature(Calculator).parameters.values()
    if p.kind == p.KEYWORD_ONLY
]


def _create_calculator(params) -> Calculator:
    """
    Returns an instance of the calculator from the params.

    We filter the params using the Calculator kwargs. This is legacy code,
    eventually we will refactor the calculator to take an object for an input and we won't need this
    """
    # Keep only the params in the list of constructor inputs
    filtered_params = {k: v for k, v in params.items() if k in KWARGS_CALCULATOR}
    return Calculator(**filtered_params)


def get_continuum_calculate_response(params: dict) -> ContinuumSensitivityResponse:
    """
    Extract the params from the request, convert them into the relevant calculator inputs,
    perform the calculations and collect the results into the response body.
    """
    # Parse the target
    target = SkyCoord(
        params["pointing_centre"],
        frame="icrs",
        unit=(u.hourangle, u.deg),
    )
    params["target"] = target

    # Main results
    result = {}

    # Create the calculator for the main continuum result
    continuum_calculator = _create_calculator(params)
    # Create the calculator for the spectral results, which are returned in the continuum response body
    spectral_params = params.copy()
    # For the spectral calculation, the bandwidth used in the calculation should be the effective resolution,
    # which is the intrinsic channel width multiplied by the spectral_averaging_factor
    effective_resolution_hz = (
        MID_CONTINUUM_CHANNEL_WIDTH_KHZ
        * 1e3
        * spectral_params.get("spectral_averaging_factor", 1)
    )
    spectral_params["bandwidth_hz"] = effective_resolution_hz
    spectral_calculator = _create_calculator(spectral_params)

    if params.get("integration_time_s"):
        sensitivity = continuum_calculator.calculate_sensitivity(
            params["integration_time_s"]
        ).to(u.Jy)
        result.update({"continuum_sensitivity": sensitivity[0]})
        # Calculate spectral sensitivity using effective resolution for bandwidth
        spectral_sensitivity = spectral_calculator.calculate_sensitivity(
            params["integration_time_s"]
        ).to(u.Jy)
        result.update({"spectral_sensitivity": spectral_sensitivity[0]})

    if params.get("sensitivity_jy"):
        integration_time = continuum_calculator.calculate_integration_time(
            params["sensitivity_jy"]
        ).to(u.s)
        result.update({"continuum_integration_time": integration_time[0]})

        # Calculate spectral integration time using resolution for bandwidth
        spectral_integration_time = spectral_calculator.calculate_integration_time(
            params["sensitivity_jy"]
        ).to(u.s)
        result.update({"spectral_integration_time": spectral_integration_time[0]})

    # Subbands - if subbands is 1 then we just return the main sensitivity calculation as the subband is the whole bandwidth
    if params.get("n_subbands", 1) != 1:
        subband_results = []
        subband_freq_centres_hz, subband_bandwidth = get_subbands(
            params["n_subbands"], params["freq_centre_hz"], params["bandwidth_hz"]
        )
        if integration_time_s := params.get("integration_time_s"):
            for subband_freq_centre_hz in subband_freq_centres_hz:
                # Create the calculator for each subband result
                subband_params = params.copy()
                subband_params["freq_centre_hz"] = subband_freq_centre_hz
                subband_params["bandwidth_hz"] = subband_bandwidth
                subband_calculator = _create_calculator(subband_params)

                sensitivity = subband_calculator.calculate_sensitivity(
                    integration_time_s
                ).to(u.Jy)
                subband_results.append(
                    {
                        "subband_freq_centre": u.Quantity(subband_freq_centre_hz, "Hz"),
                        "sensitivity": sensitivity[0],
                    }
                )
            result.update({"continuum_subband_sensitivities": subband_results})

        if params.get("sensitivity_jy"):
            for subband_freq_centre_hz, subband_sensitivity_jy in zip(
                subband_freq_centres_hz, params.get("subband_sensitivities_jy")
            ):
                # Create the calculator for each subband result
                subband_params = params.copy()
                subband_params["freq_centre_hz"] = subband_freq_centre_hz
                subband_params["bandwidth_hz"] = subband_bandwidth
                subband_calculator = _create_calculator(subband_params)

                integration_time = subband_calculator.calculate_integration_time(
                    subband_sensitivity_jy
                ).to(u.s)
                subband_results.append(
                    {
                        "subband_freq_centre": u.Quantity(subband_freq_centre_hz, "Hz"),
                        "integration_time": integration_time[0],
                    }
                )
            result.update({"continuum_subband_integration_times": subband_results})

    spectropolarimetry_input = SpectropolarimetryInput(
        bandwidth=u.Quantity(params["bandwidth_hz"], "Hz"),
        frequency=u.Quantity(params["freq_centre_hz"], "Hz"),
        effective_channel_width=u.Quantity(effective_resolution_hz, "Hz"),
    )

    result.update(
        {
            "spectropolarimetry_results": get_spectropolarimetry_results(
                spectropolarimetry_input
            )
        }
    )
    return ContinuumSensitivityResponse(**result)


def get_zoom_calculate_response(
    params,
) -> list[SingleZoomSensitivityResponse]:
    """
    Extract the params from the request, convert them into the relevant calculator inputs,
    perform the calculations and collect the results into the response body.
    """
    # Parse the target
    target = SkyCoord(
        params["pointing_centre"],
        frame="icrs",
        unit=(u.hourangle, u.deg),
    )
    params["target"] = target

    zoom_results = []

    if params.get("integration_time_s"):
        for zoom_freq_centre_hz, zoom_spectral_resolution_hz, total_bandwidth_hz in zip(
            params["freq_centres_hz"],
            params["spectral_resolutions_hz"],
            params["total_bandwidths_hz"],
        ):
            zoom_params = params.copy()
            zoom_params["freq_centre_hz"] = zoom_freq_centre_hz
            effective_resolution_hz = zoom_spectral_resolution_hz * params.get(
                "spectral_averaging_factor", 1
            )
            zoom_params["bandwidth_hz"] = effective_resolution_hz
            zoom_calculator = _create_calculator(zoom_params)
            sensitivity = zoom_calculator.calculate_sensitivity(
                params["integration_time_s"]
            ).to(u.Jy)

            spectropolarimetry_input = SpectropolarimetryInput(
                bandwidth=Quantity(total_bandwidth_hz, "Hz"),
                frequency=Quantity(zoom_freq_centre_hz, "Hz"),
                effective_channel_width=Quantity(effective_resolution_hz, "Hz"),
            )

            spectropolarimetry_results = get_spectropolarimetry_results(
                spectropolarimetry_input
            )

            zoom_results.append(
                SingleZoomSensitivityResponse(
                    freq_centre=zoom_freq_centre_hz * u.Hz,
                    spectral_sensitivity=sensitivity[0],
                    spectropolarimetry_results=spectropolarimetry_results,
                )
            )

    if params.get("sensitivities_jy"):
        for (
            zoom_freq_centre_hz,
            zoom_spectral_resolution_hz,
            zoom_sensitivity_jy,
            total_bandwidth_hz,
        ) in zip(
            params["freq_centres_hz"],
            params["spectral_resolutions_hz"],
            params["sensitivities_jy"],
            params["total_bandwidths_hz"],
        ):
            zoom_params = params.copy()
            zoom_params["freq_centre_hz"] = zoom_freq_centre_hz
            effective_resolution_hz = zoom_spectral_resolution_hz * params.get(
                "spectral_averaging_factor", 1
            )
            zoom_params["bandwidth_hz"] = effective_resolution_hz
            zoom_calculator = _create_calculator(zoom_params)

            integration_time = zoom_calculator.calculate_integration_time(
                zoom_sensitivity_jy
            ).to(u.s)

            spectropolarimetry_input = SpectropolarimetryInput(
                bandwidth=Quantity(total_bandwidth_hz, "Hz"),
                frequency=Quantity(zoom_freq_centre_hz, "Hz"),
                effective_channel_width=Quantity(effective_resolution_hz, "Hz"),
            )

            spectropolarimetry_results = get_spectropolarimetry_results(
                spectropolarimetry_input
            )

            zoom_results.append(
                SingleZoomSensitivityResponse(
                    freq_centre=zoom_freq_centre_hz * u.Hz,
                    spectral_integration_time=integration_time[0],
                    spectropolarimetry_results=spectropolarimetry_results,
                )
            )

    return zoom_results


def get_subarray_response():
    """
    return the appropriate subarray objects
    """
    return [
        {
            "name": subarray.name,
            "label": subarray.label,
            "n_ska": subarray.n_ska,
            "n_meer": subarray.n_meer,
        }
        for subarray in subarray_storage.list()
    ]
