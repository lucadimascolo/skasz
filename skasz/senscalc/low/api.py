"""
These functions map to the API paths, with the returned value being the API response

Connexion maps the function name to the operationId in the OpenAPI document path
"""
import logging
from http import HTTPStatus
from typing import List, Optional

from skasz.enscalc.common.api import ResponseTuple, error_handler
from skasz.enscalc.common.service import (
    ContinuumWeightingResponse,
    SingleZoomWeightingResponse,
    get_continuum_weighting_response,
    get_zoom_weighting_response,
)
from skasz.enscalc.low.service import (
    ContinuumSensitivityResponse,
    PSSSensitivityResponse,
    SingleZoomSensitivityResponse,
    convert_continuum_input_and_calculate,
    convert_pss_input_and_calculate,
    convert_zoom_input_and_calculate,
    get_subarray_response,
)
from skasz.enscalc.low.validation import (
    validate_and_convert_continuum_weighting_params,
    validate_and_convert_zoom_weighting_params,
    validate_and_set_defaults_for_continuum,
    validate_and_set_defaults_for_pss,
    validate_and_set_defaults_for_zoom,
)

LOGGER = logging.getLogger("senscalc")


@error_handler
def continuum_calculate(**kwargs) -> ResponseTuple[ContinuumSensitivityResponse]:
    """
    Function which HTTP GET requests to /api/low/continuum/calculate are routed to.

    :param kwargs: the HTTP parameters
    :return: a tuple of the response body (which is either a :class:`ska_ost_senscalc.low.service.SensitivityResponse`
        or an :class:`ErrorResponse`) and HTTP status, which Connexion will wrap into a Response
    """
    validated_params = validate_and_set_defaults_for_continuum(kwargs)
    return (
        convert_continuum_input_and_calculate(validated_params),
        HTTPStatus.OK,
    )


@error_handler
def subarrays():
    """
    Function that GET requests to the /api/low/subarrays are routed to.

    Returns a response containing a list of available subarrays
    """
    return (
        get_subarray_response(),
        HTTPStatus.OK,
    )


@error_handler
def zoom_calculate(**kwargs) -> ResponseTuple[list[SingleZoomSensitivityResponse]]:
    """
    Function which HTTP GET requests to /api/low/zoom/calculate are routed to.

    :param kwargs: the HTTP parameters
    :return: a tuple of the response body (which is either a :class:`ska_ost_senscalc.low.service.SensitivityResponse`
        or an :class:`ErrorResponse`) and HTTP status, which Connexion will wrap into a Response
    """
    validated_params = validate_and_set_defaults_for_zoom(kwargs)
    return (
        convert_zoom_input_and_calculate(validated_params),
        HTTPStatus.OK,
    )


@error_handler
def pss_calculate(**kwargs) -> ResponseTuple[PSSSensitivityResponse]:
    """
    Function which HTTP GET requests to /api/low/pss/calculate are routed to.

    :param kwargs: the HTTP parameters
    :return: a tuple of the response body (which is either a :class:`ska_ost_senscalc.low.service.SensitivityResponse`
        or an :class:`ErrorResponse`) and HTTP status, which Connexion will wrap into a Response
    """
    validated_params = validate_and_set_defaults_for_pss(kwargs)
    return (
        convert_pss_input_and_calculate(validated_params),
        HTTPStatus.OK,
    )


@error_handler
def continuum_weighting(
    *,  # force kw-only args
    spectral_mode: str,
    freq_centre_mhz: float | int,
    pointing_centre: str,
    subarray_configuration: str,
    weighting_mode: str,
    robustness: int = 0,
    subband_freq_centres_mhz: Optional[List[int]] = None,
    **other,
) -> ResponseTuple[ContinuumWeightingResponse]:
    if other:
        LOGGER.warning(f"Unexpected query argument(s): {other}")

    validated_and_updated_params = validate_and_convert_continuum_weighting_params(
        spectral_mode=spectral_mode,
        freq_centre_mhz=freq_centre_mhz,
        pointing_centre=pointing_centre,
        subarray_configuration=subarray_configuration,
        weighting_mode=weighting_mode,
        robustness=robustness,
        subband_freq_centres_mhz=subband_freq_centres_mhz,
    )

    return (
        get_continuum_weighting_response(validated_and_updated_params),
        HTTPStatus.OK,
    )


@error_handler
def zoom_weighting(
    *,  # force kw-only args
    freq_centres_mhz: list[float],
    pointing_centre: str,
    subarray_configuration: str,
    weighting_mode: str,
    robustness: int = 0,
    **other,
) -> ResponseTuple[list[SingleZoomWeightingResponse]]:
    if other:
        LOGGER.warning(f"Unexpected query argument(s): {other}")

    validated_and_updated_params = validate_and_convert_zoom_weighting_params(
        freq_centres_mhz=freq_centres_mhz,
        pointing_centre=pointing_centre,
        subarray_configuration=subarray_configuration,
        weighting_mode=weighting_mode,
        robustness=robustness,
    )

    return (
        get_zoom_weighting_response(validated_and_updated_params),
        HTTPStatus.OK,
    )
