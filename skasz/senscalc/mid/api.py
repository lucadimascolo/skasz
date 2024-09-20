"""
These functions map to the API paths, with the returned value being the API response

Connexion maps the function name to the operationId in the OpenAPI document path
"""
import logging
from http import HTTPStatus

from skasz.senscalc.common.api import ResponseTuple, error_handler
from skasz.senscalc.common.service import (
    get_continuum_weighting_response,
    get_zoom_weighting_response,
)
from skasz.senscalc.mid.service import (
    ContinuumSensitivityResponse,
    SingleZoomSensitivityResponse,
    get_continuum_calculate_response,
    get_subarray_response,
    get_zoom_calculate_response,
)
from skasz.senscalc.mid.validation import (
    validate_and_convert_continuum_weighting_params,
    validate_and_convert_zoom_weighting_params,
    validate_and_set_defaults_for_continuum,
    validate_and_set_defaults_for_zoom,
)

LOGGER = logging.getLogger("senscalc")


@error_handler
def continuum_calculate(**kwargs) -> ResponseTuple[ContinuumSensitivityResponse]:
    """
    Function which HTTP GET requests to /mid/continuum/calculate are routed to.

    Sends the requests parameters to the service layer and builds a Response
    from the calculator output.

    :param kwargs: the HTTP parameters
    :return: a tuple of the response body (which is either a :class:`ska_ost_senscalc.mid.service.SensitivityResponse`
        or an :class:`ErrorResponse`) and HTTP status, which Connexion will wrap into a Response
    """
    result = get_continuum_calculate_response(
        validate_and_set_defaults_for_continuum(kwargs)
    )
    # Connexion will create a response from the body, status, header tuple
    return (
        result,
        HTTPStatus.OK,
    )


@error_handler
def zoom_calculate(**kwargs) -> ResponseTuple[list[SingleZoomSensitivityResponse]]:
    """
    Function which HTTP GET requests to /mid/zoom/calculate are routed to.

    Sends the requests parameters to the service layer and builds a Response
    from the calculator output.

    :param kwargs: the HTTP parameters
    :return: a tuple of the response body (which is either a :class:`ska_ost_senscalc.mid.service.SensitivityResponse`
        or an :class:`ErrorResponse`) and HTTP status, which Connexion will wrap into a Response
    """
    result = get_zoom_calculate_response(validate_and_set_defaults_for_zoom(kwargs))
    # Connexion will create a response from the body, status, header tuple
    return (
        result,
        HTTPStatus.OK,
    )


@error_handler
def continuum_weighting(**kwargs):
    result = get_continuum_weighting_response(
        validate_and_convert_continuum_weighting_params(**kwargs)
    )
    return (
        result,
        HTTPStatus.OK,
    )


@error_handler
def zoom_weighting(**kwargs):
    result = get_zoom_weighting_response(
        validate_and_convert_zoom_weighting_params(**kwargs)
    )
    return (
        result,
        HTTPStatus.OK,
    )


def subarrays():
    """
    Function that GET requests to the /subarrays endpoint are mapped to.

    Returns a response containing a list of available subarrays
    """
    try:
        LOGGER.debug("Request received for MID subarrays")
        return (
            get_subarray_response(),
            HTTPStatus.OK,
        )
    except Exception as err:
        LOGGER.exception("Exception occurred with MID api.subarrays")
        return (
            {
                "title": "Internal Server Error",
                "detail": repr(err),
            },
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )
