import logging
from functools import wraps
from http import HTTPStatus
from typing import Callable, Optional, Tuple, TypedDict, TypeVar, Union

LOGGER = logging.getLogger("senscalc")


class ErrorResponse(TypedDict):
    """
    ErrorResponse represents the JSON body returned by the API when validation of
    the request fails or there is another error.

    It should match the OpenAPI specification
    """

    title: str
    detail: Optional[str]


T = TypeVar("T")
ResponseTuple = Tuple[Union[T, ErrorResponse], HTTPStatus]


def error_handler(
    api_fn: Callable[[str], ResponseTuple]
) -> Callable[[str], ResponseTuple]:
    """
    A decorator function to catch general errors and wrap in the correct HTTP response,
    otherwise Flask just returns a generic error message which isn't very useful.

    :param api_fn: A function which an HTTP request is mapped
        to and returns an HTTP response
    """

    @wraps(api_fn)
    def wrapper(*args, **kwargs):
        try:
            LOGGER.debug(
                "Request to %s with args: %s and kwargs: %s", api_fn, args, kwargs
            )
            return api_fn(*args, **kwargs)
        except ValueError as err:
            LOGGER.exception(
                "ValueError occurred when calling the API function %s likely some semantic"
                " validation failed",
                api_fn,
            )
            return (
                ErrorResponse(
                    title="Validation Error",
                    detail=";".join(str(a) for a in err.args),
                ),
                HTTPStatus.BAD_REQUEST,
            )
        except Exception as err:
            LOGGER.exception(
                "Exception occurred when calling the API function %s with args %s",
                api_fn,
                err.args,
            )
            return (
                ErrorResponse(title="Internal Server Error"),
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    return wrapper
