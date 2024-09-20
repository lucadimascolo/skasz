"""
This module provides semantic validation for inputs to the Sensitivity Calculator,
including checking for required values, setting default values, and domain related checks.

Syntactic validation and basic validation, for example of min/max values of numbers, is done
by Connexion and the OpenAPI spec.
"""

from typing import Optional

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.units import Quantity

from skasz.senscalc.common.model import (
    ContinuumWeightingRequestParams,
    Weighting,
    WeightingSpectralMode,
    ZoomWeightingRequestParams,
)
from skasz.senscalc.subarray import LOWArrayConfiguration, SubarrayStorage
from skasz.senscalc.utilities import Telescope

subarray_storage = SubarrayStorage(Telescope.LOW)

LOW_CONTINUUM_CHANNEL_WIDTH_KHZ = 24 * 781.25 / (4096 * 27 / 32)

DEFAULT_CONTINUUM_PARAMS = {
    "freq_centre_mhz": 200,
    "bandwidth_mhz": 300,
    "pointing_centre": "10:00:00 -30:00:00",
    "integration_time_h": 1,
    "elevation_limit": 20,
    "spectral_averaging_factor": 1,
}

# Note that bandwidth_mhz and spectral_resolution_hz are not exposed to the user
# They are defined here for ease of use within the backend
PSS_BANDWIDTH_MHZ = 118.518513664  # 8192 channels * 14467.592 Hz
PSS_CHAN_WIDTH_HZ = 14467.592
DEFAULT_PSS_PARAMS = {
    "freq_centre_mhz": 200,
    "pointing_centre": "10:00:00 -30:00:00",
    "integration_time_h": 1,
    "elevation_limit": 20,
    "dm": 0.0,
    "pulse_period": 33,  # Assume Crab as default (33 ms)
    "intrinsic_pulse_width": 0.004,  # Assume Crab as default (4 us)
}

DEFAULT_ZOOM_PARAMS = {
    "freq_centres_mhz": [200],
    "total_bandwidths_khz": [
        390.6
    ],  # Take the default value as the total bandwidth of the narrowest zoom window allowed for all subarrays
    "spectral_resolutions_hz": [
        14.1285
    ],  # Take the narrowest channel allowed in zoom mode
    "pointing_centre": "10:00:00 -30:00:00",
    "integration_time_h": 1,
    "elevation_limit": 20,
}

# The spectral resolutions for the zoom windows are given by (781250.0 * 32/27)/(4096 * 16) multiplied by increasing powers of 2
# The allowed total bandwidths for zoom mode are then the chanel resolutions multiplied by the number of channels (1728), and converted to kHz
BANDWIDTH_PRECISION_KHZ = 1  # round to nearest 0.1 kHz, to avoid difference in floating point numbers when calculated in the front end
ALLOWED_ZOOM_TOTAL_BANDWIDTHS_KHZ = [
    round(
        (2 ** (N - 1) * (781250 * 32 / 27) / (4096 * 16)) * 1728 * 1e-3,
        BANDWIDTH_PRECISION_KHZ,
    )
    for N in range(1, 9)
]

MAX_FREQUENCY_MHZ = 350
MIN_FREQUENCY_MHZ = 50

# For the subarrays not listed here, the full bandwidth is allowed defined by the limits above
MAXIMUM_BANDWIDTH_MHZ_FOR_SUBARRAY = {
    "LOW_AA05_all": 75,
    "LOW_AA1_all": 75,
    "LOW_AA2_all": 150,
    "LOW_AA2_core_only": 150,
}


def validate_and_set_defaults_for_continuum(user_input: dict) -> dict:
    """
    :param user_input: the parameters from the HTTP request for the /api/low/continuum/calculate request
    :return: A new copy of the input dict, with defaults set for missing values
    :raises: ValueError if the input data is not valid
    """
    # Merge the default params and the user input into a new dict. The union operator for a dict will
    # take the rightmost value, ie if the user_input contains a key then it will not be overwritten by the defaults
    user_input = DEFAULT_CONTINUUM_PARAMS | user_input

    err_msgs = []

    _validate_num_stations_or_subarray_configuration(user_input, err_msgs)
    _validate_max_continuum_bandwidth(user_input, err_msgs)
    _validate_spectral_window(
        user_input["freq_centre_mhz"], user_input["bandwidth_mhz"], err_msgs
    )
    _validate_pointing_centre(user_input, err_msgs)
    _validate_spectral_averaging_factor(user_input, err_msgs)

    if err_msgs:
        raise ValueError(*err_msgs)

    return user_input


def validate_and_set_defaults_for_zoom(user_input: dict) -> dict:
    """
    :param user_input: the parameters from the HTTP request for the /api/low/zoom/calculate request
    :return: A new copy of the input dict, with defaults set for missing values
    :raises: ValueError if the input data is not valid
    """
    # Merge the default params and the user input into a new dict. The union operator for a dict will
    # take the rightmost value, ie if the user_input contains a key then it will not be overwritten by the defaults
    user_input = DEFAULT_ZOOM_PARAMS | user_input

    err_msgs = []

    _validate_zoom_parameter_combinations(user_input)

    _validate_num_stations_or_subarray_configuration(user_input, err_msgs)
    _validate_zoom_bandwidth_for_subarray(user_input, err_msgs)

    for freq_centre_mhz, total_bandwidth_khz in zip(
        user_input["freq_centres_mhz"], user_input["total_bandwidths_khz"]
    ):
        _validate_spectral_window(freq_centre_mhz, total_bandwidth_khz * 1e-3, err_msgs)

    _validate_pointing_centre(user_input, err_msgs)

    if err_msgs:
        raise ValueError(*err_msgs)

    return user_input


def validate_and_set_defaults_for_pss(user_input: dict) -> dict:
    """
    :param user_input: the parameters from the HTTP request for the /api/low/pss/calculate request
    :return: A new copy of the input dict, with defaults set for missing values
    :raises: ValueError if the input data is not valid
    """

    # Merge the default params and the user input into a new dict.
    # The union operator for a dict will take the rightmost value
    # (i.e.) if the user_input contains a key then it will not be overwritten
    # by the defaults
    user_input = DEFAULT_PSS_PARAMS | user_input

    # Insert PSS bandwidth and channel width to the user_input
    user_input["bandwidth_mhz"] = PSS_BANDWIDTH_MHZ
    user_input["spectral_resolution_hz"] = PSS_CHAN_WIDTH_HZ

    err_msgs = []

    _validate_num_stations_or_subarray_configuration(user_input, err_msgs)

    _validate_spectral_window(
        user_input["freq_centre_mhz"], user_input["bandwidth_mhz"], err_msgs
    )
    _validate_pointing_centre(user_input, err_msgs)
    if user_input["intrinsic_pulse_width"] >= user_input["pulse_period"]:
        err_msgs.append("Intrinsic pulse width cannot be larger than the pulse period.")

    if err_msgs:
        raise ValueError(*err_msgs)

    return user_input


def validate_and_convert_continuum_weighting_params(
    *,  # force kw-only args
    spectral_mode: str,
    freq_centre_mhz: float | int,
    pointing_centre: str,
    subarray_configuration: str,
    weighting_mode: str,
    robustness: Optional[int] = None,
    subband_freq_centres_mhz: Optional[list[float]] = None,
) -> ContinuumWeightingRequestParams:
    """
    Validate arguments for a LOW weighting query, returning a typed
    encapsulation of those arguments.
    """
    err_msgs = []

    py_pointing_centre = _validate_pointing_centre(
        dict(pointing_centre=pointing_centre), err_msgs
    )

    # OpenAPI enum strings need converting to Python enum members
    weighting_mode = EnumConversion.to_weighting(weighting_mode, err_msgs)
    spectral_mode = WeightingSpectralMode(spectral_mode)

    subarray_configuration = EnumConversion.to_array_configuration(
        subarray_configuration, err_msgs
    )

    if weighting_mode == Weighting.ROBUST and robustness is None:
        err_msgs.append("Parameter 'robustness' should be set for 'robust' weighting")

    if err_msgs:
        raise ValueError("; ".join(err_msgs))

    # OpenAPI spec requires frequency in MHz, weighting expects it in Hz
    # weighting currently expects a list of frequencies instead of a single
    # frequency value
    freq_centre = Quantity(freq_centre_mhz * 1e6, unit=u.Hz)

    subband_freq_centres = (
        [
            Quantity(subband_freq_centre_mhz * 1e6, unit=u.Hz)
            for subband_freq_centre_mhz in subband_freq_centres_mhz
        ]
        if subband_freq_centres_mhz
        else []
    )

    return ContinuumWeightingRequestParams(
        telescope=Telescope.LOW,
        spectral_mode=spectral_mode,
        freq_centre=freq_centre,
        dec=py_pointing_centre.dec,
        subarray_configuration=subarray_configuration,
        weighting_mode=weighting_mode,
        robustness=0 if robustness is None else robustness,
        subband_freq_centres=subband_freq_centres,
    )


def validate_and_convert_zoom_weighting_params(
    *,  # force kw-only args
    freq_centres_mhz: list[float],
    pointing_centre: str,
    subarray_configuration: str,
    weighting_mode: str,
    robustness: Optional[int] = None,
) -> ZoomWeightingRequestParams:
    """
    TODO the validation for weighting is different to the other calculations, in that it converts
        the input to an object with astropy qualities, etc. We should unify the approaches, along with
        handling defaults properly and consistently

    Validate arguments for a LOW weighting query, returning a typed
    encapsulation of those arguments.
    """
    err_msgs = []

    pointing_centre = _validate_pointing_centre(
        dict(pointing_centre=pointing_centre), err_msgs
    )

    weighting_mode = EnumConversion.to_weighting(weighting_mode, err_msgs)
    subarray_configuration = EnumConversion.to_array_configuration(
        subarray_configuration, err_msgs
    )

    if weighting_mode == Weighting.ROBUST and robustness is None:
        err_msgs.append("Parameter 'robustness' should be set for 'robust' weighting")

    if err_msgs:
        raise ValueError("; ".join(err_msgs))

    freq_centres = [
        Quantity(freq_centre_mhz * 1e6, unit=u.Hz)
        for freq_centre_mhz in freq_centres_mhz
    ]

    return ZoomWeightingRequestParams(
        telescope=Telescope.LOW,
        freq_centres=freq_centres,
        dec=pointing_centre.dec,
        subarray_configuration=subarray_configuration,
        weighting_mode=weighting_mode,
        robustness=0 if robustness is None else robustness,
    )


def _validate_spectral_averaging_factor(user_input: dict, err_msgs: list):
    n_continuum_channels = (
        user_input["bandwidth_mhz"] * 1e3 / LOW_CONTINUUM_CHANNEL_WIDTH_KHZ
    )
    max_channels = int(n_continuum_channels // 2)  # Floor div
    valid = 1 <= user_input["spectral_averaging_factor"] <= max_channels
    if not valid:
        err_msgs.append(
            f"The spectral averaging factor must lie between 1 and {max_channels}"
        )


def _validate_num_stations_or_subarray_configuration(
    user_input: dict, err_msgs: list
) -> None:
    """
    Either num_stations or a subarray_configuration should be given by the user.

    :param user_input: the parameters passed to the API.
    :param err_msgs: the list of error messages to append a validation error to
    """
    if ("subarray_configuration" in user_input) == ("num_stations" in user_input):
        err_msgs.append(
            "Only 'subarray_configuration' or 'num_stations' should be specified."
        )


def _validate_spectral_window(
    freq_centre: float, bandwidth_mhz: float, err_msgs: list
) -> None:
    min_freq = freq_centre - bandwidth_mhz / 2
    max_freq = freq_centre + bandwidth_mhz / 2
    if min_freq < MIN_FREQUENCY_MHZ or max_freq > MAX_FREQUENCY_MHZ:
        err_msgs.append(
            "Spectral window defined by central frequency and bandwidth does"
            " not lie within the 50 - 350 MHz range."
        )


def _validate_pointing_centre(user_input: dict, err_msgs: list) -> SkyCoord:
    try:
        return SkyCoord(user_input["pointing_centre"], unit=(u.hourangle, u.deg))
    except ValueError:
        err_msgs.append(
            "Specified pointing centre is invalid, expected format HH:MM:SS[.ss]"
            " DD:MM:SS[.ss]."
        )


def _validate_zoom_parameter_combinations(user_input):
    # Create a set with the length of each of the inputs. If they are all the same
    # length then the set should have one element which is the common length
    set_of_lengths = {
        len(user_input.get("freq_centres_mhz", [])),
        len(user_input.get("spectral_resolutions_hz", [])),
        len(user_input.get("total_bandwidths_khz", [])),
    }

    # If they are not all the same length, or none of the values are set, raise a validation error
    if len(set_of_lengths) != 1 or 0 in set_of_lengths:
        raise ValueError(
            "Parameters 'freq_centres_mhz', 'spectral_resolutions_hz' and 'total_bandwidths_khz' must all be set together and have the same length."
        )


def _validate_max_continuum_bandwidth(user_input: dict, err_msgs: list):
    """
    Validates the maximum bandwidth allowed for a continuum calculation is allowed for the given subarray configuration.

    :param user_input: the parameters passed to the API.
    :param err_msgs: the list of error messages to append a validation error to
    """

    max_allowed_bandwidth = MAXIMUM_BANDWIDTH_MHZ_FOR_SUBARRAY.get(
        user_input.get("subarray_configuration"), MAX_FREQUENCY_MHZ - MIN_FREQUENCY_MHZ
    )

    if user_input["bandwidth_mhz"] > max_allowed_bandwidth:
        err_msgs.append(
            f"Maximum bandwidth ({max_allowed_bandwidth} MHz) for this subarray has been exceeded."
        )


def _validate_zoom_bandwidth_for_subarray(user_input: dict, err_msgs: list):
    """
    Validates the zoom mode is allowed for the given subarray configuration, by validating the total bandwidth

    :param user_input: the parameters passed to the API.
    :param err_msgs: the list of error messages to append a validation error to
    """
    # AA0.5 and AA1 are not supported in zoom mode so not defined in the API. AA2 only supports the larger zoom
    # modes (hence taking the final 4 elements here). The later arrays support all the zoom modes, as does a custom array.
    allowed_bandwidth_for_subarray = (
        ALLOWED_ZOOM_TOTAL_BANDWIDTHS_KHZ
        if ("subarray_configuration" not in user_input)
        or (
            user_input.get("subarray_configuration")
            not in ["LOW_AA2_core_only", "LOW_AA2_all"]
        )
        else ALLOWED_ZOOM_TOTAL_BANDWIDTHS_KHZ[-4:]
    )

    for total_bandwidth_khz in user_input["total_bandwidths_khz"]:
        if (
            not round(total_bandwidth_khz, BANDWIDTH_PRECISION_KHZ)
            in allowed_bandwidth_for_subarray
        ):
            err_msgs.append(
                f"Bandwidth {total_bandwidth_khz} not one the of allowed"
                f" values in zoom mode for {user_input.get('subarray_configuration', 'custom')} subarray configuration: {allowed_bandwidth_for_subarray}"
            )


class EnumConversion:
    """
    Utility class to convert OpenAPI enumeration members to Python Enum
    members.

    OpenAPI enums generally map to Python enums but their name will usually
    be formatted differently, as the Python convention is for enumeration
    member names to be all upper case. This class decouples the OpenAPI naming
    convention from that all-caps requirement.
    """

    @staticmethod
    def to_weighting(val: str, msgs: list[str]) -> Weighting:
        return EnumConversion._convert(Weighting, val, msgs)

    @staticmethod
    def to_array_configuration(val: str, msgs) -> LOWArrayConfiguration:
        return EnumConversion._convert(LOWArrayConfiguration, val, msgs)

    @staticmethod
    def _convert(cls, val: str, msgs: list[str]):
        try:
            return cls[val.upper()]
        except (ValueError, KeyError):
            msg = f"{val} could not be mapped to a {cls.__name__} enum member"
            msgs.append(msg)
