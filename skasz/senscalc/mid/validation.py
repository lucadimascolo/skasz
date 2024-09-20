"""
This module provides semantic validation for inputs to the Sensitivity Calculator,
including checking for required values, setting default values, and domain related checks.

Syntactic validation and basic validation, for example of min/max values of numbers, is done
by Connexion and the OpenAPI spec.
"""
from typing import Optional

import astropy.units as u
from astropy.coordinates import SkyCoord

from skasz.senscalc.common.model import (
    ContinuumWeightingRequestParams,
    Weighting,
    WeightingSpectralMode,
    ZoomWeightingRequestParams,
)
from skasz.senscalc.subarray import MIDArrayConfiguration, SubarrayStorage
from skasz.senscalc.utilities import Telescope

subarray_storage = SubarrayStorage(Telescope.MID)

DEFAULT_CALCULATE_PARAMS = {
    "pmv": 10,
    "el": 45,
    "alpha": 2.75,
    "n_subbands": 1,
}

DEFAULT_WEIGHTING_PARAMS = {
    "taper": 0.0,
}

SUBARRAY_CONFIGURATIONS_ALLOWED_FOR_ZOOM = [
    MIDArrayConfiguration.MID_AASTAR_ALL,
    MIDArrayConfiguration.MID_AASTAR_SKA_ONLY,
    MIDArrayConfiguration.MID_AA4_ALL,
    MIDArrayConfiguration.MID_AA4_MEERKAT_ONLY,
    MIDArrayConfiguration.MID_AA4_SKA_ONLY,
]

MID_CONTINUUM_CHANNEL_WIDTH_KHZ = 13.44

BAND_LIMITS = {
    "Band 1": [
        {"type": "ska", "limits": [0.35e9, 1.05e9]},
        {"type": "meerkat", "limits": [0.58e9, 1.015e9]},
        {"type": "mixed", "limits": [0.58e9, 1.015e9]},
    ],
    "Band 2": [
        {"type": "ska", "limits": [0.95e9, 1.76e9]},
        {"type": "meerkat", "limits": [0.95e9, 1.67e9]},
        {"type": "mixed", "limits": [0.95e9, 1.67e9]},
    ],
    "Band 3": [
        {"type": "ska", "limits": [1.65e9, 3.05e9]},
        {"type": "meerkat", "limits": [1.75e9, 3.05e9]},
        {"type": "mixed", "limits": [1.75e9, 3.05e9]},
    ],
    "Band 4": [{"type": "ska", "limits": [2.8e9, 5.18e9]}],
    "Band 5a": [{"type": "ska", "limits": [4.6e9, 8.5e9]}],
    "Band 5b": [{"type": "ska", "limits": [8.3e9, 15.4e9]}],
}

# For the subarrays not listed here, the full bandwidth is allowed defined by the limits above
MAXIMUM_BANDWIDTH_FOR_SUBARRAY = {
    MIDArrayConfiguration.MID_AA05_ALL: 800e6,
    MIDArrayConfiguration.MID_AA1_ALL: 800e6,
    MIDArrayConfiguration.MID_AA2_ALL: 800e6,
}


def validate_and_set_defaults_for_continuum(user_input: dict) -> dict:
    """
    :param user_input: the parameters from the HTTP request to /continuum/calculate
    :return: A new copy of the input dict, with defaults set for missing values
    :raises: ValueError if the input data is not valid
    """
    # Merge the default params and the user input into a new dict. The union operator for a dict will
    # take the rightmost value, ie if the user_input contains a key then it will not be overwritten by the defaults
    user_input = DEFAULT_CALCULATE_PARAMS | user_input

    if ("integration_time_s" in user_input) == ("sensitivity_jy" in user_input):
        raise ValueError(
            "Either 'sensitivity_jy' or 'integration_time_s' must be specified, but not both at once."
        )

    _validate_array_config_and_antennas(user_input)
    _validate_continuum_bandwidth_for_array_config(user_input)
    _validate_spectral_window(user_input)
    _validate_subband_parameters(user_input)

    return user_input


def validate_and_set_defaults_for_zoom(user_input: dict) -> dict:
    """
    :param user_input: the parameters from the HTTP request to /zoom/calculate
    :return: A new copy of the input dict, with defaults set for missing values
    :raises: ValueError if the input data is not valid
    """
    # Merge the default params and the user input into a new dict. The union operator for a dict will
    # take the rightmost value, ie if the user_input contains a key then it will not be overwritten by the defaults
    user_input = DEFAULT_CALCULATE_PARAMS | user_input

    if ("integration_time_s" in user_input) == ("sensitivities_jy" in user_input):
        raise ValueError(
            "Either 'sensitivities_jy' or 'integration_time_s' must be specified, but not both at once."
        )

    _validate_array_config_and_antennas(user_input)
    _validate_zoom_parameters(user_input)
    _validate_spectral_window(user_input)

    return user_input


def validate_and_convert_continuum_weighting_params(
    *,  # force kw-only args
    spectral_mode: str,
    freq_centre_hz: float | int,
    pointing_centre: str,
    subarray_configuration: str,
    weighting_mode: str,
    robustness: Optional[int] = None,
    taper: float = 0.0,
    subband_freq_centres_hz: Optional[list[float]] = None,
) -> ContinuumWeightingRequestParams:
    """
    Validate arguments for a MID weighting query, returning a typed
    encapsulation of those arguments.
    """
    err_msgs = []

    pointing_centre = _validate_pointing_centre(pointing_centre, err_msgs)

    weighting_mode = Weighting(weighting_mode)

    subarray_configuration = MIDArrayConfiguration(subarray_configuration)

    if weighting_mode == Weighting.ROBUST and robustness is None:
        err_msgs.append("Parameter 'robustness' should be set for 'robust' weighting")

    if err_msgs:
        raise ValueError("; ".join(err_msgs))

    freq_centre = u.Quantity(freq_centre_hz, unit=u.Hz)

    subband_freq_centres = (
        [
            subband_freq_centre_hz * u.Hz
            for subband_freq_centre_hz in subband_freq_centres_hz
        ]
        if subband_freq_centres_hz
        else []
    )

    return ContinuumWeightingRequestParams(
        telescope=Telescope.MID,
        spectral_mode=WeightingSpectralMode(spectral_mode),
        freq_centre=freq_centre,
        dec=pointing_centre.dec,
        subarray_configuration=subarray_configuration,
        weighting_mode=weighting_mode,
        robustness=0 if robustness is None else robustness,
        taper=taper * u.arcsec,
        subband_freq_centres=subband_freq_centres,
    )


def validate_and_convert_zoom_weighting_params(
    *,  # force kw-only args
    freq_centres_hz: list[float],
    pointing_centre: str,
    subarray_configuration: str,
    weighting_mode: str,
    robustness: Optional[int] = None,
    taper: float = 0,
) -> ZoomWeightingRequestParams:
    """
    TODO the validation for weighting is different to the other calculations, in that it converts
        the input to an object with astropy qualities, etc. We should unify the approaches, along with
        handling defaults properly and consistently

    Validate arguments for a MID weighting query, returning a typed
    encapsulation of those arguments.
    """
    err_msgs = []

    pointing_centre = _validate_pointing_centre(pointing_centre, err_msgs)

    if weighting_mode == Weighting.ROBUST and robustness is None:
        err_msgs.append("Parameter 'robustness' should be set for 'robust' weighting")

    if err_msgs:
        raise ValueError("; ".join(err_msgs))

    freq_centres = [
        u.Quantity(freq_centre_hz, unit=u.Hz) for freq_centre_hz in freq_centres_hz
    ]

    return ZoomWeightingRequestParams(
        telescope=Telescope.MID,
        freq_centres=freq_centres,
        dec=pointing_centre.dec,
        subarray_configuration=MIDArrayConfiguration(subarray_configuration),
        weighting_mode=Weighting(weighting_mode),
        robustness=0 if robustness is None else robustness,
        taper=taper * u.arcsec,
    )


def _validate_pointing_centre(pointing_centre: str, err_msgs: list) -> SkyCoord:
    try:
        return SkyCoord(pointing_centre, unit=(u.hourangle, u.deg))
    except ValueError:
        err_msgs.append(
            "Specified pointing centre is invalid, expected format HH:MM:SS[.ss]"
            " DD:MM:SS[.ss]."
        )


def _validate_zoom_parameters(user_input: dict) -> None:
    """
    :param user_input: the parameters from the HTTP request
    :raises: ValueError if the input data relevant for zoom mode is not valid
    """

    # Create a set with the length of each of the inputs. If they are all the same
    # length then the set should have one element which is the common length
    set_of_lengths = {
        len(user_input.get("freq_centres_hz", [])),
        len(user_input.get("spectral_resolutions_hz", [])),
        len(user_input.get("total_bandwidths_hz", [])),
    }

    # If they are not all the same length, or none of the values are set, raise a validation error
    if len(set_of_lengths) != 1 or 0 in set_of_lengths:
        raise ValueError(
            "Parameters 'freq_centres_hz', 'spectral_resolutions_hz' and 'total_bandwidths_hz' must all be set together and have the same length."
        )

    if "sensitivities_jy" in user_input:
        if len(user_input["sensitivities_jy"]) != next(iter(set_of_lengths)):
            raise ValueError(
                "Parameter 'sensitivities_jy' must be set to calculate an integration time for the zoom window. It should have the same length as 'freq_centres_hz', 'spectral_resolutions_hz' and 'total_bandwidths_hz'."
            )

    if user_input.get("freq_centres_hz") or user_input.get("spectral_resolutions_hz"):
        array_configuration = user_input.get(
            "subarray_configuration"
        )  # Could be none for a Custom input
        if (
            array_configuration
            and array_configuration not in SUBARRAY_CONFIGURATIONS_ALLOWED_FOR_ZOOM
        ):
            raise ValueError("No zoom modes are available for this array assembly.")

    if user_input.get("freq_centres_hz"):
        # Check that freq_centres_hz has the same length as spectral_resolutions_hz
        if not user_input.get("spectral_resolutions_hz"):
            raise ValueError(
                "Parameter 'spectral_resolutions_hz' must also be set when setting"
                " 'freq_centres_hz'."
            )
        if len(user_input.get("freq_centres_hz")) != len(
            user_input.get("spectral_resolutions_hz")
        ):
            raise ValueError(
                "Parameters 'spectral_resolutions_hz' and 'freq_centres_hz' must"
                " have the same length."
            )
        if user_input.get("sensitivity_jy"):
            if user_input.get("sensitivities_jy"):
                if len(user_input.get("sensitivities_jy")) != len(
                    user_input.get("spectral_resolutions_hz")
                ):
                    raise ValueError(
                        "Parameters 'sensitivities_jy' and"
                        " 'freq_centres_hz' must have the same length."
                    )
            else:
                # Check that sensitivities_jy are specified if requesting a sensitivity
                raise ValueError(
                    "Parameter 'sensitivities_jy' must be set when setting"
                    " 'freq_centres_hz' and 'sensitivity_jy'."
                )
    elif user_input.get("spectral_resolutions_hz"):
        raise ValueError(
            "Parameter 'freq_centres_hz' must also be set when setting"
            " 'spectral_resolutions_hz'."
        )


def _validate_subband_parameters(user_input: dict) -> None:
    """
    :param user_input: the parameters from the HTTP request
    :raises: ValueError if the input data relevant for subband calculations is not valid
    """
    if not user_input.get("sensitivity_jy"):
        # Validation currently only needs to be done for the sensitivity -> integration time calculation
        return

    n_subbands = user_input.get("n_subbands")
    subband_sensitivities_jy = user_input.get("subband_sensitivities_jy")

    if n_subbands > 1 and not subband_sensitivities_jy:
        raise ValueError(
            "Parameter 'subband_sensitivities_jy' must be set when setting 'sensitivity_jy' and"
            "'n_subbands' is greater than 1."
        )

    if subband_sensitivities_jy and n_subbands <= 1:
        raise ValueError(
            "Parameter 'n_subbands' must be greater than 1 when setting 'subband_sensitivities_jy' and 'sensitivity_jy'."
        )

    if (
        n_subbands
        and subband_sensitivities_jy
        and n_subbands != len(subband_sensitivities_jy)
    ):
        raise ValueError(
            "Parameter 'subband_sensitivities_jy' must have the same length as the value of 'n_subbands' for"
            "'n_subbands' greater than 1."
        )


def _validate_array_config_and_antennas(user_input):
    """
    Validates that if the user is using a custom array (ie by giving n_ska and n_meer)
    that they are not also passing an array_configuration. Also validates that both n_ska
    and n_meer are given.

    It does not validate that either an array_configuration or custom numbers are given, as the user
    can specify neither and the default will be used.
    """

    n_ska_but_not_n_meer = "n_ska" in user_input and "n_meer" not in user_input
    n_meer_but_not_n_ska = "n_meer" in user_input and "n_ska" not in user_input
    one_of_array_config_and_n_antennas = (
        "subarray_configuration" in user_input
    ) is not ("n_ska" in user_input or "n_meer" in user_input)

    if (
        n_ska_but_not_n_meer
        or n_meer_but_not_n_ska
        or not one_of_array_config_and_n_antennas
    ):
        raise ValueError(
            "Only 'array_configuration' or the number of antennas ('n_ska' AND 'n_meer') should be specified."
        )

    if "subarray_configuration" in user_input:
        user_input["subarray_configuration"] = MIDArrayConfiguration(
            user_input["subarray_configuration"]
        )


def _validate_continuum_bandwidth_for_array_config(user_input: dict) -> None:
    """
    Validates that the continuum bandwidth is less than the maximum for the subarray.
    For earlier subarrays, this is a static value.

    For the later ones, the full bandwidth is allowed and this is
    checked by checking the spectral window, as the limits change depending on the subarray.
    """
    if (
        "subarray_configuration" in user_input
        and user_input["subarray_configuration"] in MAXIMUM_BANDWIDTH_FOR_SUBARRAY
    ):
        max_continuum_bandwidth_hz = MAXIMUM_BANDWIDTH_FOR_SUBARRAY[
            user_input["subarray_configuration"]
        ]
        if user_input["bandwidth_hz"] > max_continuum_bandwidth_hz:
            raise ValueError(
                f"Maximum bandwidth ({max_continuum_bandwidth_hz * 1e-6} MHz) for this subarray has been exceeded."
            )


def _validate_spectral_window(user_input: dict) -> None:
    """
    For continuum input or each zoom window within a zoom input,
    validates that the band and array configuration combination is allowed.
    Then validates that the spectral window (defined by the frequency and bandwidth)
    is within the limits for the band and subarray.
    """
    if "subarray_configuration" in user_input:
        subarray = subarray_storage.load_by_label(
            user_input["subarray_configuration"].value
        )
        n_ska = subarray.n_ska
        n_meer = subarray.n_meer
    else:
        n_ska = user_input["n_ska"]
        n_meer = user_input["n_meer"]

    antenna_type = (
        "mixed" if n_ska > 0 and n_meer > 0 else ("ska" if n_ska > 0 else "meerkat")
    )

    try:
        if user_input["rx_band"] not in BAND_LIMITS:
            raise ValueError(f"{user_input['rx_band']} not supported.")

        limits = next(
            filter(
                lambda entry: entry["type"] == antenna_type,
                BAND_LIMITS[user_input["rx_band"]],
            )
        )["limits"]
    except StopIteration:
        # This means the next function raised an error as the 'type' is not present in the band.
        raise ValueError("Subarray configuration not allowed for given observing band.")

    def validate_within_band(freq_centre_hz, bandwidth_hz):
        min_freq = freq_centre_hz - bandwidth_hz / 2
        max_freq = freq_centre_hz + bandwidth_hz / 2
        if min_freq < limits[0] or max_freq > limits[1]:
            raise ValueError(
                "Spectral window defined by central frequency and bandwidth does not lie within the band range."
            )

    #  This function is used to validate both the continuum and zoom inputs
    if "freq_centre_hz" in user_input:
        validate_within_band(user_input["freq_centre_hz"], user_input["bandwidth_hz"])
    elif "freq_centres_hz" in user_input:
        [
            validate_within_band(freq_centre_hz, total_bandwidth_hz)
            for freq_centre_hz, total_bandwidth_hz in zip(
                user_input["freq_centres_hz"], user_input["total_bandwidths_hz"]
            )
        ]
