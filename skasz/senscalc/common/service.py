"""
The service layer is responsible for turning validated inputs into the relevant calculation inputs,
calling any calculation functions and collating the results.
"""
from typing import List, Optional, Tuple, TypedDict

import astropy.units as u

from skasz.senscalc.common.beam import calculate_multi_weighting, calculate_weighting
from skasz.senscalc.common.model import (
    BeamSize,
    ConfusionNoise,
    ContinuumWeightingRequestParams,
    Limit,
    WeightingInput,
    WeightingMultiInput,
    WeightingSpectralMode,
    ZoomWeightingRequestParams,
)

# TypedDicts to constrain response dicts to match the OpenAPI spec -----------


class ConfusionNoiseResponse(TypedDict):
    """
    ConfusionNoiseResponse is a typed dictionary constrained to match the
    schema of a confusion noise JSON object, as contained in the parent JSON
    result of a weighting endpoint query.
    """

    value: float | int
    limit_type: str


class BeamSizeResponse(TypedDict):
    """
    BeamSizeResponse is a typed dictionary constrained to match the schema of
    the JSON object outlining the synthesized beam size, as contained in the
    parent JSON result of a weighting endpoint query.
    """

    beam_maj_scaled: float
    beam_min_scaled: float
    beam_pa: float


class SingleWeightingResponse(TypedDict):
    """
    SingleWeightingResponse is a typed dictionary constrained to match the
    schema of a single weighting calculation, as performed for the main
    continuum or zoom weighting calculation and for each subband frequency.
    """

    weighting_factor: float
    sbs_conv_factor: float
    confusion_noise: ConfusionNoiseResponse
    beam_size: BeamSizeResponse


class SubbandWeightingResponse(SingleWeightingResponse):
    subband_freq_centre: u.Quantity


class ContinuumWeightingResponse(SingleWeightingResponse):
    """
    A typed dictionary constrained to match the schema of
    the weighting endpoint query for continuum.
    """

    subbands: Optional[list[SubbandWeightingResponse]]


class SingleZoomWeightingResponse(SingleWeightingResponse):
    """
    A typed dictionary constrained to match the schema of
    the weighting endpoint query for each line.
    """

    freq_centre: u.Quantity


# end TypedDict definitions --------------------------------------------------


def get_continuum_weighting_response(
    user_input: ContinuumWeightingRequestParams,
) -> ContinuumWeightingResponse:
    """
    Converts the input into the relevant calculation inputs, calls the calculation functions with them, then combines the results into the correct response body object.
    """

    params = WeightingInput(
        freq_centre=user_input.freq_centre,
        dec=user_input.dec,
        weighting_mode=user_input.weighting_mode,
        robustness=user_input.robustness,
        subarray_configuration=user_input.subarray_configuration,
        spectral_mode=user_input.spectral_mode,
        taper=user_input.taper,
        telescope=user_input.telescope,
    )
    main_result = calculate_weighting(params)

    response = ContinuumWeightingResponse(
        weighting_factor=main_result.weighting_factor,
        sbs_conv_factor=main_result.surface_brightness_conversion_factor.value,
        confusion_noise=convert_confusion_noise_to_response(
            main_result.confusion_noise
        ),
        beam_size=convert_beam_size_to_response(main_result.beam_size),
    )

    if user_input.subband_freq_centres:
        subband_params = WeightingMultiInput(
            freq_centres=user_input.subband_freq_centres,
            dec=user_input.dec,
            weighting_mode=user_input.weighting_mode,
            robustness=user_input.robustness,
            subarray_configuration=user_input.subarray_configuration,
            spectral_mode=user_input.spectral_mode,
            taper=user_input.taper,
            telescope=user_input.telescope,
        )
        response["subbands"] = [
            SubbandWeightingResponse(
                subband_freq_centre=subband_result.freq_centre,
                weighting_factor=subband_result.weighting_factor,
                sbs_conv_factor=subband_result.surface_brightness_conversion_factor.value,
                confusion_noise=convert_confusion_noise_to_response(
                    subband_result.confusion_noise
                ),
                beam_size=convert_beam_size_to_response(subband_result.beam_size),
            )
            for subband_result in calculate_multi_weighting(subband_params)
        ]

    return response


def get_zoom_weighting_response(
    user_input: ZoomWeightingRequestParams,
) -> list[SingleZoomWeightingResponse]:
    """
    Converts the input into the relevant calculation inputs, calls the calculation functions with them, then combines the results into the correct response body object.
    """
    # Currently the zoom calculator shows only the spectral results, so always use 'line' in the lookup table
    spectral_mode = WeightingSpectralMode.LINE

    params = WeightingMultiInput(
        freq_centres=user_input.freq_centres,
        dec=user_input.dec,
        weighting_mode=user_input.weighting_mode,
        robustness=user_input.robustness,
        subarray_configuration=user_input.subarray_configuration,
        spectral_mode=spectral_mode,
        taper=user_input.taper,
        telescope=user_input.telescope,
    )
    return [
        SingleZoomWeightingResponse(
            freq_centre=single_result.freq_centre,
            weighting_factor=single_result.weighting_factor,
            sbs_conv_factor=single_result.surface_brightness_conversion_factor.value,
            confusion_noise=convert_confusion_noise_to_response(
                single_result.confusion_noise
            ),
            beam_size=convert_beam_size_to_response(single_result.beam_size),
        )
        for single_result in calculate_multi_weighting(params)
    ]


def convert_confusion_noise_to_response(
    confusion_noise: ConfusionNoise,
) -> ConfusionNoiseResponse:
    # If confusion noise is labelled as an upper limit set it to
    # have a value of 0 Jy and a label of 'value'
    if confusion_noise.limit == Limit.UPPER:
        return {"value": 0, "limit_type": Limit.VALUE.value}
    return {
        "value": confusion_noise.value.value,
        "limit_type": confusion_noise.limit.value,
    }


def convert_beam_size_to_response(beam_size: BeamSize) -> BeamSizeResponse:
    return {
        "beam_maj_scaled": beam_size.beam_maj.value,
        "beam_min_scaled": beam_size.beam_min.value,
        "beam_pa": beam_size.beam_pa.value,
    }


def get_subbands(
    n_subbands: int, freq_centre: float, bandwidth: float
) -> Tuple[List[float], float]:
    """
    Function to get the centres (and common width) of the N subbands of
    the spectral window

    Note: the units for the freq_centre and bandwidth should be the same, then the outputs will also be in those units

    :param n_subbands: Number of subbands to split the spectral window into
    :param freq_centre: Central frequency of the spectral window
    :param bandwidth: Total bandwidth of the spectral window
    :return: A tuple containing a list of the frequency centres for the subbands, and the subband width
    """
    left = freq_centre - (0.5 * bandwidth)
    subband_width = bandwidth / n_subbands
    return [
        left + ((i + 0.5) * subband_width) for i in range(n_subbands)
    ], subband_width
