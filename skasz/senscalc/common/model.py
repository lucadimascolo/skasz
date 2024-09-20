import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import astropy.units as u
from astropy.coordinates import Latitude
from astropy.units import Quantity

from skasz.senscalc.low.model import CalculatorInput
from skasz.senscalc.subarray import LOWArrayConfiguration, MIDArrayConfiguration
from skasz.senscalc.utilities import Telescope


class Weighting(Enum):
    """
    Enumeration for different weighting
    """

    NATURAL = "natural"
    ROBUST = "robust"
    UNIFORM = "uniform"


class WeightingSpectralMode(Enum):
    """
    Enumeration spectral modes supported by the calculator, which are used in the look up table.
    """

    LINE = "line"
    CONTINUUM = "continuum"


@dataclass
class ContinuumWeightingRequestParams:
    """
    Represents the parameters sent to the /continuum/weighting end points, after they have been deserialised into enums and astropy objects.
    """

    spectral_mode: WeightingSpectralMode
    telescope: Telescope
    weighting_mode: Weighting
    subarray_configuration: LOWArrayConfiguration | MIDArrayConfiguration
    dec: Latitude
    freq_centre: Quantity
    taper: Quantity = 0.0 * u.arcsec
    robustness: int = 0
    subband_freq_centres: Optional[list[Quantity]] = None


@dataclass
class ZoomWeightingRequestParams:
    """
    Represents the parameters sent to the /zoom/weighting end points, after they have been deserialised into enums and astropy objects.
    """

    telescope: Telescope
    weighting_mode: Weighting
    subarray_configuration: LOWArrayConfiguration | MIDArrayConfiguration
    dec: Latitude
    freq_centres: list[Quantity]
    taper: Quantity = 0.0 * u.arcsec
    robustness: int = 0


class Limit(Enum):
    """
    Enumeration for different types of limit
    """

    UPPER = "upper limit"
    LOWER = "lower limit"
    VALUE = "value"


@dataclasses.dataclass
class BeamSize:
    beam_maj: u.Quantity
    beam_min: u.Quantity
    beam_pa: u.Quantity


@dataclasses.dataclass
class ConfusionNoise:
    value: u.Quantity
    limit: Limit


@dataclasses.dataclass(kw_only=True)
class _WeightingInput:
    dec: Latitude
    weighting_mode: Weighting
    subarray_configuration: Union[MIDArrayConfiguration, LOWArrayConfiguration]
    telescope: Telescope
    spectral_mode: WeightingSpectralMode
    robustness: int = 0
    taper: u.Quantity


@dataclasses.dataclass(kw_only=True)
class WeightingInput(_WeightingInput):
    freq_centre: u.Quantity


@dataclasses.dataclass(kw_only=True)
class WeightingMultiInput(_WeightingInput):
    freq_centres: list[u.Quantity]


@dataclasses.dataclass
class WeightingResult:
    weighting_factor: float
    surface_brightness_conversion_factor: u.Quantity
    beam_size: BeamSize
    confusion_noise: ConfusionNoise


@dataclasses.dataclass
class WeightingMultiResultElement(WeightingResult):
    freq_centre: u.Quantity


@dataclasses.dataclass
class CalculatorInputPSS(CalculatorInput):
    """
    This dataclass represents the internal model of the Calculator for the PSS mode.

    The following units are implicitly assumed (first three parameters are inherited
    from CalculatorInput):
    - freq_centre, bandwidth: [MHz]
    - duration: [h]
    - chan_width: [Hz]
    - dm: [pc/cm^3]
    - pulse_period, intrinsic_pulse_width: [ms]
    """

    _: dataclasses.KW_ONLY
    chan_width: float
    dm: float
    pulse_period: float
    intrinsic_pulse_width: float
