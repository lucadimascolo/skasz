import dataclasses

from astropy.units import Quantity


@dataclasses.dataclass
class CalculatorInput:
    """
    This dataclass represents the internal model of the Calculator for continuum and
    zoom modes.
    """

    _: dataclasses.KW_ONLY
    freq_centre_mhz: float
    bandwidth_mhz: float
    num_stations: int
    pointing_centre: str
    integration_time_h: float
    elevation_limit: float


@dataclasses.dataclass
class CalculatorResult:
    """
    This dataclass represents the internal result of a single Calculation.
    """

    sensitivity: float
    units: str

    def _to_quantity(self) -> Quantity:
        """Convert into an astropy Quantity"""
        return Quantity(self.sensitivity, unit=self.units)

    def __eq__(self, other):
        return self._to_quantity() == other._to_quantity()

    def __le__(self, other):
        return self._to_quantity() <= other._to_quantity()

    def __lt__(self, other):
        return self._to_quantity() < other._to_quantity()

    def __ge__(self, other):
        return self._to_quantity() >= other._to_quantity()

    def __gt__(self, other):
        return self._to_quantity() > other._to_quantity()
