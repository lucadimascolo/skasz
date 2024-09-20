from math import sqrt

import numpy as np
from astropy import units as u
from astropy.table import Table
from scipy.interpolate import interp1d

from skasz.senscalc.common.model import BeamSize, ConfusionNoise, Limit
from skasz.senscalc.utilities import STATIC_DATA_PATH

data_path = STATIC_DATA_PATH / "lookups/"


class ConfusionNoiseTable:
    """
    Encapsulation of the beam weighting lookup table to be used for SKA MID and LOW weighting factor and
    beam size calculations.
    """

    def __init__(self):
        self._table = Table.read(data_path / "confusion_noise.ecsv")

        # original look up table in linear space, converting the table to log space for
        # the calculation
        for k in self._table.keys():
            self._table[k] = np.log10(self._table[k])

    def get_confusion_noise(
        self, frequency: u.Quantity, beam: BeamSize
    ) -> ConfusionNoise:
        """
        method to calculate the confusion noise of beam
        """
        log_frequency = np.log10(frequency.value)

        beam_geomean = sqrt(beam.beam_min.value * beam.beam_maj.value)
        log_beam_geomean = np.log10(beam_geomean)

        sigma_min = self._table["sigma"].min(axis=0)
        sigma_max = self._table["sigma"].max(axis=0)

        sigma_m = interp1d(
            self._table["beam"],
            self._table["sigma"],
            bounds_error=False,
            axis=0,
            fill_value=(sigma_min, sigma_max),
        )(log_beam_geomean)

        sigma = interp1d(
            self._table["frequency"][0],
            sigma_m,
            fill_value="extrapolate",
        )(log_frequency)

        confusion_noise = 10**sigma

        if log_beam_geomean < self._table["beam"].min():
            lim = Limit.UPPER
        elif log_beam_geomean > self._table["beam"].max():
            lim = Limit.LOWER
        else:
            lim = Limit.VALUE

        return ConfusionNoise(value=confusion_noise * u.Jy, limit=lim)
