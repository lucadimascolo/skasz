import os
from typing import Union

import astropy.units as u
from astropy.table import Table
from astropy.units import Quantity
from scipy.interpolate import interp1d

from skasz.senscalc.common.model import BeamSize, Weighting, WeightingSpectralMode
from skasz.senscalc.subarray import (
    LOWArrayConfiguration,
    MIDArrayConfiguration,
    SubarrayStorage,
)
from senscalc.utilities import STATIC_DATA_PATH, Telescope

lookup_path = STATIC_DATA_PATH / "lookups/"


class WeightingTable:
    def __init__(self, weighting_table, natural_table, reference_freq):
        self._table = weighting_table
        self._natural_table = natural_table
        self._reference_frequency = reference_freq

    def get_beam_size(self, freq_centre: Quantity, dec: Quantity) -> BeamSize:
        self._validate_declination(dec)
        interp_bmaj = interp1d(
            self._table["declination"],
            self._table["cleanbeam_bmaj"],
            fill_value="extrapolate",
        )
        interp_bmin = interp1d(
            self._table["declination"],
            self._table["cleanbeam_bmin"],
            fill_value="extrapolate",
        )
        interp_bpa = interp1d(
            self._table["declination"],
            self._table["cleanbeam_bpa"],
            fill_value="extrapolate",
        )

        beam_maj = interp_bmaj(dec) * u.deg
        beam_min = interp_bmin(dec) * u.deg
        beam_pa = interp_bpa(dec) * u.deg

        return BeamSize(
            beam_maj=beam_maj * self._reference_frequency / freq_centre,
            beam_min=beam_min * self._reference_frequency / freq_centre,
            beam_pa=beam_pa,
        )

    def get_weighting_factor(self, dec: Quantity) -> float:
        """
        method to calculate the beam weighting factor of the observation
        """
        # As of PI#15 we are no longer interpolating over frequency, elevation or integration time;
        # only declination. This has considerably simplified this method. If interpolation over more
        # than one parameter is required see commit 3640f432
        self._validate_declination(dec)

        interp_weighted = interp1d(
            self._table["declination"],
            self._table["pss_casa"],
            fill_value="extrapolate",
        )
        interp_natural = interp1d(
            self._natural_table["declination"],
            self._natural_table["pss_casa"],
            fill_value="extrapolate",
        )

        return interp_weighted(dec) / interp_natural(dec)

    def _validate_declination(self, dec: Quantity):
        # As of PI19, the weighting tables do not have values for every
        # allowed declination. It was decided to extrapolate the table
        # values within reasonable limits: to 1 degree of the maximum
        # calculated declination.
        extrapolation_tolerance = 1.0

        dec_min = self._table["declination"].min()
        dec_max = self._table["declination"].max() + extrapolation_tolerance

        # Raise RuntimeError if declination value is outside the allowed
        # extrapolation limits
        if dec.value < dec_min or dec.value > dec_max:
            raise ValueError(
                "Cannot extrapolate beyond declination limits of "
                f"{dec_min} <= dec < {dec_max}"
            )


class WeightingTableFactory:
    """
    Encapsulation of the beam weighting lookup table to be used for SKA MID and LOW weighting factor and
    beam size calculations.
    """

    def __init__(self, data_path=None):
        if not data_path:
            data_path = lookup_path
        mid_weighting_table = Table.read(data_path / "beam_weighting_mid.ecsv")
        mid_reference_frequency = mid_weighting_table.meta["reference_frequency"]

        low_weighting_table = Table.read(data_path / "beam_weighting_low.ecsv")
        low_reference_frequency = low_weighting_table.meta["reference_frequency"]

        # look up table has the subarrays stored according to their filename stem. creating
        # a new column that holds the subarray label data
        mid_sub_storage = SubarrayStorage(Telescope.MID)
        file_mapping = mid_sub_storage.filename_label_mapping()
        mid_weighting_table.add_column(
            [
                file_mapping[os.path.splitext(file_stem)[0]]
                for file_stem in mid_weighting_table["subarray"]
            ],
            name="subarray_label",
        )

        low_sub_storage = SubarrayStorage(Telescope.LOW)
        file_mapping = low_sub_storage.filename_label_mapping()
        low_weighting_table.add_column(
            [
                file_mapping[os.path.splitext(file_stem)[0]]
                for file_stem in low_weighting_table["subarray"]
            ],
            name="subarray_label",
        )

        self._weighting_tables = {
            Telescope.MID: mid_weighting_table,
            Telescope.LOW: low_weighting_table,
        }
        self._ref_frequencies = {
            Telescope.MID: mid_reference_frequency,
            Telescope.LOW: low_reference_frequency,
        }

        self._table = None
        self._natural_table = None
        self._reference_frequency = None

    def get_table(
        self,
        telescope: Telescope,
        array_config: Union[MIDArrayConfiguration, LOWArrayConfiguration],
        spectral_mode: WeightingSpectralMode,
        weighting_mode: Weighting,
        taper: Quantity = 0.0 * u.arcsec,
        robustness: int = 0,
    ):
        wtable = self._weighting_tables[telescope].copy()

        weighted_table = wtable[
            (
                (wtable["subarray_label"] == array_config.value)
                & (wtable["spec_mode"] == spectral_mode.value)
                & (wtable["weighting"] == weighting_mode.value)
                & (wtable["taper_arcsec"] == taper.value)
            )
        ]
        if weighting_mode == Weighting.ROBUST:
            weighted_table = weighted_table[
                (weighted_table["robustness"] == robustness)
            ]

        natural_table = wtable[
            (
                (wtable["subarray_label"] == array_config.value)
                & (wtable["spec_mode"] == spectral_mode.value)
                & (wtable["weighting"] == Weighting.NATURAL.value)
                & (wtable["taper_arcsec"] == 0.0)
            )
        ]

        return WeightingTable(
            weighting_table=weighted_table,
            natural_table=natural_table,
            reference_freq=self._ref_frequencies[telescope],
        )
