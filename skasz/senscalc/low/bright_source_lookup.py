from pathlib import Path

import numpy
from astropy import units as u
from astropy.constants import c as speed_of_light
from astropy.coordinates import SkyCoord

low_station_diameter = 38.0  # [m]


class BrightSourceCatalog:
    """
    Class to lookup bright sources relevant for SKA LOW senscalc
    """

    def __init__(self, catalog_file=None, threshold_jy=10.0):
        if catalog_file is None:
            catalog_file = (
                Path(__file__).resolve().parents[1] / "static/lookups/GLEAM_1Jy_cat.npy"
            )

        # Read in the bright source reference catalogue and select
        # sources brighter than threshold_jy
        bright_cat = numpy.load(catalog_file)
        threshold_mask = bright_cat[:, 2].astype(float) > threshold_jy
        selected_sources = bright_cat[threshold_mask]
        self._bright_source_coords = SkyCoord(
            selected_sources[:, 0],
            selected_sources[:, 1],
            unit=(u.hourangle, u.degree),
        )

    def check_for_bright_sources(
        self, pointing_centre: str, central_freq_MHz: float
    ) -> bool:
        """
        Check if there is at least one source brighter than threshold_jy
        within the FWHM of the primary beam at frequency specified by central_freq_MHz.

        :param pointing_centre: Source coordinate as a string
        :param central_freq_MHz: Frequency in MHz to estimate the FWHM of the primary beam
        :return: True | False
        """
        # Compute the FWHM of the primary beam at central_freq_hz
        # Note that this is a generic relation. This must be updated once Maciej's
        # beam models are available
        fwhm_deg = numpy.degrees(
            (speed_of_light.value / (central_freq_MHz * 1e6)) / low_station_diameter
        )

        # Compute the angular separation between the catalogue sources and the pointing centre
        pointing_centre_coord = SkyCoord(pointing_centre, unit=(u.hourangle, u.degree))
        separation = pointing_centre_coord.separation(self._bright_source_coords)

        return numpy.any(separation < (fwhm_deg / 2 * u.degree))
