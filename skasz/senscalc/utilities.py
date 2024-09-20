"""
Module holding functions that handle celestial emission, atmospheric behaviour and Telescope
Parameters
"""
import logging
from enum import Enum
from pathlib import Path

import astropy.constants as ac
import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation, Galactic
from astropy.io import fits
from astropy_healpix import HEALPix
from scipy.interpolate import RectBivariateSpline

logger = logging.getLogger("senscalc")

SKA_NDISHES_MAX = 133
SKA_DISH_DIAMETER = 15.0 * u.m
SKA_DISH_POINTING_ERROR = 10.0 * u.arcsec
MEERKAT_NDISHES_MAX = 64
MEERKAT_DISH_DIAMETER = 13.97 * u.m
MEERKAT_DISH_POINTING_ERROR = 10.0 * u.arcsec
STATION_N_MAX = 512

STATIC_DATA_PATH = Path(__file__).resolve().parent / "static"


class Telescope(Enum):
    """
    Enumeration for the different telescope types
    """

    MID = "mid"
    LOW = "low"


class DishType(Enum):
    """
    Enumeration for different dish types
    """

    MeerKAT = 0
    SKA1 = 1

    def __str__(self):
        return str(self.name)


class Celestial:
    """
    Class to handle Celestial emission
    """

    HASLAM_408 = (
        STATIC_DATA_PATH
        / "resources"
        / "haslam408_ds_Remazeilles2014_nonIDL_nside256.fits"
    )

    Tcmb = 2.73 * u.K

    def __init__(self):
        # open Haslam Healpix map and take a copy of the data

        hdul = fits.open(Celestial.HASLAM_408)
        self.hp = HEALPix(
            nside=hdul[1].header["NSIDE"],
            order=hdul[1].header["ORDERING"],
            frame=Galactic(),
        )
        self.hp_data = np.array(hdul[1].data.field(0))
        hdul.close()

    def _gaussian_beam(self, pixvals, offsets, fwhm):
        """Function to apply a Gaussian beam to a list of pixel
        values/offsets. The method assumes that all pixels have
        equal area.

        :param pixvals: the pixel values
        :type pixvals: scalar or astropy.units.Quantity
        :param offsets: pixel offsets from beam centre
        :type offsets: astropy.units.Quantity
        :param fwhm: the fwhm of the Gaussian
        :type fwhm: astropy.units.Quantity
        :return: the result at the beam centre of the convolution
                 of the pixels with the Gaussian
        :rtype: same unit as pixvals
        """

        sumdata = 0.0
        sumwt = 0.0
        c = fwhm.to(u.arcmin) / 2.355
        for i, v in enumerate(pixvals):
            wt = np.exp(-offsets[i].to(u.arcmin) ** 2 / (2 * c**2))
            sumdata += wt * v
            sumwt += wt
        return sumdata / sumwt

    def calculate_Tgal(
        self, target, obs_freq, dish_type, alpha
    ):  # pylint: disable=invalid-name
        """
        Calculate Galactic Temperature TODO: Make use of target direction.

        :param target: target direction
        :type target: astropy.coordinates.SkyCoord
        :param obs_freq: observing frequency
        :type obs_freq: astropy.units.Quantity
        :param dish_type: the type of dish
        :type dish_type: DishType
        :param alpha: spectral index of emission
        :type alpha: float

        :return: brightness temperature of Galactic emission in beam
        :rtype: astropy.units.Quantity
        """

        # Result assuming 50th percentile for Galactic contribution to Tsky.
        # This version passes current tests.
        # return u.K * 17.1 * (0.408 / obs_freq.to_value("GHz")) ** alpha

        # get Haslam map pixels within radius=beam_fwhm of target,
        # calculate pixel offsets from target

        fwhm = TelParams.dish_fwhm(obs_freq, dish_type)
        pixels = self.hp.cone_search_skycoord(target, radius=fwhm)
        pixvals = self.hp_data[pixels]
        skypixels = self.hp.healpix_to_skycoord(pixels)
        offsets = target.separation(skypixels)

        # convolve the pixels with the beam
        convolved_408 = self._gaussian_beam(pixvals, offsets, fwhm)
        result = (convolved_408 * (0.408 / obs_freq.to_value("GHz")) ** alpha) * u.K

        logger.debug(
            f"Celestial.calculate_Tgal({target}, {obs_freq}, {dish_type},"
            f" {alpha}) -> {result}"
        )
        return result


# Auxiliary atmospheric methods and tables
WEATHER_PWV = np.array([5, 10, 20])
T_ATM_PATH = STATIC_DATA_PATH / "lookups" / "T_atm.txt"
TAU_PATH = STATIC_DATA_PATH / "lookups" / "tau.txt"
T_atm_table = np.genfromtxt(T_ATM_PATH)
Tau_table = np.genfromtxt(TAU_PATH)
interp_atm = RectBivariateSpline(
    T_atm_table[:, 0], WEATHER_PWV, T_atm_table[:, 1:], ky=1
)
interp_tau = RectBivariateSpline(Tau_table[:, 0], WEATHER_PWV, Tau_table[:, 1:], ky=1)


class Atmosphere:
    """
    Class to handle atmospheric properties
    """

    @staticmethod
    def calculate_Tatm(weather, obs_freq):  # pylint: disable=invalid-name
        """
        Calculate Atmospheric Temperature

        :param weather: PWV in mm
        :type weather: float
        :param obs_freq: observing frequency
        :type obs_freq: astropy.units.Quantity
        :return: Brightness temperature of atmosphere
        :rtype: astropy.units.Quantity
        """
        freq_ghz = obs_freq.to_value("GHz")
        # Linear interpolate from lookup table values to find corresponding T_atm
        # for this frequency
        result = u.K * interp_atm(freq_ghz, weather).T[0]
        logger.debug(f"Atmosphere.calculate_Tatm({weather}, {obs_freq}) -> {result}")
        return result

    @staticmethod
    def get_tauz_atm(weather, obs_freq):
        """
        Calculate atmospheric optical depth at zenith

        :param weather: PWV in mm
        :type weather: float
        :param obs_freq: observing frequency
        :type obs_freq: astropy.units.Quantity
        :return: optical depth
        :rtype: float
        """
        freq_ghz = obs_freq.to_value("GHz")
        # Interpolate from lookup table to find zenith opacity (tau) based on observing frequency
        result = interp_tau(freq_ghz, weather).T[0]
        logger.debug(f"Atmosphere.get_tauz_atm({weather}, {obs_freq}) -> {result}")
        return result

    @staticmethod
    def tau_atm(weather, obs_freq, elevation):
        """
        Get atmospheric optical depth at given elevation.

        :param weather: "Good", "Average" or "Bad"
        :type weather: str
        :param obs_freq: observing frequency
        :type obs_freq: astropy.units.Quantity
        :param elevation: target elevation
        :type elevation: astropy.units.Quantity
        :return: optical depth
        :rtype: float
        """

        tauz = Atmosphere.get_tauz_atm(weather, obs_freq)
        zenith = 90.0 * u.deg - elevation
        tau = tauz / np.cos(zenith)
        tau[elevation <= 0.0 * u.deg] = -1.0

        logger.debug(f"Atmosphere.tau_atm({weather}, {obs_freq}, {elevation}) -> {tau}")
        return tau


class TelParams:
    """
    Class for handling Telescope Parameters
    """

    @staticmethod
    def calculate_Tspl(dish_type):  # pylint: disable=invalid-name
        """
        Calculate spillover temperature. This is signal from the ground that
        gets onto the detector. In reality it could depend on the
        alt/az pointing of the dish - for now it is assumed to be 3K for
        SKA1 and 4K for MeerKAT.

        :param dish_type: the type of dish
        :type dish_type: DishType
        :return: T spillover
        :rtype: astropy.units.Quantity
        """
        if dish_type is DishType.SKA1:
            result = 3.0 * u.K
        elif dish_type is DishType.MeerKAT:
            result = 4.0 * u.K
        else:
            raise RuntimeError("bad dish_type: %s" % dish_type)

        logger.debug(f"TelParams.calculate_Tspl({dish_type}) -> {result}")
        return result

    @staticmethod
    def calculate_Trcv(obs_freq, obs_band, dish_type):  # pylint: disable=invalid-name
        """
        Calculate Receiver Temperature. Works using obs freq only,
        not band. For SKA1 where bands overlap e.g. band 1, 2,
        returns the value reflecting the better performer.

        :param obs_freq: the observing frequency
        :type obs_freq: astropy.units.Quantity
        :param obs_band: the observing band ["Band 1", "Band 2", "Band 3", "Band 4", "Band 5a", "Band 5b"]
        :type obs_band: str
        :param dish_type: the type of dish
        :type dish_type: DishType
        :return: T receiver
        :rtype: astropy.units.Quantity
        """

        obs_freq_ghz = np.array(obs_freq.to_value("GHz"))

        # can't check following as MeerKAT does not use bands
        # if obs_band not in ["Band 1", "Band 2", "Band 3", "Band 4", "Band 5a", "Band 5b"]:
        #    raise RuntimeError("bad obs_band: %s" % obs_band)

        if dish_type is DishType.SKA1:
            if obs_band == "Band 1":
                if not np.all([obs_freq_ghz >= 0.35] and [obs_freq_ghz <= 1.05]):
                    raise RuntimeError(
                        "bad obs freq / band for SKA1: %s %s" % (obs_freq_ghz, obs_band)
                    )
                result = (15 + 30 * (obs_freq_ghz - 0.75) ** 2) * u.K
            elif obs_band in ["Band 2", "Band 3", "Band 4"]:
                if not np.all([obs_freq_ghz >= 0.95] and [obs_freq_ghz <= 5.18]):
                    raise RuntimeError(
                        "bad obs freq / band for SKA1: %s %s" % (obs_freq_ghz, obs_band)
                    )
                result = 7.5 * u.K * np.ones(obs_freq.shape)
            elif obs_band in ["Band 5a", "Band 5b"]:
                if not np.all([obs_freq_ghz >= 4.6] and [obs_freq_ghz <= 15.3]):
                    raise RuntimeError(
                        "bad obs freq / band for SKA1: %s %s" % (obs_freq_ghz, obs_band)
                    )
                result = (4.4 + 0.69 * obs_freq_ghz) * u.K
            else:
                raise RuntimeError("obs freq too high for SKA1: %s" % obs_freq_ghz)

        elif dish_type is DishType.MeerKAT:
            # commented out because SC does not properly account for different bands of MeerKAT
            # if obs_freq_ghz > 0.58 and obs_freq_ghz < 1.02:
            if np.all(obs_freq_ghz < 1.02):
                result = (11 - 4.5 * (obs_freq_ghz - 0.58)) * u.K
            elif np.all(obs_freq_ghz < 1.67):
                result = (7.5 + 6.8 * (abs(obs_freq_ghz - 1.65)) ** 1.5) * u.K
            else:
                result = 7.5 * u.K

        # commented out because SC does not properly account for different bands of MeerKAT
        #            elif obs_freq_ghz < 3.05:
        #                result = 7.5 * u.K
        #            else:
        #                raise RuntimeError('obs freq too high for MeerKAT: %s' % obs_freq.to(u.GHz))

        else:
            raise RuntimeError("bad dish_type: %s" % dish_type)

        logger.debug(
            f"TelParams.calculate_Trcv({obs_freq}, {obs_band}, {dish_type}) ->"
            f" {result}"
        )
        return result

    @staticmethod
    def dish_area(dish_type):
        """
        Calculate geometric area of a specified dish type.

        :param dish_type: the type of dish
        :type dish_type: DishType
        :return: the dish area
        :rtype: astropy.units.Quantity
        """
        if dish_type is DishType.SKA1:
            area = np.pi * (SKA_DISH_DIAMETER / 2) ** 2
        elif dish_type is DishType.MeerKAT:
            area = np.pi * (MEERKAT_DISH_DIAMETER / 2) ** 2
        else:
            raise RuntimeError("bad dish_type: %s" % dish_type)

        logger.debug(f"TelParams.dish_area({dish_type}) -> {area}")
        return area

    @staticmethod
    def calculate_dish_efficiency(obs_freq, dish_type):
        """
        Calculate aperture efficiency taking into account losses from
        feedhorn illumination of the aperture, phase errors at the dish
        surface, and diffraction.

        :param obs_freq: the observing frequency
        :type obs_freq: astropy.units.Quantity
        :param dish_type: the type of dish
        :type dish_type: DishType
        :return: dish aperture efficiency
        :rtype: astropy.units.Quantity
        """
        if dish_type is DishType.SKA1:
            # loss due to feedhorn illumination of the aperture
            eta_f = 0.92 - abs(0.04 * np.log10(obs_freq.to_value(u.GHz)))

            # Calculate observing wavelength
            wavelength = ac.c / obs_freq

            # Constants appropriate for design optics
            A_p = 0.89  # pylint: disable=invalid-name
            A_s = 0.98  # pylint: disable=invalid-name

            # Anticipated RMS surface errors of primary/secondary reflector surfaces
            eps_p = 280e-6 * u.m
            eps_s = 154e-6 * u.m

            # Path length error and corresponding phase error
            delta = 2 * ((A_p * eps_p**2) + (A_s * eps_s**2)) ** 0.5
            delta_ph = (2 * np.pi * delta) / wavelength

            # loss due to phase error
            eta_ph = np.exp(-(delta_ph**2))

            # loss due to diffraction
            eta_d = 1 - (20 * (wavelength / SKA_DISH_DIAMETER) ** 1.5)

            eta = eta_f * eta_ph * eta_d

        elif dish_type is DishType.MeerKAT:
            eta_f = 0.8 - abs(0.04 * np.log10(obs_freq.to_value(u.GHz)))

            wavelength = ac.c / obs_freq

            A_p = 0.89  # pylint: disable=invalid-name
            A_s = 0.98  # pylint: disable=invalid-name

            eps_p = 480e-6 * u.m
            eps_s = 265e-6 * u.m

            delta = 2 * ((A_p * eps_p**2) + (A_s * eps_s**2)) ** 0.5
            delta_ph = (2 * np.pi * delta) / wavelength
            eta_ph = np.exp(-(delta_ph**2))

            eta_d = 1 - 20 * (wavelength / MEERKAT_DISH_DIAMETER) ** 1.5

            eta = eta_f * eta_ph * eta_d
        else:
            raise RuntimeError("bad dish_type: %s" % dish_type)

        eta = eta.to_value(u.dimensionless_unscaled)

        logger.debug(
            f"TelParams.calculate_dish_efficiency({obs_freq}, {dish_type}) -> {eta}"
        )
        return eta

    @staticmethod
    def dish_fwhm(obs_freq, dish_type):
        """
        Calculate the full-width at half-maximum (FWHM) of the dish at
        the observing frequency.

        :param obs_freq: the observing frequency
        :type obs_freq: astropy.units.Quantity
        :param dish_type: the type of dish
        :type dish_type: DishType
        :return: the fwhm
        :rtype: astropy.units.Quantity
        """
        if dish_type is DishType.SKA1:
            fwhm = 66.0 * u.deg * (ac.c / obs_freq) / SKA_DISH_DIAMETER
        elif dish_type is DishType.MeerKAT:
            fwhm = 66.0 * u.deg * (ac.c / obs_freq) / MEERKAT_DISH_DIAMETER
        else:
            raise RuntimeError("bad dish_type: %s" % dish_type)

        logger.debug(f"TelParams.dish_fwhm({obs_freq}, {dish_type}) -> {fwhm}")
        return fwhm

    @staticmethod
    def mid_core_location():
        """
        Return the astropy EarthLocation of the SKA Mid core site. The data are
        taken from Wikipedia as astropy does not yet hold site information for
        the SKA.

        :return: the location of the SKA Mid core site
        :rtype: astropy.coordinates.EarthLocation
        """
        # astropy doesn't have data for the ska at the time of writing

        location = EarthLocation(
            lon="21d24m40.06s", lat="-30d43m16.068s", height=1000 * u.m
        )

        logger.debug(f"TelParams.mid_core_location() -> {location}")
        return location


ASTROPY_DEFAULT_UNITS = {u.s, u.Hz, u.m, u.Jy, u.K, u.deg}


class Utilities:
    """
    Class to contain generally useful methods
    """

    # Not currently used.
    @staticmethod
    def sensible_units(flux):
        """Function to set fluxes to sensible units, where 'sensible'
        means that the numerical value lies between 0.1 and 100

        :param flux: the flux to be converted
        :type flux: astropy.units.Quantity
        :return: the flux with value between 0.1 and 100 units
        :rtype: astropy.units.Quantity
        """
        if flux < 0.0 * u.Jy:
            raise RuntimeError("negative flux: %s" % flux)

        # iterate down the unit scale and stop when the associated
        # value is > 0.1

        input_flux = flux
        unit_range = ["MJy", "kJy", "Jy", "mJy", "uJy", "nJy"]
        for unit in unit_range:
            flux = flux.to(unit)
            if flux.value > 0.1:
                break

        logger.debug(f"Utilities.sensible_units({input_flux}) -> {flux}")
        return flux

    @staticmethod
    def to_astropy(quantity, unit):
        """Convert input to astropy quantity with SKA standard unit

        :param quantity: the value to be converted
        :type quantity: scalar or astropy.units.Quantity
        :param unit: the standard unit to use
        :type unit: astropy unit
        :return: the quantity in SKA standard units
        :rtype: astropy.units.Quantity
        """
        if unit not in ASTROPY_DEFAULT_UNITS:
            raise ValueError(
                "The unit must be one of the following default units:"
                f" {ASTROPY_DEFAULT_UNITS}"
            )

        if quantity is None:
            result = None
        elif isinstance(quantity, u.quantity.Quantity):
            result = quantity.to(unit)
        else:
            result = quantity * unit

        logger.debug(f"Utilities.to_astropy({quantity}, {unit}) -> {result}")
        return result

    @staticmethod
    def Tx(freq, T):  # pylint: disable=invalid-name
        """Function to apply correction to Rayleigh-Jeans temperature to
        describe the roll-off at high frequency of 'Johnson noise' in a
        resistor. Typically denoted by adding a subscript "x" to the
        temperature

        :param freq: the frequency
        :type freq: astropy.units.Quantity
        :param T: the temperature
        :type T: astropy.units.Quantity
        :return: the corrected temperature
        :rtype: astropy.units.Quantity
        """
        if np.any([freq < 0.0 * u.Hz]):
            raise RuntimeError("negative frequency: %s" % freq)
        if np.any([T < 0.0 * u.K]):
            raise RuntimeError("negative T: %s" % T)

        hvkT = (ac.h * freq) / (ac.k_B * T)  # pylint: disable=invalid-name
        result = T * (hvkT / (np.exp(hvkT) - 1))

        logger.debug(f"Utilities.Tx({freq}, {T}) -> {result}")
        return result
