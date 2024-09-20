"""
Module holding functions useful to the MidCalculator
"""
import logging

import astropy.units as u
import numpy as np

from skasz.senscalc.utilities import (
    MEERKAT_DISH_POINTING_ERROR,
    SKA_DISH_POINTING_ERROR,
    Atmosphere,
    Celestial,
    DishType,
    TelParams,
    Utilities,
)

logger = logging.getLogger("senscalc")

# pylint: disable=invalid-name

celestial = Celestial()


def eta_bandpass():
    """Efficiency factor for due to the departure of the bandpass from an ideal, rectangular
    shape. For now this is a placeholder.

    :return: the efficiency, eta
    :rtype: float
    """
    eta = 1.0
    logger.debug(f"eta_bandpass() -> {eta}")
    return eta


def eta_coherence(obs_freq):
    """Efficiency factor for the sensitivity degradation due to the
    incoherence on a baseline.

    :param obs_freq: the observing frequency
    :type obs_freq: astropy.units.Quantity
    :return: the efficiency, eta
    :rtype: float
    """
    eta = np.exp(np.log(0.98) * obs_freq.to_value(u.GHz) ** 2 / 15.4**2)
    logger.debug(f"eta_coherence({obs_freq}) -> {eta}")
    return eta


def eta_correlation():
    """Efficiency factor due to imperfection in the correlation algorithm, e.g. truncation error.

    :return: the efficiency, eta
    :rtype: float
    """
    eta = 0.98
    logger.debug(f"eta_correlation() -> {eta}")
    return eta


def eta_digitisation(obs_band):
    """Efficiency factor due to losses from quantisation during signal digitisation.
    This process is independent of the telescope and environment, but only depends on
    the 'effective number of bits' (ENOB) of the system, which depends in turn on
    digitiser quality and clock jitter, and on band flatness.

    :param obs_band: the observing band
    :type obs_band: str
    :return: the efficiency, eta
    :rtype: float
    """

    if obs_band == "Band 1":
        eta = 0.999
    elif obs_band == "Band 2":
        eta = 0.999
    elif obs_band == "Band 3":
        eta = 0.998
    elif obs_band == "Band 4":
        eta = 0.98
    elif obs_band == "Band 5a":
        eta = 0.955
    elif obs_band == "Band 5b":
        eta = 0.955
    else:
        raise RuntimeError("bad obs_band: %s" % obs_band)

    logger.debug(f"eta_digitisation() -> {eta}")
    return eta


def eta_dish(obs_freq, dish_type):
    """Efficiency factor due to losses for specified dish type.

    :param obs_freq: the observing frequency
    :type obs_freq: astropy.units.Quantity
    :param dish_type: the type of dish
    :type dish_type: DishType
    :return: the efficiency, eta
    :rtype: float
    """

    eta = TelParams.calculate_dish_efficiency(obs_freq, dish_type)
    logger.debug(f"eta_dish({obs_freq}, {dish_type}) -> {eta}")
    return eta


def eta_point(obs_freq, dish_type):
    """Efficiency factor at the observing frequency due to the dish pointing error.

    :param obs_freq: the observing frequency
    :type obs_freq: astropy.units.Quantity
    :param dish_type: the type of dish
    :type dish_type: DishType
    :return: the efficiency, eta
    :rtype: float
    """
    if dish_type is DishType.SKA1:
        pointing_error = SKA_DISH_POINTING_ERROR
    elif dish_type is DishType.MeerKAT:
        pointing_error = MEERKAT_DISH_POINTING_ERROR
    else:
        raise RuntimeError("bad dish_type: %s" % dish_type)

    eta = 1.0 / (
        1.0
        + (
            8
            * np.log(2)
            * (pointing_error / TelParams.dish_fwhm(obs_freq, dish_type)) ** 2
        )
    )
    eta = eta.to_value(u.dimensionless_unscaled)
    logger.debug(f"eta_point({obs_freq}, {dish_type}) -> {eta}")
    return eta


# Not currently used.
def eta_rfi():
    """Efficiency factor due to Radio Frequency Interference (RFI)

    :return: the efficiency, eta
    :rtype: float
    """
    eta = 1.00
    logger.debug(f"eta_rfi() -> {eta}")
    return eta


def eta_system(
    eta_point, eta_coherence, eta_digitisation, eta_correlation, eta_bandpass
):
    """System efficiency for SKA interferometer

    :param eta_point: efficiency loss due to pointing errors
    :type eta_point: float
    :param eta_coherence: efficiency due to loss of coherence
    :type eta_coherence: float
    :param eta_digitisation: efficiency loss due to errors in the digitisation process
    :type eta_digitisation: float
    :param eta_correlation: efficiency loss due to errors in the correlation process
    :type eta_correlation: float
    :param eta_bandpass: efficiency loss due to the bandpass not being rectangular
    :type eta_bandpass: float

    :return: the system efficiency
    :rtype: float
    """

    eta_system = (
        eta_point * eta_coherence * eta_digitisation * eta_correlation * eta_bandpass
    )
    logger.debug(
        f"eta_system({eta_point}, {eta_coherence}, {eta_digitisation},"
        f" {eta_correlation}, {eta_bandpass}) -> {eta_system}"
    )
    return eta_system


def Tgal(target, obs_freq, dish_type, alpha):
    """Brightness temperature of Galactic background in target direction at observing frequency.

    :param target: target direction
    :type target: astropy.SkyCoord
    :param obs_freq: the observing frequency
    :type obs_freq: astropy.units.Quantity
    :param dish_type: the type of dish
    :type dish_type: DishType
    :param alpha: spectral index of emission
    :type alpha: float
    :return: the brightness temperature of the Galactic background
    :rtype: astropy.units.Quantity
    """

    result = celestial.calculate_Tgal(target, obs_freq, dish_type, alpha)
    logger.debug(f"Tgal({target}, {obs_freq}, {dish_type}, {alpha}) -> {result}")
    return result


def Trcv(obs_freq, obs_band, dish_type):
    """Receiver temperature for specified freq, band and dish.

    :param obs_freq: the observing frequency
    :type obs_freq: astropy.units.Quantity
    :param obs_band: the observing band
    :type obs_band: str
    :param dish_type: the type of dish
    :type dish_type: DishType
    :return: the receiver temperature
    :rtype: astropy.units.Quantity
    """

    result = TelParams.calculate_Trcv(obs_freq, obs_band, dish_type)
    logger.debug(f"Trcv({obs_freq}, {obs_band}, {dish_type}) -> {result}")
    return result


def Tsky(Tgal, obs_freq, elevation, weather):
    """Brightness temperature of sky in target direction.

    :param Tgal: brightness temperature of Galactic background
    :type Tgal: astropy.units.Quantity
    :param obs_freq: the observing frequency
    :type obs_freq: astropy.units.Quantity
    :param elevation: the observing elevation
    :type elevation: astropy.units.Quantity
    :param weather: the atmosphere PWV
    :type weather: float
    :return: the brightness temperature of the sky
    :rtype: astropy.units.Quantity
    """

    # Tatm, Tcmb cannot be overriden by the user
    # Tatm at zenith
    Tatm_zen = Atmosphere.calculate_Tatm(weather, obs_freq)

    # Tcmb and Tgal are attenuated by the atmosphere,
    # Tatm also varies with zenith angle
    tau_zen = Atmosphere.get_tauz_atm(weather, obs_freq)

    zenith = 90.0 * u.deg - elevation
    tau = tau_zen / np.cos(zenith)

    # approximating Tatm ~ Tphys * (1 - exp(-tau))
    Tphys = Tatm_zen / (1 - np.exp(-tau_zen))

    result = Tphys * (1 - np.exp(-tau)) + (Tgal + celestial.Tcmb) * np.exp(-tau)
    logger.debug(f"Tsky({Tgal}, {obs_freq}, {elevation}, {weather}) -> {result}")
    return result


def Tspl(dish_type):
    """Spillover temperature for specified dish type.

    :param dish_type: the type of dish
    :type dish_type: DishType
    :return: the spillover temperature
    :rtype: astropy.units.Quantity
    """

    result = TelParams.calculate_Tspl(dish_type)
    logger.debug(f"Tspl() -> {result}")
    return result


def Tsys_dish(Trcv, Tspl, Tsky, obs_freq):
    """System temperature.

    :param Trcv: the receiver temperature
    :type Trcv: astropy.units.Quantity
    :param Tspl: the spillover temperature
    :type Tspl: astropy.units.Quantity
    :param Tsky: the sky temperature
    :type Tsky: astropy.units.Quantity
    :param obs_freq: the observing frequency
    :type obs_freq: astropy.units.Quantity
    :return: the dish system temperature
    :rtype: astropy.units.Quantity
    """

    # Add the temperatures to get the system temperature on the ground
    Tsys_ground = Trcv + Tspl + Tsky

    # Apply high-frequency correction to Rayleigh-Jeans temperature
    result = Utilities.Tx(obs_freq, Tsys_ground)
    result = result.to(u.K)
    logger.debug(f"Tsys_dish({Trcv}, {Tspl}, {Tsky}, {obs_freq}) -> {result}")

    return result
