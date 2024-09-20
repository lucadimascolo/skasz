"""This module contains classes and methods for use in the SKA MID
Sensitivity Calculator.
"""
import logging

import astropy.constants as ac
import numpy as np

logger = logging.getLogger("senscalc")


def SEFD_antenna(Tsys, effective_dish_area):  # pylint: disable=invalid-name
    """Method to calculate the system equivalent flux density of an antenna

    :param Tsys: the system temperature for the dish
    :type Tsys: astropy.units.Quantity
    :param effective_dish_area: product of dish area and dish efficiency
    :type effective_dish_area: astropy.units.Quantity
    :return: the SEFD of the dish
    :rtype: astropy.units.Quantity
    """
    result = 2 * ac.k_B * Tsys / effective_dish_area
    logger.debug(f"SEFD_antenna({Tsys}, {effective_dish_area}) -> {result}")
    return result


def SEFD_array(n_type1, n_type2, sefd_dish_type1, sefd_dish_type2):
    # pylint: disable=invalid-name
    """Function to compute the system equivalent flux density of an heterogeneous
    array composed of two dish types.

    :param n_type1: the number of dishes of type 1
    :type n_type1: int
    :param n_type2: the number of dishes of type 2
    :type n_type2: int
    :param sefd_dish_type1: the dish SEFD for type 1
    :type sefd_dish_type1: astropy.units.Quantity
    :param sefd_dish_type2: the dish SEFD for type 2
    :type sefd_dish_type2: astropy.units.Quantity
    :return: the SEFD of the array
    :rtype: astropy.units.Quantity
    """
    result = 1 / np.sqrt(
        (n_type1 * (n_type1 - 1) / sefd_dish_type1**2)
        + (2 * n_type1 * n_type2 / (sefd_dish_type1 * sefd_dish_type2))
        + (n_type2 * (n_type2 - 1) / sefd_dish_type2**2)
    )
    logger.debug(
        f"SEFD_array({n_type1}, {n_type2}, {sefd_dish_type1},"
        f" {sefd_dish_type2} -> {result}"
    )
    return result
