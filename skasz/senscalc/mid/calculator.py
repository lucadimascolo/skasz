"""This module contains classes and methods for use in the SKA
Sensitivity Calculator.
It implements the loading of variables using properties and it
is meant to substitute the old class in the future.
"""
import astropy.units as u
import numpy as np

import skasz.senscalc.mid_utilities as mid_utils
from skasz.senscalc.mid.sefd import SEFD_antenna, SEFD_array
from skasz.senscalc.subarray import SubarrayStorage
from skasz.senscalc.utilities import (
    Atmosphere,
    DishType,
    Telescope,
    TelParams,
    Utilities,
)

subarray_storage = SubarrayStorage(Telescope.MID)

OVERRIDE_DOUBLE_ENTRIES = {
    "alpha": {"alpha_ska", "alpha_meer"},
    "target": {"target_ska", "target_meer"},
    "subarray_configuration": {
        "array_configuration_meer",
        "array_configuration_ska",
    },
}


def clean_overrides(overrides):
    """Function to provide a clean set of overrides in which duplicated overrides for a single
    parameter are translated into the name of the single parameter.
    For example, "alpha" corresponds to a simgle parameter that is returned only if it appears
    as "alpha_ska" and "alpha_meer" simultaneously in the original set of overrides.
    """
    output = overrides.copy()
    for parameter, entries in OVERRIDE_DOUBLE_ENTRIES.items():
        if not entries.difference(output):
            output = output.union(
                {parameter}
            )  # Adds the name of the parameter to the output set
        output = output.difference(
            entries
        )  # Removes the duplicate override flags always
    return output


class Calculator:
    """Class to calculate the sensitivity of the integration time"""

    # AT2-631: The variables eta and Temperatures can be updated after the creation
    #  of the object. This is working now but added a lot of complexity to the code.
    #  It is preferred to consider the object immutable once it is created. The setters
    #  of the variables should probably be removed and the cascade of computations moved
    #  to the __init__ function. This will clarify the code at the expense of some
    #  functionality that may not be needed at all.

    def __init__(
        self,
        *,  # Makes the constructor keyword-only to avoid mistakes
        rx_band=None,
        freq_centre_hz=None,
        target=None,
        bandwidth_hz=None,
        subarray_configuration=None,
        pwv=10.0,
        el=45 * u.deg,
        eta_system=None,
        eta_pointing=None,
        eta_coherence=None,
        eta_digitisation=None,
        eta_correlation=None,
        eta_bandpass=None,
        n_ska=None,
        eta_ska=None,
        t_sys_ska=None,
        t_spl_ska=None,
        t_rx_ska=None,
        n_meer=None,
        eta_meer=None,
        t_sys_meer=None,
        t_spl_meer=None,
        t_rx_meer=None,
        t_sky_ska=None,
        t_sky_meer=None,
        t_gal_ska=None,
        t_gal_meer=None,
        alpha=2.75,
    ):  # pylint: disable=invalid-name, too-many-arguments, too-many-locals
        # Set required values
        self._overriden = set()  # Set of overriden variables
        self._initialised = False  # Indicate that the input is being processed

        self._frequency = Utilities.to_astropy(freq_centre_hz, u.Hz)

        # TODO: Validate band and frequency match
        self._rx_band = rx_band

        # check target is visible at some time
        self.location = TelParams.mid_core_location()
        if target.icrs.dec > 90 * u.deg + self.location.lat:
            # target is never above the horizon
            raise RuntimeError("target always below horizon")
        self._target = target
        self._bandwidth = Utilities.to_astropy(bandwidth_hz, u.Hz)

        self._pwv = pwv
        # Check elevation compatible with target
        el = Utilities.to_astropy(el, u.deg)
        if (
            90.0 * u.deg - np.abs(self.location.to_geodetic().lat - target.icrs.dec)
            < el
        ):
            self._el = 90.0 * u.deg - np.abs(
                self.location.to_geodetic().lat - target.icrs.dec
            )
        else:
            self._el = el
        self._tau = Atmosphere.tau_atm(self._pwv, self._frequency, self._el)

        # Codependencies of eta_system
        if eta_pointing is not None:
            self._eta_pointing = eta_pointing
        else:
            self._eta_pointing = mid_utils.eta_point(self._frequency, DishType.SKA1)
        if eta_coherence is not None:
            self._eta_coherence = eta_coherence
        else:
            self._eta_coherence = mid_utils.eta_coherence(self._frequency)
        if eta_digitisation is not None:
            self._eta_digitisation = eta_digitisation
        else:
            self._eta_digitisation = mid_utils.eta_digitisation(self._rx_band)
        if eta_correlation is not None:
            self._eta_correlation = eta_correlation
        else:
            self._eta_correlation = mid_utils.eta_correlation()
        if eta_bandpass is not None:
            self._eta_bandpass = eta_bandpass
        else:
            self._eta_bandpass = mid_utils.eta_bandpass()

        # eta_system
        if eta_system is not None:
            # Manage overrides
            self._overriden = self._overriden.union(
                {
                    "eta_pointing",
                    "eta_coherence",
                    "eta_digitisation",
                    "eta_correlation",
                    "eta_bandpass",
                }
            )
            self._eta_system = eta_system
        else:
            self._overriden = self._overriden.difference(
                {
                    "eta_pointing",
                    "eta_coherence",
                    "eta_digitisation",
                    "eta_correlation",
                    "eta_bandpass",
                }
            )
            self._eta_system = mid_utils.eta_system(
                self._eta_pointing,
                self._eta_coherence,
                self._eta_digitisation,
                self._eta_correlation,
                self._eta_bandpass,
            )

        # eta dish
        if eta_ska is not None:
            self._eta_ska = eta_ska
        else:
            self._eta_ska = mid_utils.eta_dish(self._frequency, DishType.MeerKAT)
        if eta_meer is not None:
            self._eta_meer = eta_meer
        else:
            self._eta_meer = mid_utils.eta_dish(self._frequency, DishType.MeerKAT)

        # Validation has checked that either n_ska and n_meet are both set, or array_configuration is set
        # TODO does the Calculator even need to know the array_configuration? We could set nmeer/nska in the validation layer instead
        if subarray_configuration:
            self._array_configuration = subarray_storage.load_by_label(
                subarray_configuration.value
            )
            self._n_ska = self._array_configuration.n_ska
            self._n_ska_set = False
            # Manage overrides
            self._overriden = self._overriden.difference({"array_configuration_ska"})
            self._n_meer = self._array_configuration.n_meer
            self._n_meer_set = False
            # Manage overrides
            self._overriden = self._overriden.difference({"array_configuration_meer"})
        else:
            self._n_ska = n_ska
            self._n_ska_set = True
            # Manage overrides
            self._overriden = self._overriden.union({"array_configuration_ska"})
            self._n_meer = n_meer
            self._n_meer_set = True
            # Manage overrides
            self._overriden = self._overriden.union({"array_configuration_meer"})

        # alpha and Tgal
        self._alpha = alpha
        # *****************
        t_gal_ska_parsed = Utilities.to_astropy(t_gal_ska, u.K)
        if t_gal_ska_parsed is not None:
            self._t_gal_ska = t_gal_ska_parsed
            self._t_gal_ska_set = True
            # Manage overrides
            self._add_overrides_t_gal_ska()
        else:
            self._t_gal_ska = mid_utils.Tgal(
                self._target, self._frequency, DishType.SKA1, self._alpha
            )
            self._t_gal_ska_set = False
            # Manage overrides
            self._remove_overrides_t_gal_ska()
        t_gal_meer_parsed = Utilities.to_astropy(t_gal_meer, u.K)
        if t_gal_meer_parsed is not None:
            self._t_gal_meer = t_gal_meer_parsed
            self._t_gal_meer_set = True
            # Manage overrides
            self._add_overrides_t_gal_meer()
        else:
            self._t_gal_meer = mid_utils.Tgal(
                self._target, self._frequency, DishType.MeerKAT, self._alpha
            )
            self._t_gal_meer_set = False
            # Manage overrides
            self._remove_overrides_t_gal_meer()

        # Tsky
        t_sky_ska_parsed = Utilities.to_astropy(t_sky_ska, u.K)
        if t_sky_ska_parsed is not None:
            self._t_sky_ska = t_sky_ska_parsed
            self._t_sky_ska_set = True
            # Manage overrides.
            self._add_overrides_t_sky_ska()
        else:
            self._t_sky_ska = mid_utils.Tsky(
                self._t_gal_ska,
                self._frequency,
                self._el,
                self._pwv,
            )
            self._t_sky_ska_set = False
            # Manage overrides.
            self._remove_overrides_t_sky_ska()
        t_sky_meer_parsed = Utilities.to_astropy(t_sky_meer, u.K)
        if t_sky_meer_parsed is not None:
            self._t_sky_meer = t_sky_meer_parsed
            self._t_sky_meer_set = True
            # Manage overrides.
            self._add_overrides_t_sky_meer()
        else:  # Default value
            self._t_sky_meer = mid_utils.Tsky(
                self._t_gal_meer,
                self._frequency,
                self._el,
                self._pwv,
            )
            self._t_sky_meer_set = False
            # Manage overrides.
            self._remove_overrides_t_sky_meer()

        # Tspl
        t_spl_ska_parsed = Utilities.to_astropy(t_spl_ska, u.K)
        if t_spl_ska_parsed is not None:
            self._t_spl_ska = t_spl_ska_parsed
        else:
            self._t_spl_ska = mid_utils.Tspl(DishType.SKA1)
        t_spl_meer_parsed = Utilities.to_astropy(t_spl_meer, u.K)
        if t_spl_meer_parsed is not None:
            self._t_spl_meer = t_spl_meer_parsed
        else:
            self._t_spl_meer = mid_utils.Tspl(DishType.MeerKAT)

        # Treceiver
        t_rx_ska_parsed = Utilities.to_astropy(t_rx_ska, u.K)
        if t_rx_ska_parsed is not None:
            self._t_rx_ska = t_rx_ska_parsed
        else:
            self._t_rx_ska = mid_utils.Trcv(
                self._frequency, self._rx_band, DishType.SKA1
            )
        t_rx_meer_parsed = Utilities.to_astropy(t_rx_meer, u.K)
        if t_rx_meer_parsed is not None:
            self._t_rx_meer = t_rx_meer_parsed
        else:
            self._t_rx_meer = mid_utils.Trcv(
                self._frequency, self._rx_band, DishType.MeerKAT
            )

        # Tsys
        t_sys_ska_parsed = Utilities.to_astropy(t_sys_ska, u.K)
        if t_sys_ska_parsed is not None:
            self._t_sys_ska = t_sys_ska_parsed
            # Manage overrides.
            self._add_overrides_t_sys_ska()
        else:
            self._t_sys_ska = mid_utils.Tsys_dish(
                self._t_rx_ska,
                self._t_spl_ska,
                self._t_sky_ska,
                self._frequency,
            )
            # Manage overrides.
            self._remove_overrides_t_sys_ska()
        t_sys_meer_parsed = Utilities.to_astropy(t_sys_meer, u.K)
        if t_sys_meer_parsed is not None:
            self._t_sys_meer = t_sys_meer_parsed
            # Manage overrides.
            self._add_overrides_t_sys_meer()
        else:  # Default value
            self._t_sys_meer = mid_utils.Tsys_dish(
                self._t_rx_meer,
                self._t_spl_meer,
                self._t_sky_meer,
                self._frequency,
            )
            # Manage overrides.
            self._remove_overrides_t_sys_meer()

        # Compute SEFD
        self._sefd_ska = SEFD_antenna(
            self._t_sys_ska,
            self._eta_ska * TelParams.dish_area(DishType.SKA1),
        )
        self._sefd_meer = SEFD_antenna(
            self._t_sys_meer,
            self._eta_meer * TelParams.dish_area(DishType.MeerKAT),
        )
        self._sefd_array = SEFD_array(
            self._n_ska, self._n_meer, self.sefd_ska, self.sefd_meer
        )

    @property
    def overriden(self):
        return clean_overrides(self._overriden)

    @property
    def frequency(self):
        return self._frequency

    @property
    def rx_band(self):
        return self._rx_band

    @property
    def target(self):
        return self._target

    @property
    def bandwidth(self):
        return self._bandwidth

    @property
    def subarray_configuration(self):
        if hasattr(self, "_array_configuration"):
            return self._array_configuration.label
        return None

    @property
    def pwv(self):
        return self._pwv

    @property
    def el(self):
        return self._el

    @property
    def tau(self):
        return self._tau

    # eta
    # ..the contributors to eta_system
    # ..eta_point is forced to be same for SKA1 and MeerKAT as eta_system currently
    # ..covers both dish types

    @property
    def eta_pointing(self):
        return self._eta_pointing

    @property
    def eta_coherence(self):
        return self._eta_coherence

    @property
    def eta_digitisation(self):
        return self._eta_digitisation

    @property
    def eta_correlation(self):
        return self._eta_correlation

    @property
    def eta_bandpass(self):
        return self._eta_bandpass

    @property
    def eta_system(self):
        return self._eta_system

    @property
    def eta_ska(self):
        return self._eta_ska

    @property
    def eta_meer(self):
        return self._eta_meer

    @property
    def n_ska(self):
        return self._n_ska

    @property
    def n_meer(self):
        return self._n_meer

    # Temperatures
    @property
    def alpha(self):
        return self._alpha

    # T_gal
    @property
    def t_gal_ska(self):
        return self._t_gal_ska

    def _add_overrides_t_gal_ska(self):
        self._overriden = self._overriden.union({"alpha_ska", "target_ska"})

    def _remove_overrides_t_gal_ska(self):
        self._overriden = self._overriden.difference({"alpha_ska", "target_ska"})

    @property
    def t_gal_meer(self):
        return self._t_gal_meer

    def _add_overrides_t_gal_meer(self):
        self._overriden = self._overriden.union({"alpha_meer", "target_meer"})

    def _remove_overrides_t_gal_meer(self):
        self._overriden = self._overriden.difference({"alpha_meer", "target_meer"})

    # T_sky
    @property
    def t_sky_ska(self):
        return self._t_sky_ska

    def _add_overrides_t_sky_ska(self):
        self._overriden = self._overriden.union({"t_gal_ska"})
        # Cascade up overrides
        self._add_overrides_t_gal_ska()

    def _remove_overrides_t_sky_ska(self):
        self._overriden = self._overriden.difference({"t_gal_ska"})
        # Cascade up removal of overrides
        if not self._t_gal_ska_set:
            self._remove_overrides_t_gal_ska()

    @property
    def t_sky_meer(self):
        return self._t_sky_meer

    def _add_overrides_t_sky_meer(self):
        self._overriden = self._overriden.union({"t_gal_meer"})
        # Cascade up overrides
        self._add_overrides_t_gal_meer()

    def _remove_overrides_t_sky_meer(self):
        self._overriden = self._overriden.difference({"t_gal_meer"})
        # Cascade up removal of overrides
        if not self._t_gal_meer_set:
            self._remove_overrides_t_gal_meer()

    # T spl
    @property
    def t_spl_ska(self):
        return self._t_spl_ska

    @property
    def t_spl_meer(self):
        return self._t_spl_meer

    # T receiver
    @property
    def t_rx_ska(self):
        return self._t_rx_ska

    @property
    def t_rx_meer(self):
        return self._t_rx_meer

    # T_sys
    @property
    def t_sys_ska(self):
        return self._t_sys_ska

    def _add_overrides_t_sys_ska(self):
        self._overriden = self._overriden.union({"t_sky_ska", "t_spl_ska", "t_rx_ska"})
        # Cascade up overrides
        self._add_overrides_t_sky_ska()

    def _remove_overrides_t_sys_ska(self):
        self._overriden = self._overriden.difference(
            {"t_sky_ska", "t_spl_ska", "t_rx_ska"}
        )
        # Cascade up removal of overrides
        if not self._t_sky_ska_set:
            self._remove_overrides_t_sky_ska()

    @property
    def t_sys_meer(self):
        return self._t_sys_meer

    def _add_overrides_t_sys_meer(self):
        self._overriden = self._overriden.union(
            {"t_sky_meer", "t_spl_meer", "t_rx_meer"}
        )
        # Cascade up overrides
        self._add_overrides_t_sky_meer()

    def _remove_overrides_t_sys_meer(self):
        self._overriden = self._overriden.difference(
            {"t_sky_meer", "t_spl_meer", "t_rx_meer"}
        )
        # Cascade up removal of overrides
        if not self._t_sky_meer_set:
            self._remove_overrides_t_sky_meer()

    # SEFD
    @property
    def sefd_ska(self):
        return self._sefd_ska

    @property
    def sefd_meer(self):
        return self._sefd_meer

    @property
    def sefd_array(self):
        return self._sefd_array

    def calculate_sensitivity(self, integration_time):
        """Calculate sensitivity in Janskys for a specified integration time.

        :param integration_time: the integration time (in seconds or equivalent)
        :type integration_time: astropy.units.Quantity
        :return: the sensitivity of the telescope
        :rtype: astropy.units.Quantity
        """
        integration_time = Utilities.to_astropy(integration_time, u.s)
        # test that integration_time is a time > 0
        if integration_time.to_value(u.s) < 0.0:
            raise RuntimeError("negative integration time")

        sensitivity = self.sefd_array / (
            self.eta_system * np.sqrt(2 * self.bandwidth * integration_time)
        )
        # Now want to calculate Tsys in space so that we can compare with
        # astronomical fluxes. This means correcting Tsys for the attenuation
        # due to the atmosphere in the direction of the target.
        return sensitivity * np.exp(self.tau)

    def calculate_integration_time(self, sensitivity):
        """Calculate the integration time (in seconds) required to reach the specified sensitivity.

        :param sensitivity: the required sensitivity (in Jy or equivalent)
        :type sensitivity: astropy.units.Quantity
        :return: the integration time required
        :rtype: astropy.units.Quantity
        """
        sensitivity = Utilities.to_astropy(sensitivity, u.Jy)
        # test that sensitivity converts to Jy and is > 0
        if sensitivity.to_value(u.Jy) < 0.0:
            raise RuntimeError("negative sensitivity")

        integration_time = (
            np.exp(self.tau) * self.sefd_array / (sensitivity * self.eta_system)
        ) ** 2 / (2 * self.bandwidth)
        integration_time[self.el <= 0.0 * u.deg] = -1.0 * u.s

        return integration_time.to(u.s)

    def state(self):
        """
        extracts values that are either provided explicitly or calculated implicitly and
        pass them to the front end to populate the SC form.
        """
        var_names_not_astropy_quantities = (
            "pwv",
            "eta_system",
            "eta_pointing",
            "eta_coherence",
            "eta_digitisation",
            "eta_correlation",
            "eta_bandpass",
            "n_ska",
            "eta_ska",
            "n_meer",
            "eta_meer",
            "alpha",
        )
        var_names_astropy_quantities = (
            ("frequency", u.Hz),
            ("bandwidth", u.Hz),
            ("t_sys_ska", u.K),
            ("t_spl_ska", u.K),
            ("t_rx_ska", u.K),
            ("t_sys_meer", u.K),
            ("t_spl_meer", u.K),
            ("t_rx_meer", u.K),
            ("t_sky_ska", u.K),
            ("t_sky_meer", u.K),
            ("t_gal_ska", u.K),
            ("t_gal_meer", u.K),
            ("el", u.deg),
        )

        return_dict = {
            var_name: getattr(self, var_name)
            for var_name in var_names_not_astropy_quantities
        }
        return_dict.update(
            {
                var_name: getattr(self, var_name).value
                for var_name, var_unit in var_names_astropy_quantities
            }
        )
        ra_str = self._target.ra.to_string(unit=u.degree, sep=":")
        dec_str = self._target.dec.to_string(unit=u.degree, sep=":")
        return_dict.update(
            {
                "rx_band": self._rx_band,
                "target": f"{ra_str} {dec_str}",
            }
        )
        if self.array_configuration:
            return_dict.update({"subarray_configuration": self.array_configuration})
        else:
            return_dict.update({"n_ska": self.n_ska, "n_meer": self.n_meer})

        return return_dict
