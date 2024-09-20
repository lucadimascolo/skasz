import logging
from datetime import date, datetime, timedelta
from typing import List, Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from ephem import AlwaysUpError, FixedBody, NeverUpError, Observer

from skasz.senscalc.low.model import CalculatorInput, CalculatorResult
from skasz.senscalc.low.sefd_lookup import SEFDTable

LOGGER = logging.getLogger("senscalc")

# Longitude and Latitude are from the SKA1 LOW Configuration
#    Coordinates document (SKA-TEL-SKO-0000422 revision 04).
(LOW_LONGITUDE, LOW_LATITUDE, LOW_ELEVATION) = (
    "116.7644482",
    "-26.82472208",
    377.8,
)

# Create a module level table rather than instantiating for each calculation
sefd_table = SEFDTable()


def calculate_sensitivity(
    calculator_input: CalculatorInput,
) -> CalculatorResult:
    """
    For a given set of input parameters, estimate the Stokes I sensitivity for SKA LOW.

    We have a look up table which contains sensitivity for a coarse grid of
    frequency, LST, and (Az, El). For each LST time bin, this function

    1. Fetches the SEFD value for the corresponding (Az, El) and coarse frequency grid,
    2. Fits a smooth function along the frequency axis to interpolate
       SEFD on a fine frequency grid
    3. computes the Stokes I sensitivity in each time and frequency bin (fine).

    The computed Stokes I sensitivity are averaged to estimate the final Stokes I value.


    :param calculator_input: an instance of CalculatorInput with validated user parameters
    :return: a Calculator result containing the sensitivity and units
    """

    LOGGER.debug("Calculating LOW sensitivity for input: %s", calculator_input)
    ska_low = _get_skalow_object(calculator_input.elevation_limit)
    skalow_lon = np.degrees(float(ska_low.lon))
    skalow_lat = np.degrees(float(ska_low.lat))

    # Find the start and end time of the observation
    start_utc, end_utc, n_scans = _find_observation_time_utc(
        calculator_input.pointing_centre, calculator_input.integration_time_h, ska_low
    )

    # Make appropriate time and frequency grids.
    freq_fine_mhz, bandwidth_mhz = _get_frequency_grids(
        calculator_input.freq_centre_mhz, calculator_input.bandwidth_mhz
    )

    time_grid_utc = np.arange(start_utc, end_utc, step=timedelta(hours=0.5))
    # Note that the last cell in the time grid can be shorter than 0.5h.
    # We need to keep this in mind while estimating sensitivity in that cell.

    target = _get_target(calculator_input.pointing_centre)

    # Next, estimate the sensitivity in each time grid point
    n_time_elem = len(time_grid_utc)

    def calculate_sensitivity_for_timeslot(timeslot_id: int) -> float:
        LOGGER.debug(
            f"Processing time slot {timeslot_id}/{n_time_elem} -"
            f" {time_grid_utc[timeslot_id]}"
        )
        # Find the LST corresponding to this UTC
        start_lst = _utc_to_lst(time_grid_utc[timeslot_id], skalow_lon, skalow_lat)
        end_lst = (start_lst + 0.5) % 24

        # Determine the end time of this time cell in LST
        # The time resolution in the look-up table is 30 minutes
        # Note that the last time cell might be shorter than 0.5h
        # and must be treated as appropriate
        if timeslot_id == n_time_elem - 1:
            delta_t_sec = (
                end_utc - time_grid_utc[timeslot_id].astype(datetime)
            ).total_seconds()
        else:
            delta_t_sec = 0.5 * 3600.0

        # Convert the pointing RA, DEC to the horizontal coordinates (Az, El)
        # at this LST
        ska_low.date = time_grid_utc[timeslot_id].astype(datetime)
        target.compute(ska_low)
        az = np.degrees(float(target.az))
        el = np.degrees(float(target.alt))

        # Now that we have all the relevant boundary conditions for this time cell, query the DB
        # and get the SEFD values corresponding the the values in the fine frequency grid
        sefd_i_fine = sefd_table.lookup_stokes_i_sefd(
            az, el, start_lst, end_lst, freq_fine_mhz
        )
        # Calculate sensitivity from SEFD
        return _sefd_to_sensitivity(
            sefd_i_fine,
            calculator_input.num_stations,
            delta_t_sec,
            bandwidth_mhz,
        )

    LOGGER.debug("Calculating sensitivities for %s timeslots", n_time_elem)
    sensitivity_intervals = [
        calculate_sensitivity_for_timeslot(timeslot_id)
        for timeslot_id in range(n_time_elem)
    ]

    return _stack_sensitivities(sensitivity_intervals, n_time_elem, n_scans)


def _get_skalow_object(elevation_limit) -> Observer:
    """
    TODO: Elevation corresponds to MWA. Insert SKA LOW array centre elevation once known.
    :return: an ephem.Observer() object with details containing the SKA LOW array
    coordinates.
    """
    ska_low = Observer()
    ska_low.lon = LOW_LONGITUDE
    ska_low.lat = LOW_LATITUDE
    ska_low.elevation = LOW_ELEVATION
    # There are problems calculating the sensitivity for low-elevation sources -
    # this introduces a low-elevation limit of 15 degrees
    ska_low.horizon = str(elevation_limit)
    return ska_low


def _find_observation_time_utc(
    pointing: str, integration_time_h: float, observatory: Observer
) -> Tuple[datetime, datetime, float]:
    """
    For a given pointing, find a suitable time range when the source is visible, splitting into
    multiple scans if necessary.

    TODO:
    Allow a user-defined elevation limit.
    Accept pointing in SkyCoord() format.

    :return: (start_utc, end_utc, n_scans) where n_scans is the number of scans, each with length end_utc-start_utc
        in case the source is over the horizon for only a short period of time.
        Note that n_scans is a float.
    """
    target = _get_target(pointing)
    # Fix the date rather than using today to ensure consistency of results
    observatory.date = date.fromisoformat("2023-01-01")
    target.compute(observatory)

    try:
        next_rise = observatory.next_rising(target)
        next_transit = observatory.next_transit(target)
        next_set = observatory.next_setting(target)
        LOGGER.debug(
            "Source rise time: %s, Source transit time: %s, Source set time: %s",
            next_rise,
            next_transit,
            next_set,
        )
    except NeverUpError as err:
        raise ValueError(
            "Specified pointing centre is always below the horizon from the"
            " SKA LOW site"
        ) from err
    except AlwaysUpError:
        # Source is always up
        start_utc = target.transit_time.datetime() - timedelta(
            hours=integration_time_h / 2
        )
        end_utc = target.transit_time.datetime() + timedelta(
            hours=integration_time_h / 2
        )
        n_scans = 1
        LOGGER.debug(
            "Source always up. Start UTC: %s, End UTC: %s,  Num of scans: %s",
            start_utc,
            end_utc,
            n_scans,
        )
    else:
        visible_time_sec = (next_set.datetime() - next_rise.datetime()).total_seconds()
        if visible_time_sec < 0:
            previous_rise = observatory.previous_rising(target)
            visible_time_sec = (
                next_set.datetime() - previous_rise.datetime()
            ).total_seconds()

        if integration_time_h < visible_time_sec / 3600.0:
            # The user-specified duration can fit within a single scan
            start_utc = next_transit.datetime() - timedelta(
                hours=integration_time_h / 2
            )
            end_utc = next_transit.datetime() + timedelta(hours=integration_time_h / 2)
            n_scans = 1
            LOGGER.debug(
                "Duration fits within visible time of the source. Start"
                " UTC: %s, End UTC: %s,  Num of scans: %s",
                start_utc,
                end_utc,
                n_scans,
            )
        else:
            # The observation duration is longer than the source visibility time
            # Split integration_time_h into multiple scans with length less than visible_time_sec
            start_utc = next_transit.datetime() - timedelta(
                seconds=visible_time_sec / 2
            )
            end_utc = next_transit.datetime() + timedelta(seconds=visible_time_sec / 2)
            n_scans = integration_time_h * 3600.0 / visible_time_sec
            LOGGER.debug(
                "Duration longer than visible time of the source. Start"
                " UTC: %s, End UTC: %s,  Num of scans: %s",
                start_utc,
                end_utc,
                n_scans,
            )

    return start_utc, end_utc, n_scans


def _utc_to_lst(
    time_utc: datetime, obs_lon: np.ndarray[float], obs_lat: np.ndarray[float]
) -> float:
    """
    Converts UTC to Local Sidereal Time at the given location

    :param time_utc: the UTC time to convert
    :param obs_lat: the latitude of the geolocation
    :param obs_lon: the longitude of the geolocation

    :return: the corresponding LST time
    """
    time_utc = Time(time_utc, location=(obs_lon, obs_lat))
    return time_utc.sidereal_time("apparent").value


def _get_target(pointing_centre: str) -> FixedBody:
    """
    Create an ephem.FixedBody() object out of the specified pointing centre

    :param pointing_centre: eg 10:00:00 -30:00:00
    :return: a corresponding ephem.FixedBody()
    """
    point_coord = SkyCoord(pointing_centre, unit=(u.hourangle, u.deg))
    target = FixedBody()
    target._epoch = "2000."
    target._ra = point_coord.ra.radian
    target._dec = point_coord.dec.radian
    return target


def _get_frequency_grids(
    freq_centre: int, bandwidth: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create an incrementing frequency grid using the freq_centre and bandwidth

    :param freq_centre: the user defined input for the observation
    :param bandwidth: the user defined input for the observation
    :return: a 2 element tuple containing the frequency grid and a grid of ones of the same dimensions
    """
    freq_fine_mhz = np.arange(
        freq_centre - bandwidth / 2, freq_centre + bandwidth / 2, step=1
    )
    bandwidth_mhz = np.ones_like(freq_fine_mhz)
    # Note that the last frequency cell can be narrower than 1 MHz
    # Adjust the last frequency cell value
    bandwidth_mhz[-1] = (freq_centre + bandwidth / 2) - freq_fine_mhz[-1]

    return freq_fine_mhz, bandwidth_mhz


def _sefd_to_sensitivity(
    sefd: np.ndarray,
    num_stations: int,
    delta_t_sec: float,
    bandwidth_mhz: np.ndarray,
) -> float:
    """
    Compute Sensitivity from SEFD for a given number of antennas, observing time, and bandwidth.
    If SEFD is an array with multiple elements, return the stacked sensitivity.

    :param sefd: value of SEFD from the look up table
    :param num_stations: the user defined input for the observation
    :param delta_t_sec: the size of the time cell in seconds
    :param bandwidth_mhz: the array of 1s representing the bandwidth
    :return: the Stokes I sensitivity for this cell
    """

    sensitivity = sefd / np.sqrt(
        num_stations * (num_stations - 1) * delta_t_sec * bandwidth_mhz * 1e6
    )
    stacked_sensitivity = np.sqrt(np.sum(sensitivity**2)) / len(sefd)
    return stacked_sensitivity


def _stack_sensitivities(
    sensitivity_intervals: List[float], n_time_elem, n_scans
) -> CalculatorResult:
    """
    Scale the per-scan sensitivity to get the final sensitivity

    :param sensitivity_intervals:
    :param n_time_elem: The number of elements in the time grid
    :param n_scans: The number of scans the observation is split into
    :return: The result, in uJy/beam
    """
    sensitivity_intervals = np.asarray(sensitivity_intervals)
    weight = 1 / sensitivity_intervals**2
    sensitivity_per_scan = np.sqrt(
        np.sum((sensitivity_intervals * weight) ** 2)
    ) / np.sum(weight)

    final_sensitivity = sensitivity_per_scan / np.sqrt(n_scans)

    return CalculatorResult(final_sensitivity * 1e6, "uJy/beam")
