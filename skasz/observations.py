"""
This script is ported from the Karabo-pipeline
https://github.com/i4Ds/Karabo-Pipeline
"""

import copy
from abc import ABC
from datetime import datetime, timedelta
from itertools import cycle
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from typing_extensions import TypeAlias
IntFloat: TypeAlias = Union[int,float]
    
class ObservationAbstract(ABC):
    """Abstract base class for observations

    Holds all important information about an observation.
    """

    def __init__(
        self,
        *,
        mode: str = "Tracking",
        start_frequency_hz: IntFloat = 0,
        start_date_and_time: Union[datetime, str],
        length: timedelta = timedelta(hours=4),
        number_of_channels: int = 1,
        frequency_increment_hz: IntFloat = 0,
        phase_centre_ra_deg: IntFloat = 0,
        phase_centre_dec_deg: IntFloat = 0,
        number_of_time_steps: int = 1,
        scan_time: Optional[timedelta] = None,
        bandwidth_hz: Optional[IntFloat] = None
    ) -> None:
        """

        Args:
            start_date_and_time (Union[datetime, str]): Start time UTC and date for
            the observation. Strings are converted to datetime objects
            using datetime.fromisoformat.

            mode (str, optional): TODO. Defaults to "Tracking".

            start_frequency_hz (IntFloat, optional): The frequency at the start of the
            first channel in Hz.
            Defaults to 0.

            length (timedelta, optional): Length of observation.
            Defaults to timedelta(hours=4).

            number_of_channels (int, optional): Number of channels / bands to use.
            Defaults to 1.

            frequency_increment_hz (IntFloat, optional): Frequency increment between
            successive channels in Hz.
            Defaults to 0.

            phase_centre_ra_deg (IntFloat, optional): Right Ascension of
            the observation pointing (phase centre) in degrees.
            Defaults to 0.

            phase_centre_dec_deg (IntFloat, optional): Declination of the observation
            pointing (phase centre) in degrees.
            Defaults to 0.

            number_of_time_steps (int, optional): Number of time steps in the output
            data during the observation length. This corresponds to the number of
            correlator dumps for interferometer simulations, and the number of beam
            pattern snapshots for beam pattern simulations.
            Defaults to 1.
        """

        if scan_time is not None:
            number_of_time_steps = int(length.total_seconds()/scan_time.total_seconds())
        if bandwidth_hz is not None:
            frequency_increment_hz = bandwidth_hz/number_of_channels
            
        self.start_frequency_hz = start_frequency_hz

        if isinstance(start_date_and_time, str):
            self.start_date_and_time = datetime.fromisoformat(start_date_and_time)
        else:
            self.start_date_and_time = start_date_and_time

        self.length = length
        self.mode = mode

        # optional
        self.number_of_channels = number_of_channels
        self.frequency_increment_hz = frequency_increment_hz
        self.phase_centre_ra_deg = phase_centre_ra_deg
        self.phase_centre_dec_deg = phase_centre_dec_deg
        self.number_of_time_steps = number_of_time_steps

    def set_length_of_observation(
        self,
        hours: IntFloat,
        minutes: IntFloat,
        seconds: IntFloat,
        milliseconds: IntFloat,
    ) -> None:
        """
        Set a new length for the observation.
        Overriding the observation length set in the constructor.

        :param hours: hours
        :param minutes: minutes
        :param seconds: seconds
        :param milliseconds: milliseconds
        """
        self.length = timedelta(
            hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds
        )

    def __strfdelta(
        self,
        tdelta: timedelta,
    ) -> str:
        hours = tdelta.seconds // 3600 + tdelta.days * 24
        rm = tdelta.seconds % 3600
        minutes = rm // 60
        seconds = rm % 60
        milliseconds = tdelta.microseconds // 1000
        return "{}:{}:{}:{}".format(hours, minutes, seconds, milliseconds)

    def get_phase_centre(self) -> List[float]:
        return [self.phase_centre_ra_deg, self.phase_centre_dec_deg]

    def compute_hour_angles_of_observation(self) -> NDArray[np.float_]:
        """
        Given a total observation length and an integration time interval,
        determine the corresponding hour angles of observation.
        This utility function is used during simulations using the RASCIL backend.
        Approach based on https://gitlab.com/ska-sdp-china/rascil/-/blob/9002d853b64465238177b37e941c7445fed50d35/examples/performance/mid_write_ms.py#L32-40 # noqa: E501
        """
        total_observation_length = self.length
        integration_time = timedelta(
            seconds=self.length.total_seconds() / self.number_of_time_steps
        )

        if self.number_of_time_steps == 1:
            # If both times are the same, we create one observation
            # at hour angle = 0 that lasts integration_time seconds
            hour_angles = np.array([0])
        else:
            hour_angles = np.arange(
                int(-0.5 * total_observation_length.total_seconds()),
                int(0.5 * total_observation_length.total_seconds()),
                int(integration_time.total_seconds()),
            ) * (2 * np.pi / timedelta(days=1).total_seconds())

        return hour_angles


class Observation(ObservationAbstract):
    ...


class ObservationLong(ObservationAbstract):
    """
    This class allows the use of several observations on different
    days over a certain period of time within one day.
    If only ONE observation is desired, even if it takes a little longer,
    this is already possible using `Observation`.
    This class extends `Observation` so its parameters (except `length`)
    are not discussed here.
    `length` is little different, which describes the duration of ONE observation,
    whose maximum duration for `ObservationLong` is 24h.

    :ivar number_of_days: Number of successive days to observe
    """

    def __init__(
        self,
        *,
        mode: str = "Tracking",
        start_frequency_hz: IntFloat = 0,
        start_date_and_time: Union[datetime, str],
        length: timedelta = timedelta(hours=4),
        number_of_channels: int = 1,
        frequency_increment_hz: IntFloat = 0,
        phase_centre_ra_deg: IntFloat = 0,
        phase_centre_dec_deg: IntFloat = 0,
        number_of_time_steps: int = 1,
        number_of_days: int = 2,
    ) -> None:
        self.enable_check = False
        super().__init__(
            mode=mode,
            start_frequency_hz=start_frequency_hz,
            start_date_and_time=start_date_and_time,
            length=length,
            number_of_channels=number_of_channels,
            frequency_increment_hz=frequency_increment_hz,
            phase_centre_ra_deg=phase_centre_ra_deg,
            phase_centre_dec_deg=phase_centre_dec_deg,
            number_of_time_steps=number_of_time_steps,
        )
        self.number_of_days: int = number_of_days
        self.__check_attrs()

    def __check_attrs(self) -> None:
        if not isinstance(self.number_of_days, int):
            raise TypeError(
                "`number_of_days` must be of type int but "
                + f"is of type {type(self.number_of_days)}!"
            )
        if self.number_of_days <= 1:
            raise ValueError(
                f"`number_of_days` must be >=2 but is {self.number_of_days}!"
            )
        if self.length > timedelta(hours=12):
            raise ValueError(f"`length` should be max 12 hours but is {self.length}!")


class ObservationParallelized(ObservationAbstract):
    """
    This class allows the use of several observations on different
    days over a certain period of time within one day.
    If only ONE observation is desired, even if it takes a little longer,
    this is already possible using `Observation`.
    This class extends `Observation` so its parameters (except `length`)
    are not discussed here.
    `length` is little different, which describes the duration of ONE observation,
    whose maximum duration for `ObservationLong` is 24h.

    :ivar number_of_days: Number of successive days to observe
    """

    def __init__(
        self,
        *,
        mode: str = "Tracking",
        center_frequencies_hz: Union[IntFloat, List[IntFloat]] = 100e6,
        start_date_and_time: Union[datetime, str],
        length: timedelta = timedelta(hours=4),
        n_channels: Union[int, List[int]] = [0, 1, 2, 3, 4, 5],
        channel_bandwidths_hz: Union[IntFloat, List[IntFloat]] = [1],
        phase_centre_ra_deg: IntFloat = 0,
        phase_centre_dec_deg: IntFloat = 0,
        number_of_time_steps: int = 1,
    ) -> None:
        self.enable_check = False
        super().__init__(
            mode=mode,
            start_frequency_hz=100e6,
            start_date_and_time=start_date_and_time,
            length=length,
            number_of_channels=1,
            frequency_increment_hz=0,
            phase_centre_ra_deg=phase_centre_ra_deg,
            phase_centre_dec_deg=phase_centre_dec_deg,
            number_of_time_steps=number_of_time_steps,
        )
        self.center_frequencies_hz = center_frequencies_hz
        self.n_channels = n_channels
        self.channel_bandwidths_hz = channel_bandwidths_hz