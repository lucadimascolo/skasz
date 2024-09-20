import h5py
import numpy as np
from scipy.interpolate import UnivariateSpline

from skasz.senscalc.utilities import STATIC_DATA_PATH


def calculate_sefd_eff(sefd_array: np.array) -> float:
    """
    Calculate the effective SEFD of an observation, given an array of SEFD values

    Effective overall SEFD is weighted so that the radiometer equation holds:
    ```
    dS = SEFD_eff / sqrt(bw * t)
    ```
    Where bw is bandwidth, t is total integration time, and dS is noise level.

    If the mean SEFD value is used, this leads to incorrect estimates of observing times.

    To calculate SEFD_eff, we apply
    ```
    SEFD_eff = sqrt(N) / sqrt( sum( 1/(SEFD^2) ) )
    ```
    :param sefd_array: array of SEFD values. Data should have shape (time, frequency)
    The integration time for each entry must be the same, such that overall integration
    time is N = length of time axis.
    :return: sefd_eff a float corresponding to an effective SEFD value over the observation
    """
    s_eff = np.sqrt(np.sum(1 / (sefd_array**2), axis=0))
    N = sefd_array.shape[0]
    sefd_eff = np.sqrt(N) / s_eff
    return sefd_eff


class SEFDTable:
    """
    Encapsulation of the SEFD lookup table to be used for SKA LOW calculations
    """

    def __init__(self, db_path=None):
        if db_path is None:
            db_path = STATIC_DATA_PATH / "lookups/ska_station_sensitivity_AAVS2.h5"
        self._db_path = db_path

        # Load data from HDF5 into memory
        with h5py.File(db_path, "r") as h5:
            self.sefd = h5["sefd"][:]
            self.freqs = h5["dimensions"]["frequency"][:]
            self.az = h5["dimensions"]["azimuth"][:]
            self.za = h5["dimensions"]["zenith_angle"][:]
            self.pol = h5["dimensions"]["polarisation"][:]
            self.lst = h5["dimensions"]["lst"][:]

    def _select_data(
        self, az: float, el: float, start_lst: float, end_lst: float
    ) -> np.ndarray:
        """
        Helper function for lookup_stokes_i_sefd

        Makes data selection from self.sefd data array by finding indexes of closest data

        :param az: Azimuth angle in degrees (0-360 deg)
        :param el: Elevation angle in degrees (0-90 deg)
        :param start_lst: Start local sidereal time (LST) in hours (0-24 hrs)
        :param end_lst: End local sidereal time (LST) in hours (0-24 hrs)

        :return: data selection matching az/el/lst requirements
        """
        za = 90 - el

        # If near the zenith, we need to use az=za=0 entry
        if za <= 2.5:
            idx_za = 0
            idx_az = 0
        else:
            idx_az = np.argmin(np.abs(self.az - az))
            idx_za = np.argmin(np.abs(self.za - za))

        # If the observation crosses midnight LST (24), we need to wrap
        idxr_lst = np.zeros_like(self.lst, dtype=bool)  # Indexer array
        idx_lst_start = np.argmin(np.abs(self.lst - start_lst))
        idx_lst_end = np.argmin(np.abs(self.lst - end_lst))
        if idx_lst_start < idx_lst_end:
            idxr_lst[idx_lst_start:idx_lst_end] = True
        elif idx_lst_start == idx_lst_end:
            idxr_lst[idx_lst_start] = True
        else:
            idxr_lst[idx_lst_start:] = True
            idxr_lst[:idx_lst_end] = True

        # Apply indexing to select out data. We grab all frequencies and both pols
        dsel = self.sefd[idxr_lst, idx_az, idx_za, :, :]

        self._idxr_lst = idxr_lst
        self._idx_az = idx_az
        self._idx_el = idx_za

        return dsel

    def lookup_stokes_i_sefd(
        self,
        az: float,
        el: float,
        start_lst: float,
        end_lst: float,
        freq_fine_mhz: np.ndarray,
    ) -> np.ndarray:
        """
        Look up the SEFD values from the database corresponding to the local coordinates
        (az, el) between LST start_lst and end_lst.

        :param az: Azimuth angle in degrees (0-360 deg)
        :param el: Elevation angle in degrees (0-90 deg)
        :param start_lst: Start local sidereal time (LST) in hours (0-24 hrs)
        :param end_lst: End local sidereal time (LST) in hours (0-24 hrs)
        :param freq_fine_mhz: Frequency array, in MHz (50-350 MHz)

        :return: an array of Stokes I SEFD values at frequencies in fine_freq_mhz.
        """

        # Apply indexing to select out data. We grab all frequencies and both pols
        dsel = self._select_data(az, el, start_lst, end_lst)
        sefd_x_coarse = dsel[..., 0]
        sefd_y_coarse = dsel[..., 1]

        # Fit a cubic spline to coarse SEFD and interpolate to get
        # SEFD corresponding to the fine frequency grid.
        sefd_x_eff = calculate_sefd_eff(sefd_x_coarse)
        sefd_y_eff = calculate_sefd_eff(sefd_y_coarse)

        spline_x = UnivariateSpline(self.freqs, sefd_x_eff, s=0)
        spline_y = UnivariateSpline(self.freqs, sefd_y_eff, s=0)

        sefd_x_fine = spline_x(freq_fine_mhz)
        sefd_y_fine = spline_y(freq_fine_mhz)

        sefd_i_fine = 0.5 * np.sqrt(sefd_x_fine**2 + sefd_y_fine**2)
        return sefd_i_fine
