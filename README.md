
# SKASZ
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31015/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![stability-alpha](https://img.shields.io/badge/stability-alpha-f4d03f.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#alpha)

A minimal library for generating mock SKA observations of the SZ effect. 

This package relies _heavily_ on RASCIL and the [SKA Science Data Processor](https://gitlab.com/ska-telescope/sdp) (SKA SDP) packages, and makes use of dedicated portings of parts of the [Karabo pipeline](https://github.com/i4Ds/Karabo-Pipeline) and of the python backend of the [SKA Sensitivity Calculator](https://gitlab.com/ska-telescope/ost/ska-ost-senscalc).

## Installation

> [!WARNING]
> This package was developed using Python 3.10, and there might be some incompatibilities with other versions. This is mostly due to some conflicts between the SKA packages, various external dependencies, and the wheels available for higher Python releases.

First of all, clone this repository on your local machine and move into the cloned directory:
```
git clone https://github.com/lucadimascolo/skasz.git
cd skasz
```

To build the simulation environment, you can use `conda` and install all the required dependencies using the embedded `environment.yml` file:
```
conda env create -f environment.yml [-n custom-env-name]
conda activate custom-env-name
```
If you don't pass any name when creating the environment, it will use the default name `skate` (SKA Testing Environment).

Once the installation is completed, you should be ready to go and fire `skasz` up.

## Notes
- When importing `skasz` for the first time, it will ask you whether you would like to download the [RASCIL data](https://gitlab.com/ska-telescope/external/rascil-main/-/tree/master/data?ref_type=heads) package. This is essential for running some simulations, and it shouldn't take too long. <br>
From time to time, it might happen that the `ska-telescope` servers results to be unreachable. In such a case, you can retrigger the download by re-loading `skasz` or by running `skasz.download_rascil_data()` from a python shell. <br>
For information on the RASCIL data package, you can take a look at the official [documentation page](https://rascil-main.readthedocs.io/en/latest/RASCIL_install.html#installation-via-pip) for RASCIL.
- When performing standard operations with `skasz`, you might get a bunch of future and deprecation warnings. This is due to some updates in the dependencies of RASCIL and of the SKA SDP packages. They should not introduce any issue with the simulation, and, as such, can be neglected.

## Example
This is a basic example of how to use `skasz` for generating mock SZ observations from an input FITS file, and to obtain noise-free and noisy dirty images.

```python
from datetime import datetime, timedelta

from astropy import units as u
from astropy.io import fits

# import input model
hdu = fits.open('input/SGRB_full_gt.fits')

# define the observation details
obs = Observation(start_frequency_hz = 1.20E+10,
                 start_date_and_time = datetime(2021,1,1,0,0,0),
                              length = timedelta(hours=10),
                  number_of_channels = 1,
              frequency_increment_hz = 2.00E+09,
                 phase_centre_ra_deg = hdu.header['CRVAL1']*u.deg
                phase_centre_dec_deg = hdu.header['CRVAL2']*u.deg,
                number_of_time_steps = 20)

# generate (noise-free) mock observations
vis = sz.comptontovis(hdu=hdu,obs=obs,config='AA4_15m')

# produce dirty image and psf for the simulated visibilities
dirty, psf = vis.getimage(imsize=hdu.header['NAXIS2'],imcell=hdu.header['CDELT2']*u.deg)

# add noise based on the observation specifications
vis.addnoise()
noisy, psf = vis.getimage(imsize=hdu.header['NAXIS2'],imcell=hdu.header['CDELT2']*u.deg)

```

## Future developments
- [ ] Efficient generation of simulated visibilities in the case heterogeneous arrays.
