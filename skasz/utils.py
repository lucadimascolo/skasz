from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u

import numpy as np

def hdu_from_obs(obs,imsize,imcell):
    w = WCS(naxis=4)
    w.wcs.crpix = [1.00+0.50*imsize,
                   1.00+0.50*imsize,1,1]
    w.wcs.cdelt = np.array([-imcell.to(u.deg).value,
                             imcell.to(u.deg).value,
                             obs.frequency_increment_hz,
                             1])
    w.wcs.crval = [obs.phase_centre_ra_deg,
                   obs.phase_centre_dec_deg,
                   obs.start_frequency_hz+0.50*obs.frequency_increment_hz,
                   1]
    w.wcs.ctype = ['RA---SIN','DEC--SIN','FREQ','STOKES']
    w.wcs.radesys = 'ICRS'

    hdu = fits.PrimaryHDU(data=np.zeros((1,obs.number_of_channels,imsize,imsize)))
    hdu.header.update(w.to_header())

    hdu.header['CUNIT4'] = ''
    hdu.header['DATE-OBS'] = obs.start_date_and_time.isoformat()
    return fits.HDUList(hdu)