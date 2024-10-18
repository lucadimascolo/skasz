from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u

import numpy as np

# Generate hdu from obs object
# ------------------------------------------------------------------------------
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


# WCS 2D mesh grid
# ------------------------------------------------------------------------------
def generateWCSMesh(header,ext=False):
  headerWCS = header.copy()

  if ext:
    headerWCS['CRPIX1'] = headerWCS['CRPIX1']+int(headerWCS['NAXIS1']/2)
    headerWCS['CRPIX2'] = headerWCS['CRPIX2']+int(headerWCS['NAXIS2']/2)
    headerWCS['NAXIS1'] = headerWCS['NAXIS1']+2*int(headerWCS['NAXIS1']/2)
    headerWCS['NAXIS2'] = headerWCS['NAXIS2']+2*int(headerWCS['NAXIS2']/2)

  cdelt1 = headerWCS[list(filter(lambda x: x in headerWCS,['CDELT1','CD1_1']))[0]]
  cdelt2 = headerWCS[list(filter(lambda x: x in headerWCS,['CDELT2','CD2_2']))[0]]

  gridWCS = WCS(headerWCS).celestial

  gridmx, gridmy = np.meshgrid(np.arange(headerWCS['NAXIS1']),np.arange(headerWCS['NAXIS2']))
  gridwx, gridwy = gridWCS.all_pix2world(gridmx,gridmy,0)
  
  if (np.abs(gridwx.max()-gridwx.min()-3.6e2)<np.abs(2.0*cdelt1)): 
    gridix = np.where(gridwx>headerWCS['CRVAL1']+cdelt1*(headerWCS['NAXIS1']-headerWCS['CRPIX1']+1)+3.6e2)
    gridwx[gridix] = gridwx[gridix]-3.6e2
  return np.array([gridwx,gridwy])