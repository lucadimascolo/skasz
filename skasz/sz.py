from skasz.visibilities import Visibility
from astropy.coordinates import SkyCoord
from astropy import constants as const
from astropy import units as u

import numpy as np
import scipy.integrate

Tcmb = 2.7255*u.Kelvin

# Unit conversions
# ==============================================================================
# Adimensional frequency
# ------------------------------------------------------------------------------
def getx(freq):
  factor = const.h*freq*u.Hz/const.k_B/Tcmb
  return factor.to(u.dimensionless_unscaled).value

# CMB surface brightness
# ------------------------------------------------------------------------------
def getJynorm():
  factor  = 2e26
  factor *= (const.k_B*Tcmb)**3
  factor /= (const.h*const.c)**2
  return factor.value

# Compton y to Jy/pixel
# ------------------------------------------------------------------------------
def ytszToJyPix(freq,ipix,jpix):
  x = getx(freq)
  factor  = getJynorm()
  factor *= -4.00+x/np.tanh(0.50*x)
  factor *= (x**4)*np.exp(x)/(np.expm1(x)**2)
  factor *= np.abs((ipix.to(u.rad)*jpix.to(u.rad)).value)
  return factor


# Simulation 
# ==============================================================================
# Generate visibilities starting from a Compton y map
# ------------------------------------------------------------------------------
def comptontovis(hdu,obs,config='AA4',addnoise=False,**kwargs):
    vis = Visibility(config=config)
    vis.simulate(obs,**kwargs)

    hdu = vis.rascilhdu(hdu)

    def cfoo(x):
        bw = obs.frequency_increment_hz
        xpix = np.abs(hdu.header['CDELT1'])*u.deg
        ypix = np.abs(hdu.header['CDELT2'])*u.deg
        factor = scipy.integrate.quad(ytszToJyPix,x-0.50*bw,x+0.50*bw,args=(xpix,ypix))[0]
        return factor/2.00/bw
    
    conv = np.array([cfoo(f) for f in vis.frequency_channel_centers])
    hdu.data = hdu.data*conv[:,None,None,None]
    
    vis.addimage(hdu,**kwargs)
    
    if addnoise:
       vis.addnoise(**kwargs)

    return vis


