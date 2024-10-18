from skasz.visibilities import Visibility
from astropy.coordinates import SkyCoord
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from astropy.cosmology import Planck18

import numpy as np
import scipy.integrate

from numba import jit, vectorize, float32, float64

from skasz import utils

# Unit conversions
# ==============================================================================
Tcmb  = 2.7255*u.Kelvin
ynorm = (const.sigma_T/const.m_e/const.c**2).to(u.cm**2/u.keV)

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
def comptontovis(obs,hdu=None,imsize=None,imcell=None,config='AA4',addnoise=False,**kwargs):
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

# Pressure profiles
# ==============================================================================

# General radius class
# ------------------------------------------------------------------------------
class Radius:
    def __init__(self,r):
        self.r500 = r
        self.kpc = None
        self.deg = None
    
    def update(self,cosmo,z,m500=None,r500=None):
        if   (m500 is not None) and (r500 is None):
            self.kpc = m500/(4.00*np.pi/3.00)/500/cosmo.critical_density(z)
            self.kpc = np.cbrt(self.kpc).to(u.kpc)
        elif (m500 is None) and (r500 is not None):
            self.kpc = r500.kpc
        else: ValueError('Either m500 or r500 should be None')
        self.kpc = self.r500*self.kpc

        self.deg = self.kpc/cosmo.angular_diameter_distance(z).to(u.kpc)
        self.deg = self.deg.to(u.dimensionless_unscaled)*u.rad
        self.deg = self.deg.to(u.deg)

# General self-similar pressure model
# ------------------------------------------------------------------------------
class Pressure:
    def __init__(self,**kwargs):
        self.cosmo = kwargs.get('cosmo',Planck18)

        rmin = kwargs.get('rmin',1.00E-07)
        rmax = kwargs.get('rmax',5.00E+00)
        rnum = kwargs.get('rnum',100)

        self.rmax = Radius(rmax)
        self.rval = Radius(np.append(0.00,np.logspace(np.log10(rmin),np.log10(rmax),rnum-1)))

        self.r500 = Radius(1.00)
        
    def integrate(self):
        @jit(forceobj=True, cache=True)
        def _intarg(r,r0):
            return self.profile(r)*r/np.sqrt(r**2-r0**2)
        
        @vectorize([float32(float32, float32),
                    float64(float64, float64)], forceobj=True)
        def _intfoo(r,rmax):
            res, _ = scipy.integrate.quad(_intarg,r,rmax,args=(r,))
            return res*2.00

        rcut = self.rval.r500!=self.rmax.r500
        result = np.zeros_like(self.rval.r500)
        result[rcut] = _intfoo(self.rval.r500[rcut],self.rmax.r500)
        result = result*(u.keV/u.cm**3)*self.r500.kpc*ynorm
        return result.to(u.dimensionless_unscaled).value
    
    def __call__(self,**kwargs):
        if (self.z is None) and ('z' not in kwargs):
            ValueError('You should provide a valid redshift value')
        else: self.z = kwargs.get('z',self.z)
        
        if (self.m500 is None) and ('m500' not in kwargs):
            ValueError('You should provide a valid m500 value')
        else: self.m500 = kwargs.get('m500',self.m500).to(u.M_sun)

        self.r500.update(cosmo=self.cosmo,z=self.z,m500=self.m500)

        self.rmax.update(cosmo=self.cosmo,z=self.z,r500=self.r500)
        self.rval.update(cosmo=self.cosmo,z=self.z,r500=self.r500)

        self.yprof = self.integrate()

        mode = kwargs.get('mode','1D')
        if   mode=='1D': return self.yprof
        elif mode=='2D':            
            hdu = kwargs.get('hdu',None)
            if hdu is None:
                obs = kwargs.get('obs',None)
                imsize = kwargs.get('imsize',None)
                imcell = kwargs.get('imcell',None)
                if (obs    is not None) and \
                   (imsize is not None) and \
                   (imcell is not None):
                    hdu = utils.hdu_from_obs(obs,imsize,imcell)
                else:
                    ValueError('2D mode - you should provide a valid reference hdu or an (obs,imsize,imcell) set')
            
            xgrid, ygrid = utils.generateWCSMesh(hdu.header)

            xcval = kwargs.get('xc',hdu.header['CRVAL1']*u.deg)
            ycval = kwargs.get('yc',hdu.header['CRVAL2']*u.deg)

            rgrid = np.hypot((xgrid-xcval.to(u.deg).value)*np.cos(ycval.to(u.rad).value),
                              ygrid-ycval.to(u.deg).value)
            
            self.ygrid = np.interp(rgrid,self.rval.deg.value,self.yprof)
        
            hdu.data = self.ygrid.copy()
            hdu.header['UNITS'] = 'COMPTON Y'
            return hdu

# Arnaud+2010 models
# ------------------------------------------------------------------------------
class A10(Pressure):
    def __init__(self,model='up',**kwargs):
        super().__init__(**kwargs)
        self.model = f'A10_{model}'

        if   model=='up': 
            self.pars = dict(alpha = 1.0510E+00, beta = 5.4905E+00, gamma =  3.0810E-01, 
                             pnorm = 8.4030E+00, c500 = 1.1770E+00, ap    =  1.2000E-01)
            self.pars['pnorm'] = self.pars['pnorm']*((self.cosmo.H0.value/70.00)**(-3.00/2.00))
        elif model=='cc': 
            self.pars = dict(alpha = 1.2223E+00, beta = 5.4905E+00, gamma =  7.7360E-01,
                             pnorm = 3.2490E+00, c500 = 1.1280E+00, ap    = -1.0000E-01)
        elif model=='md':
            self.pars = dict(alpha = 1.4063E+00, beta = 5.4905E+00, gamma =  3.7980E-01,
                             pnorm = 3.2020E+00, c500 = 1.0830E+00, ap    = -1.0000E-01)

        self.pars['fb']  = kwargs.get('fb', 0.175)
        self.pars['mu']  = kwargs.get('mu', 0.590)
        self.pars['mue'] = kwargs.get('mue',1.140)

        self.m500 = kwargs.get('m500',None)
        self.z    = kwargs.get('z',None)

        if self.m500 is not None and self.z is not None:
            super().__call__(**kwargs)

    def profile(self,x):
        alpha = self.pars['alpha']
        beta  = self.pars['beta']
        gamma = self.pars['gamma']
        c500  = self.pars['c500']

        logm  = np.log10(self.m500.to(u.M_sun).value)
        Hz    = self.cosmo.H(self.z)

        p500  = (10**(self.pars['ap']*(logm-14.00)))
        p500 *= (3.00/8.00/np.pi)*(self.pars['fb']*self.pars['mu']/self.pars['mue'])
        p500 *= (((((2.5e2*Hz*Hz)**2.00)*((10**(logm-15.00))*u.solMass)/(const.G**(0.50)))**(2.00/3.00)).to(u.keV/u.cm**3)).value
        p500 *= 1e10

        factor1 = 1.00/(c500*x)**gamma

        factor2 = (1.00+((c500*x)**alpha))**((gamma-beta)/alpha)

        if self.pars['ap']!=-0.10: 
            factor3 = (self.m500.to(u.M_sun).value/3.00E+14)*(self.cosmo.H0.value/70.00)
            factor3 = factor3**((self.pars['ap']+0.10)/(1.00+(2.00*x)**3.00))
        else: factor3 = 1.00
        return self.pars['pnorm']*p500*factor1*factor2*factor3
