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

import time
import matplotlib.pyplot as plt

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
def comptontovis(obs,hdu=None,vis=None,config='AA4',addnoise=False,**kwargs):
    if vis is None:
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
    allowed_models = [{'A10': ['up','cc','md']},
                      {'M14': ['up','cc','nc']},
                      {'L15': ['ref','8.0','8.5']},
                      {'G17': ['ex','st']}]

    def __init__(self,model,**kwargs):
        self.cosmo = kwargs.get('cosmo',Planck18)

        rmin = kwargs.get('rmin',1.00E-07)
        rmax = kwargs.get('rmax',5.00E+00)
        rnum = kwargs.get('rnum',100)

        self.rmax = Radius(rmax)
        self.rval = Radius(np.append(0.00,np.logspace(np.log10(rmin),np.log10(rmax),rnum-1)))

        self.r500 = Radius(1.00)
        
        self.model = model
        
        self.pars['fb']  = kwargs.get('fb', 0.175)
        self.pars['mu']  = kwargs.get('mu', 0.590)
        self.pars['mue'] = kwargs.get('mue',1.140)

        self.m500 = kwargs.get('m500',None)
        self.z    = kwargs.get('z',None)

        if self.m500 is not None and self.z is not None:
            self.__call__(**kwargs)
            
    def integrate(self):
        @jit(forceobj=True,cache=True)
        def _intarg(r,r0):
            return self.profile(r)*r/(r**2-r0**2)**0.50
        
        @vectorize([float32(float32,float32,float32),
                    float64(float64,float64,float64)],forceobj=True)
        def _intfoo(r,rmax,eps):
            res, _ = scipy.integrate.quad(_intarg,r,rmax,args=(r,),epsrel=eps,epsabs=eps)
            return res*2.00
        
        rcut = self.rval.r500<self.rmax.r500
        result = np.zeros_like(self.rval.r500)
        result[rcut] = _intfoo(self.rval.r500[rcut],self.rmax.r500,1.00E-03)

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
            return fits.HDUList(hdu)

    def print_allowed_models(cls):
        for model in cls.allowed_models:
            for key in model:
                print(f'{key}: {model[key]}')

# Arnaud+2010 models
# ------------------------------------------------------------------------------
class A10(Pressure):
    def __init__(self,model='up',**kwargs):
        if   model=='up': 
            self.pars = dict(alpha = 1.0510E+00, beta = 5.4905E+00, gamma =  3.0810E-01, 
                             pnorm = 8.4030E+00, c500 = 1.1770E+00, ap    =  1.2000E-01)
        elif model=='cc': 
            self.pars = dict(alpha = 1.2223E+00, beta = 5.4905E+00, gamma =  7.7360E-01,
                             pnorm = 3.2490E+00, c500 = 1.1280E+00, ap    = -1.0000E-01)
        elif model=='md':
            self.pars = dict(alpha = 1.4063E+00, beta = 5.4905E+00, gamma =  3.7980E-01,
                             pnorm = 3.2020E+00, c500 = 1.0830E+00, ap    = -1.0000E-01)
            
        super().__init__(f'A10_{model}',**kwargs)

        if   model=='up': 
            self.pars['pnorm'] = self.pars['pnorm']/np.power(self.cosmo.H0.value/70.00,3.00/2.00)

    def profile(self,x):
        alpha = self.pars['alpha']
        beta  = self.pars['beta']
        gamma = self.pars['gamma']
        c500  = self.pars['c500']

        Hz    = self.cosmo.H(self.z)

        p500  = (3.00/8.00/np.pi)*self.pars['fb']*self.pars['mu']/self.pars['mue']
        p500 *= (2.50E+02*(Hz**2)/const.G**0.25)**(4.00/3.00)
        p500 *= (self.m500.to(u.M_sun)/1.00E+15)**(2.00/3.00)
        p500  = p500.to(u.keV/u.cm**3).value*1.00E+10

        factor1 = 1.00/(c500*x)**gamma
        factor2 = (1.00+((c500*x)**alpha))**((gamma-beta)/alpha)

        if self.pars['ap']!=-0.10:
            factor3 = (self.m500.to(u.M_sun).value/3.00E+14)*(self.cosmo.H0.value/70.00)
            factor3 = np.power(factor3,(self.pars['ap']+0.10)/(1.00+(2.00*x)**3.00))
        else:
            factor3 = 1.00
        return self.pars['pnorm']*p500*factor1*factor2*factor3

# Le Brun+2015
# ------------------------------------------------------------------------------
class L15(Pressure):
    def __init__(self,model='ref',**kwargs):
        if   model=='ref':
            self.pars = dict(alpha = 1.489, beta = 4.512, gamma = 1.174, delta = 0.072, epsilon = 0.245,
                             pnorm = 0.694, c500 = 0.986)
        elif model=='8.0':
            self.pars = dict(alpha = 1.517, beta = 4.625, gamma = 0.814, delta = 0.263, epsilon = 0.805,
                             pnorm = 0.791, c500 = 0.892)
        elif model=='8.5':
            self.pars = dict(alpha = 1.572, beta = 4.850, gamma = 0.920, delta = 0.246, epsilon = 0.864,
                             pnorm = 0.235, c500 = 0.597)

        super().__init__(f'L15_{model}',**kwargs)

    def profile(self,x):
        alpha   = self.pars['alpha']
        beta    = self.pars['beta']
        gamma   = self.pars['gamma']
        delta   = self.pars['delta']
        epsilon = self.pars['epsilon']

        c500  = self.pars['c500']*(self.m500.to(u.M_sun).value/1.00E+14)**delta
        pnorm = self.pars['pnorm']*(self.m500.to(u.M_sun).value/1.00E+14)**delta

        Hz    = self.cosmo.H(self.z)

        p500  = (3.00/8.00/np.pi)*self.pars['fb']*self.pars['mu']/self.pars['mue']
        p500 *= (2.50E+02*(Hz**2)/const.G**0.25)**(4.00/3.00)
        p500 *= (self.m500.to(u.M_sun)/1.00E+15)**(2.00/3.00)
        p500  = p500.to(u.keV/u.cm**3).value*1.00E+10

        factor1 = 1.00/(c500*x)**gamma
        factor2 = (1.00+((c500*x)**alpha))**((gamma-beta)/alpha)

        return pnorm*p500*factor1*factor2

# Gupta+2017
# ------------------------------------------------------------------------------
class G17(Pressure):
    def __init__(self,model='ex',**kwargs):
        if model=='ex':
            self.pars = dict(beta0 = 4.770, gamma0 =  0.502, alpha = 1.3300, ap = -0.0510,
                             beta1 = 0.056, gamma1 = -0.050, pnorm = 0.1716, cp = -0.3210,
                             beta2 = 0.254, gamma2 = -0.710, c500  = 1.2700)
        elif model=='st':
            self.pars = dict(beta0 = 5.060, gamma0 =  0.370, alpha = 1.2300, ap =  0.0105,
                             beta1 = 0.000, gamma1 =  0.000, pnorm = 0.1701, cp = -0.1210,
                             beta2 = 0.000, gamma2 =  0.000, c500  = 1.2100)

        super().__init__(f'G17_{model}',**kwargs)

    def profile(self,x):
        alpha   = self.pars['alpha']
        
        Ez = (self.cosmo.H(self.z)/self.cosmo.H0).to(u.dimensionless_unscaled).value
        mass  = self.mass.to(u.M_sun).value/3.00E+14

        beta  = self.pars['beta0']*(mass**self.pars['beta1'])*(Ez**self.pars['beta2'])
        gamma = self.pars['gamma0']*(mass**self.pars['gamma1'])*(Ez**self.pars['gamma2'])

        c500 = self.pars['c500']

        p500  = 1.65E-03
        p500 *= mass**(2.00/3.00+self.pars['ap'])
        p500 *= Ez**(8.00/3.00+self.pars['cp'])
        
        factor1 = c500**gamma
        factor1 = factor1/(c500*x)**gamma

        factor2 = (1.00+c500**alpha)**((beta-gamma)/alpha)
        factor2 = factor2/(1.00+(c500*x)**alpha)**((beta-gamma)/alpha)
    
        return self.pars['pnorm']*p500*factor1*factor2
    

# General self-similar model
# ------------------------------------------------------------------------------
class GSS(Pressure):
    def __init__(self,model,**kwargs):
        super().__init__(model,**kwargs)

    def profile(self,x):
        alpha = self.pars['alpha']
        beta  = self.pars['beta']
        gamma = self.pars['gamma']
        c500  = self.pars['c500']

        Hz    = self.cosmo.H(self.z)

        p500  = (3.00/8.00/np.pi)*self.pars['fb']*self.pars['mu']/self.pars['mue']
        p500 *= (2.50E+02*(Hz**2)/const.G**0.25)**(4.00/3.00)
        p500 *= (self.m500.to(u.M_sun)/1.00E+15)**(2.00/3.00)
        p500  = p500.to(u.keV/u.cm**3).value*1.00E+10

        factor1 = 1.00/(c500*x)**gamma
        factor2 = (1.00+((c500*x)**alpha))**((gamma-beta)/alpha)

        factor3 = (self.m500.to(u.M_sun).value/3.00E+14)*(self.cosmo.H0.value/70.00)
        factor3 = factor3**self.pars['ap']
        
        return self.pars['pnorm']*p500*factor1*factor2*factor3


# McDonald+2014 models
# ------------------------------------------------------------------------------
class M14(GSS):
    def __init__(self,model='up',**kwargs):
        if   model=='up': # McDonald+2014 universal
            self.pars = dict(alpha = 2.2700E+00, beta = 3.4800E+00, gamma =  1.5000E-01,
                             pnorm = 3.4700E+00, c500 = 2.5900E+00, ap    =  1.2000E-01)
        elif model=='cc': # McDonald+2014 cool core
            self.pars = dict(alpha = 2.3000E+00, beta = 3.3400E+00, gamma =  2.1000E-01,
                             pnorm = 3.7000E+00, c500 = 2.8000E+00, ap    =  1.2000E-01)
        elif model=='nc': # McDonald+2014 non-cool core
            self.pars = dict(alpha = 1.7000E+00, beta = 5.7400E+00, gamma =  0.5000E-01,
                             pnorm = 3.9100E+00, c500 = 1.5000E+00, ap    =  1.2000E-01)

        super().__init__(f'M14_{model}',**kwargs)

# Melin+2023
# ------------------------------------------------------------------------------
class M23(GSS):
    def __init__(self,model='up',**kwargs):
        if model=='up':
            self.pars = dict(alpha = 1.0500E+00, beta = 6.3200E+00, gamma =  7.1000E-01,
                             pnorm =   10**0.23, c500 = 6.1000E-01,    ap =  1.2000E-01)
        super().__init__(f'M23_{model}',**kwargs)

# Sayers+2023
# ------------------------------------------------------------------------------
class S23(Pressure):
    def __init__(self,model='up',**kwargs):
        if model=='up':
            self.pars = dict(alpha0 =  1.20E-01, beta0 = 7.40E-01, pnorm0 =  7.40E-01,
                             alpha1 =  1.20E-01, beta1 = 1.50E-01, pnorm1 = -2.70E-01,
                             alpha2 = -4.10E-01, beta2 = 2.00E-02, pnorm2 =  2.10E+00,
                             c500   =  1.40E+00, gamma = 3.00E-01)
            
        super().__init__(f'S23_{model}',**kwargs)
    
    def profile(self,x):
        alpha = np.power(10,self.pars['alpha0']+ \
                            self.pars['alpha1']*np.log10(self.m500.to(u.M_sun).value/1.00E+15)+ \
                            self.pars['alpha2']*np.log10(1.00+self.z))
        beta  = np.power(10,self.pars['beta0']+ \
                            self.pars['beta1']*np.log10(self.m500.to(u.M_sun).value/1.00E+15)+ \
                            self.pars['beta2']*np.log10(1.00+self.z))
        gamma = self.pars['gamma']

        pnorm = np.power(10,self.pars['pnorm0']+ \
                            self.pars['pnorm1']*np.log10(self.m500.to(u.M_sun).value/1.00E+15)+ \
                            self.pars['pnorm2']*np.log10(1.00+self.z))

        c500  = self.pars['c500']

        factor1 = 1.00/(c500*x)**gamma
        factor2 = (1.00+((c500*x)**alpha))**((gamma-beta)/alpha)

        return self.pars['pnorm']*factor1*factor2