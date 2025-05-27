import matplotlib.pyplot as plt
from astropy.coordinates import AltAz, SkyCoord
from astropy.modeling import models, fitting
from astropy.time import Time
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame
from ska_sdp_datamodels.configuration import create_named_configuration
from ska_sdp_datamodels.visibility    import create_visibility
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent
from ska_sdp_datamodels.image.image_model import Image
from ska_sdp_datamodels.image.image_create import create_image

from ska_sdp_func_python.sky_component import apply_beam_to_skycomponent
from ska_sdp_func_python.imaging.dft import dft_kernel, extract_direction_and_flux
from ska_sdp_func_python.imaging import invert_ng, predict_ng, create_image_from_visibility, advise_wide_field, taper_visibility_gaussian
from ska_sdp_func_python.imaging.weighting import weight_visibility

from ska_sdp_func_python.image.deconvolution import fit_psf

from skasz.senscalc.mid.calculator import Calculator
from skasz.senscalc.utilities import TelParams
from skasz.senscalc.subarray import MIDArrayConfiguration
from skasz.senscalc.mid.sefd import SEFD_array

from skasz.senscalc.mid.validation import BAND_LIMITS

from ska_ost_array_config.array_config import MidSubArray, array_assembly

from radio_beam import Beam

from .utils import hdu_from_obs

AAstar15m = array_assembly.MID_AA2+","+array_assembly.MID_MKAT_PLUS

midsub = {'AA*': {'config': MidSubArray(subarray_type='AA*').array_config, 'antlist': ['MEERKAT','MID','HYBRID'], 'sub': MIDArrayConfiguration.MID_AASTAR_ALL},
          'AA4': {'config': MidSubArray(subarray_type='AA4').array_config, 'antlist': ['MEERKAT','MID','HYBRID'], 'sub': MIDArrayConfiguration.MID_AA4_ALL},
      'AA4_15m': {'config': MidSubArray(subarray_type='custom',custom_stations='SKA*').array_config, 'antlist': ['MID'], 'sub': MIDArrayConfiguration.MID_AA4_SKA_ONLY},
      'AA4_13m': {'config': MidSubArray(subarray_type='custom',custom_stations='M*').array_config, 'antlist': ['MEERKAT'], 'sub': MIDArrayConfiguration.MID_AA4_MEERKAT_ONLY},
      'AA*_15m': {'config': MidSubArray(subarray_type='custom',custom_stations=AAstar15m).array_config, 'antlist': ['MID'], 'sub': MIDArrayConfiguration.MID_AASTAR_SKA_ONLY}}

from rascil.processing_components import plot_uvcoverage
from rascil.processing_components.image.operations import polarisation_frame_from_wcs

try:
    from ska_sdp_func_python.imaging.primary_beams import create_vp_generic, set_pb_header
except:
    from rascil.processing_components.imaging.primary_beams import create_vp_generic, set_pb_header

from skasz.senscalc.mid.calculator import Calculator

from astropy.coordinates import ICRS
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs

import numpy as np
import scipy.stats

import warnings

polarisation_frame = PolarisationFrame('stokesI')

# Image constructor
# ------------------------------------------------------------------------------
class ImageSet:
    def __init__(self,dirty,psf,pb,beam_area):
        self.dirty = dirty
        self.psf   = psf
        self.pb    = pb

        self.beam_area = beam_area

# Main visibility builder
# ------------------------------------------------------------------------------
class Visibility:
    def __init__(self,config='AA4_15m',rmax=None,context='2d',obs=None,**kwargs):
        self.config = midsub[config]
    
        self.obs = None
        self.vis = None
        self.sumwt = None
        self.dirty = None

        self.pbref = None

        self.context = context

        if obs is not None: self.simulate(obs,**kwargs)

  # Simulate visibilities
  # ------------------------------------------------------------------------------  
    def simulate(self,obs,**kwargs):
        self.phasecentre = SkyCoord(ra = obs.phase_centre_ra_deg*u.deg,
                                   dec = obs.phase_centre_dec_deg*u.deg,
                                 frame = 'icrs')

        self.obs = obs

        observation_hour_angles = self.obs.compute_hour_angles_of_observation()
        self.integration_time_seconds = self.obs.length.total_seconds()/self.obs.number_of_time_steps
        
        frequency_channel_starts = np.linspace(self.obs.start_frequency_hz,
                                               self.obs.start_frequency_hz+self.obs.frequency_increment_hz*self.obs.number_of_channels,
                                               num=self.obs.number_of_channels,
                                               endpoint=False)

        frequency_bandwidths = np.full(frequency_channel_starts.shape,self.obs.frequency_increment_hz)
        self.frequency_channel_centers = frequency_channel_starts+frequency_bandwidths / 2

        self.vis = create_visibility(self.config['config'],
                               times = observation_hour_angles,
                           frequency = self.frequency_channel_centers,
                   channel_bandwidth = frequency_bandwidths,
	                     phasecentre = self.phasecentre, 
                              weight = kwargs.get('weight',1.00),
	                 elevation_limit = kwargs.get('elevation_limit',None),
                  polarisation_frame = polarisation_frame,
                    integration_time = self.integration_time_seconds)

        self.vis = self.vis.assign(_imaging_weight=self.vis.weight)

        advise = advise_wide_field(self.vis,guard_band_image=3.00,delA=0.1,facets=1, 
                                    oversampling_synthesised_beam=4.0)
        
        pbsize = kwargs.get('pbsize',1000)
        pbcell = kwargs.get('pbcell',advise['npixels']*advise['cellsize']/pbsize)

        pbeam = create_image(npixel = pbsize,
                           cellsize = pbcell,
                        phasecentre = self.phasecentre,
                          frequency = self.obs.start_frequency_hz,
                  channel_bandwidth = self.obs.frequency_increment_hz,
                              nchan = self.obs.number_of_channels,
                 polarisation_frame = polarisation_frame)

        self.pbref = {'MID': self.createpb(pbeam,    'MID',self.phasecentre,0.00,False),
                  'MEERKAT': self.createpb(pbeam,'MEERKAT',self.phasecentre,0.00,False),
                   'HYBRID': self.createpb(pbeam, 'HYBRID',self.phasecentre,0.00,False)}
        
        bllist = self.vis.baselines.data
        vplist = {id: self.config['config'].vp_type.data[i] for i, id in enumerate(self.config['config'].id.data)}

        self.vpmodel = {'nska': np.array([int(vplist[bl[0]]=='MID') + \
                                          int(vplist[bl[1]]=='MID') for bl in bllist]),
                       'nmeer': np.array([int(vplist[bl[0]]=='MEERKAT') + \
                                          int(vplist[bl[1]]=='MEERKAT') for bl in bllist]),
                          'id': np.empty(len(bllist),dtype=object)}
        
        for bi, bl in enumerate(bllist):
            if vplist[bl[0]]=='MID' and vplist[bl[1]]=='MID':
                self.vpmodel['id'][bi] = 'MID'
            elif vplist[bl[0]]=='MEERKAT' and vplist[bl[1]]=='MEERKAT':
                self.vpmodel['id'][bi] = 'MEERKAT'
            else:
                self.vpmodel['id'][bi] = 'HYBRID'
        
        self.vis = self.vis.assign_coords(array=self.vpmodel['id']); self.vpmodel.pop('id')

        if 'img' in kwargs:
            self.addimage(kwargs['img']['hdu'],**kwargs['img'].get('kwargs',{}))
        
        if 'pts' in kwargs:
            for pt in kwargs['pts']:
                self.addpoint(pt['direction'],pt['flux'],**pt.get('kwargs',{}))


  # Add noise
  # ------------------------------------------------------------------------------
    def addnoise(self,eltype='default',**kwargs):    
        midloc = TelParams.mid_core_location()

        rx_band = kwargs.get('rx_band',None)

        if rx_band is None:
            for b, band in enumerate(BAND_LIMITS.keys()):
                blims = BAND_LIMITS[band][-1]['limits']

                olims = [self.frequency_channel_centers.min()-0.50*self.obs.frequency_increment_hz,
                         self.frequency_channel_centers.max()+0.50*self.obs.frequency_increment_hz]

                if np.logical_and(olims[0]>=blims[0],olims[1]<=blims[1]): rx_band = band
        
        if rx_band is None:
            ValueError('No valid receiver band found. Please change the frequency range or specify a receiver band.')

        noise = np.empty_like(self.vis['vis'].data)

        if eltype=='transit':
            dt = np.linspace(0.00,24.00,1000)*u.h
            obstime = Time(self.obs.start_date_and_time.isoformat())+dt
            altaz = self.phasecentre.transform_to(AltAz(obstime=obstime,location=midloc))

            obstime = obstime[np.nanargmax(altaz.alt)]
            obstime = obstime-np.linspace(-0.50,0.50,self.obs.number_of_time_steps)*self.obs.length.total_seconds()*u.s
            altaz = self.phasecentre.transform_to(AltAz(obstime=obstime,location=midloc))

            ellist = altaz.alt

        for ti in range(self.vis['vis'].data.shape[0]):
            
            if eltype=='default':
                el = 45.00*u.deg
            elif eltype=='tune':
                obstime = Time(self.obs.start_date_and_time.isoformat())+(ti+0.50)*self.integration_time_seconds*u.s
                altaz = self.phasecentre.transform_to(AltAz(obstime=obstime,location=midloc))
                el = altaz.alt
            elif eltype=='transit':
                el = ellist[ti]

            for fi, freq in enumerate(self.frequency_channel_centers):  
                calc = Calculator(target = self.phasecentre,
                                      el = el,
                                 rx_band = rx_band,
                          freq_centre_hz = freq*u.Hz,
                            bandwidth_hz = self.obs.frequency_increment_hz*u.Hz,
                  subarray_configuration = self.config['sub'])

                sens = SEFD_array(self.vpmodel['nska'],self.vpmodel['nmeer'],calc.sefd_ska[0],calc.sefd_meer[0])
                sens = sens*np.exp(calc.tau)/(calc.eta_system*np.sqrt(2.00*calc.bandwidth*self.integration_time_seconds*u.s))
                sens = sens.to(u.Jy).value

                noise[ti,:,fi,0] = scipy.stats.norm.rvs(loc=0.00,scale=sens) + \
                                1j*scipy.stats.norm.rvs(loc=0.00,scale=sens)
        self.vis['vis'].data += noise
        
  # Add point source component
  # ------------------------------------------------------------------------------  
    def addpoint(self,direction,flux,alpha=-0.70,reference_frequency=1.40E+10,**kwargs):
        flux = flux*(self.frequency_channel_centers/reference_frequency)**alpha
        skycomp = SkyComponent(flux = flux[:,None],
                          direction = direction,
                          frequency = self.frequency_channel_centers,
                 polarisation_frame = polarisation_frame,
                              shape = 'Point',  
                             params = {})

        for ai, ant in enumerate(self.config['antlist']):
            comp = apply_beam_to_skycomponent(skycomp,self.pbref[ant])

            subvis = self.vis.sel(array=ant)
            direction_cosines, vfluxes = extract_direction_and_flux(comp,subvis)

            comp = dft_kernel(vfluxes = vfluxes,
                    direction_cosines = direction_cosines,
                           uvw_lambda = subvis.visibility_acc.uvw_lambda,
                   dft_compute_kernel = kwargs.get('dft_compute_kernel',None))

            self.vis['vis'].data[:,self.vis['array'].data==ant] += comp[:,self.vis['array'].data==ant]

  # Add extended source
  # Adapted from RASCIL: rascil.processing_components.image.operations.import_image_from_fits
  # ------------------------------------------------------------------------------
    def addimage(self,hdu,**kwargs):
        hdu_ = self.rascilhdu(hdu,**kwargs)
        wcs  = WCS(hdu_)
        
        data = hdu_.data.copy()

        if wcs.axis_type_names[3]=='STOKES' or wcs.axis_type_names[2] == 'FREQ':
            wcs = wcs.swapaxes(2,3)
            data = np.transpose(data,(1,0,2,3))
        try:
            pol_frame_wcs = polarisation_frame_from_wcs(wcs,data.shape)
            
            if kwargs.get('fixpol',True):
                permute = pol_frame_wcs.fits_to_datamodels[polarisation_frame.type]
                newdata = data.copy()
                for ip, p in enumerate(permute):
                    newdata[:, p, ...] = data[:, ip, ...]
                data = newdata.copy()
        except ValueError:
            pol_frame_wcs = polarisation_frame

        model = create_image(npixel = np.maximum(*data[0,0].shape),
                           cellsize = np.deg2rad(wcs.wcs.cdelt[0]),
                        phasecentre = self.phasecentre,
                          frequency = self.obs.start_frequency_hz,
                  channel_bandwidth = self.obs.frequency_increment_hz,
                              nchan = self.obs.number_of_channels,
                 polarisation_frame = polarisation_frame)
        
        for ai, ant in enumerate(self.config['antlist']):
            pb = self.createpb(model,ant,self.phasecentre,0.00,False)

            img = Image.constructor(data=data,polarisation_frame=pol_frame_wcs,wcs=wcs,clean_beam=None)
            img.pixels.data *= pb.pixels.data

            subvis = self.vis.sel(array=ant)
            subvis = predict_ng(subvis,img,context=self.context)
            self.vis['vis'].data[:,self.vis['array'].data==ant] += subvis['vis'].data[:,self.vis['array'].data==ant]
            del pb, img, subvis


  # Adapt the header for 2D images
  # ------------------------------------------------------------------------------
    def rascilhdu(self,hdu,**kwargs):
        
        if hdu.header['CTYPE1'] in ['RA---GLS','RA---NCP'] or \
           hdu.header['CTYPE2'] in ['DEC--GLS','DEC--NCP']:
            frame = ICRS()
            wcs, _ = find_optimal_celestial_wcs([hdu], frame=frame)

            hdu = fits.PrimaryHDU(data=hdu.data,header=wcs.to_header())

        header = hdu.header
        data   = hdu.data
        
        if  header['NAXIS'] not in [2,4]:
            raise ValueError('NAXIS value is neither 2 nor 4. Unknown format. Exiting')
        else:
            nc = self.frequency_channel_centers.shape[0]

            if  header['NAXIS']==2:
                header['NAXIS']   = 4
                header['WCSAXES'] = 4
                
                header['NAXIS3'] =  1; header['CDELT3'] = 1.00
                header['NAXIS4'] = nc; header['CDELT4'] = self.obs.frequency_increment_hz

                header['CTYPE3']  = 'STOKES'; header['CUNIT3']  =   ''; header['CRPIX3'] = 1.00; header['CRVAL3'] = 1.0
                header['CTYPE4']  =   'FREQ'; header['CUNIT4']  = 'Hz'; header['CRPIX4'] = 1.00; header['CRVAL4'] = self.frequency_channel_centers[0]

                header['RADESYS'] = 'ICRS'

                data = np.broadcast_to(data[None,None,...],(nc,1,*data.shape)).copy()
            elif header['NAXIS']==4 and header['NAXIS4']==1:
                data = np.broadcast_to(data,(nc,*data[0].shape)).copy()

                header['NAXIS4'] =     nc; header['CDELT4'] = self.obs.frequency_increment_hz
                header['CRPIX4'] =   1.00; header['CRVAL4'] = self.frequency_channel_centers[0]
                header['CTYPE4'] = 'FREQ'; header['CUNIT4'] = 'Hz'
            
            if header['CTYPE1']!='RA---SIN' or header['CTYPE2']!='DEC--SIN':                
                warnings.warn('Reprojecting image to [RA---SIN,DEC--SIN]')
                header_ = header.copy()
                header_['CTYPE1'] = 'RA---SIN'
                header_['CTYPE2'] = 'DEC--SIN'

                reproject = kwargs.get('reproject',reproject_interp)
                data, _ = reproject((data,header),header_)
                data[np.isnan(data)] = 0.00

                header = header_.copy()

            return fits.PrimaryHDU(data,header)


  # Image visibilities
  # ------------------------------------------------------------------------------
    def getimage(self,imsize,imcell=None,scale_factor=1.80,weighting='natural',computebeam=True,**kwargs):
        advice = advise_wide_field(self.vis,guard_band_image=3.0,delA=0.1,facets=1, 
		                           oversampling_synthesised_beam=4.0)
        if imcell is None: 
            print('Overriding imcell with wide-field advice: {0}'.format(advice['cellsize']))
            imcell = scale_factor*advice['cellsize']
        else:
            imcell = imcell.to(u.rad).value  

        model = create_image_from_visibility(self.vis,cellsize=imcell,npixel=imsize,
                                             override_cellsize=kwargs.get('override_cellsize',False))
 
        inpvis = self.vis
        inpvis = weight_visibility(inpvis,model,weighting=weighting,robustness=kwargs.get('robustness',1.00))

        taper = kwargs.get('taper',None)
        if taper is not None:
            inpvis = taper_visibility_gaussian(inpvis,taper.to(u.rad).value)

        self.dirty, self.sumwt = invert_ng(inpvis,model,context=self.context)
        self.psf,            _ = invert_ng(inpvis,model,context=self.context,dopsf=True)
      
      # Beam modelling
      # ----------------
        if computebeam:
            uvdist = np.hypot(self.vis.visibility_acc.uvw_lambda[...,0],
                              self.vis.visibility_acc.uvw_lambda[...,1])
            ngcell = kwargs.get('ngfact',0.25)/uvdist.max()
            ngsize = kwargs.get('ngsize',512)

            ngbeam = create_image_from_visibility(inpvis,cellsize=ngcell,npixel=ngsize,
                                                override_cellsize=kwargs.get('override_cellsize',False))
            ngbeam = invert_ng(inpvis,ngbeam,context=self.context,dopsf=True)[0]
            ngbeam, ngkern = getbeam(ngbeam.pixels.data[0,0])
        
            if ngkern>1:
                ngarea = 0.00
                ngnorm = 0.00
                for ni in range(ngkern):
                    factor = eval(f'ngbeam.x_stddev_{ni}') * \
                             eval(f'ngbeam.y_stddev_{ni}')
                    ngarea += eval(f'ngbeam.amplitude_{ni}')*2.00*np.pi*factor
                    ngnorm += eval(f'ngbeam.amplitude_{ni}')
                self.beam_area = ngarea/ngnorm
            else:
                self.beam_area  = 2.00*np.pi*ngbeam.x_stddev*ngbeam.y_stddev
            
            self.beam_area *= (ngcell*u.rad)**2
        else:
            self.beam_area = 0.00*u.sr

      # Compute avg. PB
      # ----------------

        self.pb = np.zeros(self.dirty.pixels.data.shape)
        norm_pb = 0.00
        for ai, ant in enumerate(self.config['antlist']):
            pb = self.createpb(model,ant,self.phasecentre,0.00,False)

            norm_ai  = np.count_nonzero(self.vis['array'].data==ant)
            norm_pb += norm_ai
            self.pb += pb.pixels.data.copy()*norm_ai
            del norm_ai

        self.pb = self.pb/norm_pb; del norm_pb

      # ----------------

        return ImageSet(dirty     = self.dirty.pixels.data.copy(),
                        psf       = self.psf.pixels.data.copy(),
                        pb        = self.pb.copy(),
                        beam_area = self.beam_area.to(u.sr))

  
  # Reset visibilities to zero
  # ------------------------------------------------------------------------------
    def reset(self):
        self.vis['vis'].data = np.zeros(self.vis['vis'].data.shape,dtype=self.vis['vis'].data.dtype)


  # Create PB model
  # ------------------------------------------------------------------------------
    def createpb(self,model,array='MID',pointingcentre=None,blockage=0.00,use_local=True):
        if array=='MID':
            beam = create_vp_generic(model,pointingcentre,use_local=False,diameter=15.00,blockage=0.0)
            beam['pixels'].data = np.real(beam['pixels'].values*np.conjugate(beam['pixels'].values))
        elif array=='MEERKAT':
            beam = create_vp_generic(model,pointingcentre,use_local=False,diameter=13.50,blockage=0.0)
            beam['pixels'].data = np.real(beam['pixels'].values*np.conjugate(beam['pixels'].values))
        elif array=='HYBRID':
            beam13   = create_vp_generic(model,pointingcentre,use_local=False,diameter=13.50,blockage=0.0)
            beam15 = create_vp_generic(model,pointingcentre,use_local=False,diameter=15.00,blockage=0.0)
            
            beam15['pixels'].data = np.real(beam15['pixels'].values*np.conjugate(beam15['pixels'].values))
            beam13['pixels'].data = np.real(beam13['pixels'].values*np.conjugate(beam13['pixels'].values))
            beam13['pixels'].data = np.sqrt(beam13['pixels'].data*beam15['pixels'].data)

            beam = beam13

        set_pb_header(beam, use_local=use_local)
        return beam


# Build a model for the synthesized beam
# ------------------------------------------------------------------------------
def getbeam(data):
    if True:
        model = models.Gaussian2D(x_mean = 0.50*data.shape[1],
                                  y_mean = 0.50*data.shape[0])
        
        model.theta.min = -0.50*np.pi
        model.theta.max =  0.50*np.pi

        model.x_stddev.min = 1.00E-03
        model.y_stddev.min = 1.00E-03
        model.amplitude.min = 0.00

    fitter = fitting.LevMarLSQFitter()

    dg = data.copy()
    yg, xg = np.mgrid[:dg.shape[0],:dg.shape[1]]

    mg_out = fitter(model,xg,yg,dg)
    mg_old = 0.00
    mg_res = data-mg_out(xg,yg)
    mg_mad = scipy.stats.median_abs_deviation(mg_res,scale='normal',axis=None)

    failed = False

    step = 1
    while np.max(mg_res.max())>5.00*mg_mad \
      and np.abs(mg_res.max()-mg_old)>0.01*mg_old \
      and not failed:
        mg_old = mg_res.max()
        try:
            mg_tmp = fitter(model,xg,yg,dg-mg_out(xg,yg))
            mg_tmp = mg_tmp+mg_out

            xtied = lambda model: model.x_mean_0
            ytied = lambda model: model.y_mean_0

            for si in range(1,step+1):
                mg_par_x = eval(f'mg_tmp.x_mean_{si}')
                mg_par_y = eval(f'mg_tmp.y_mean_{si}')
                
                mg_par_x.tied = xtied
                mg_par_y.tied = ytied
                
                setattr(mg_tmp,f'x_mean_{si}',mg_par_x)
                setattr(mg_tmp,f'y_mean_{si}',mg_par_y)

            mg_out = fitter(mg_tmp,xg,yg,data)

            mg_res = data-mg_out(xg,yg)
            mg_mad = scipy.stats.median_abs_deviation(mg_res,scale='normal',axis=None)

            step += 1
        except:
            failed = True

    return mg_out, step