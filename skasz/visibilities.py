import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
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
from ska_sdp_func_python.imaging import invert_ng, predict_ng, create_image_from_visibility, advise_wide_field 

from skasz.senscalc.mid.calculator import Calculator
from skasz.senscalc.subarray import MIDArrayConfiguration
from skasz.senscalc.mid.sefd import SEFD_array

from ska_ost_array_config.array_config import MidSubArray, SubArray

midsub = {'AA*': {'config': MidSubArray(subarray_type='AA*').array_config, 'sub': MIDArrayConfiguration.MID_AASTAR_ALL},
          'AA4': {'config': MidSubArray(subarray_type='AA4').array_config, 'sub': MIDArrayConfiguration.MID_AA4_ALL},
      'AA4_15m': {'config': MidSubArray(subarray_type='custom',custom_stations='SKA*').array_config, 'sub': MIDArrayConfiguration.MID_AA4_SKA_ONLY},
      'AA4_13m': {'config': MidSubArray(subarray_type='custom',custom_stations='SKA*').array_config, 'sub': MIDArrayConfiguration.MID_AA4_MEERKAT_ONLY},
      'AA*_15m': {'config': MidSubArray(subarray_type='custom',custom_stations='SKA*').array_config, 'sub': MIDArrayConfiguration.MID_AASTAR_SKA_ONLY}}

from rascil.processing_components import plot_uvcoverage
from rascil.processing_components.image.operations import polarisation_frame_from_wcs
from rascil.processing_components.imaging.primary_beams import create_pb

from skasz.senscalc.mid.calculator import Calculator

import numpy as np
import scipy.stats

polarisation_frame = PolarisationFrame('stokesI')

# Main visibility builder
# ------------------------------------------------------------------------------
class Visibility:
    def __init__(self,config='AA4',rmax=None):
        self.config = midsub[config]
    
        self.obs = None
        self.vis = None
        self.sumwt = None
        self.dirty = None

        self.pbeam = None

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
        
        self.pbeam = create_pb(pbeam,'MID',pointingcentre=self.phasecentre,use_local=False)

        if 'img' in kwargs:
            self.addimage(kwargs['img']['hdu'],**kwargs['img']['kwargs'])
        
        if 'pts' in kwargs:
            for pt in kwargs['pts']:
                self.addpoint(pt['direction'],pt['flux'],**pt['kwargs'])


  # Add noise
  # ------------------------------------------------------------------------------
    def addnoise(self,**kwargs):
        
        bllist = self.vis.baselines.data
        vplist = {id: self.config['config'].vp_type.data[i] for i, id in enumerate(self.config['config'].id.data)}

        vpmodel = {'nska': np.array([int(vplist[bl[0]]=='MID') + \
                                     int(vplist[bl[1]]=='MID') for bl in bllist]),
                  'nmeer': np.array([int(vplist[bl[0]]=='MEERKAT') + \
                                     int(vplist[bl[1]]=='MEERKAT') for bl in bllist])}
        
        if self.frequency_channel_centers.min()-0.50*self.obs.frequency_increment_hz<=8.30E+09:
              rx_band = 'Band 5a'
        else: rx_band = 'Band 5b'

        #(10,19503,5,1)
        noise = np.empty_like(self.vis['vis'].data)
        for fi, freq in enumerate(self.frequency_channel_centers):  
            calc = Calculator(target = self.phasecentre,
                             rx_band = rx_band,
                      freq_centre_hz = freq*u.Hz,
                        bandwidth_hz = self.obs.frequency_increment_hz*u.Hz,
              subarray_configuration = self.config['sub'])

            sens = SEFD_array(vpmodel['nska'],vpmodel['nmeer'],calc.sefd_ska[0],calc.sefd_meer[0])
            sens = sens*np.exp(calc.tau)/(calc.eta_system*np.sqrt(2.00*calc.bandwidth*self.integration_time_seconds*u.s))
            sens = sens.to(u.Jy).value

            sens = np.broadcast_to(sens[None,...],(noise.shape[0],noise.shape[1]))

            noise[:,:,fi,0] = scipy.stats.norm.rvs(loc=0.00,scale=sens) + \
                           1j*scipy.stats.norm.rvs(loc=0.00,scale=sens)
        self.vis['vis'].data += noise

  # Add point source component
  # ------------------------------------------------------------------------------  
    def addpoint(self,direction,flux,alpha=-0.70,reference_frequency=1.40E+10,**kwargs):
        flux = flux*(self.frequency_channel_centers/reference_frequency)**alpha
        comp = SkyComponent(flux = flux[:,None],
                       direction = direction,
                       frequency = self.frequency_channel_centers,
              polarisation_frame = polarisation_frame,
                           shape = 'Point',  
                          params = {})
        
        comp = apply_beam_to_skycomponent(comp,self.pbeam)

        direction_cosines, vfluxes = extract_direction_and_flux(comp,self.vis)

        self.vis['vis'].data += dft_kernel(vfluxes = vfluxes,
                                 direction_cosines = direction_cosines,
                                        uvw_lambda = self.vis.visibility_acc.uvw_lambda,
                                dft_compute_kernel = kwargs.get('dft_compute_kernel',None))


  # Add extended source
  # Adapted from RASCIL: rascil.processing_components.image.operations.import_image_from_fits
  # ------------------------------------------------------------------------------
    def addimage(self,hdu,**kwargs):
        hdu = self.rascilhdu(hdu,**kwargs)
        wcs = WCS(hdu)
        
        data = hdu.data.copy()

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

        pbeam = create_image(npixel = np.maximum(*data[0,0].shape),
                           cellsize = np.deg2rad(wcs.wcs.cdelt[0]),
                        phasecentre = self.phasecentre,
                          frequency = self.obs.start_frequency_hz,
                  channel_bandwidth = self.obs.frequency_increment_hz,
                              nchan = self.obs.number_of_channels,
                 polarisation_frame = polarisation_frame)
        
        pbeam = create_pb(pbeam,'MID',pointingcentre=self.phasecentre,use_local=False)

        img = Image.constructor(data=data,polarisation_frame=pol_frame_wcs,wcs=wcs,clean_beam=None)
        img.pixels.data *= pbeam.pixels.data

        vis = predict_ng(self.vis,img,context='2d')
        self.vis['vis'].data += vis['vis'].data; del vis
    

  # Adapt the header for 2D images
  # ------------------------------------------------------------------------------
    def rascilhdu(self,hdu,**kwargs):
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
            return fits.PrimaryHDU(data,header)


  # Image visibilities
  # ------------------------------------------------------------------------------
    def getimage(self,imsize,imcell=None,scale_factor=1.80,**kwargs):
        advice = advise_wide_field(self.vis,guard_band_image=3.0,delA=0.1,facets=1, 
		                           oversampling_synthesised_beam=4.0)
        if imcell is None: 
            print('Overriding imcell with wide-field advice: {0}'.format(advice['cellsize']))
            imcell = scale_factor*advice['cellsize']
        else:
            imcell = imcell.to(u.rad).value  

        model = create_image_from_visibility(self.vis,cellsize=imcell,npixel=imsize,
                                             override_cellsize=kwargs.get('override_cellsize',False))

        self.dirty, self.sumwt = invert_ng(self.vis,model,context='2d')
        self.psf,            _ = invert_ng(self.vis,model,context='2d',dopsf=True)

        return self.dirty.pixels.data, self.psf.pixels.data

  
  # Reset visibilities to zero
  # ------------------------------------------------------------------------------
    def reset(self):
        self.vis['vis'].data = np.zeros(self.vis['vis'].data.shape,dtype=self.vis['vis'].data.dtype)


  # Plot uv coverage
  # ------------------------------------------------------------------------------  
    def showuv(self):
        plot_uvcoverage([self.vis])
        plt.show(); plt.close()