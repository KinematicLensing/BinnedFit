import numpy as np
from scipy.optimize import curve_fit
import time
import emcee
from multiprocessing import Pool

import galsim
from binnedFit_utilities import cal_e_int

import sys
import pathlib
dir_repo = str(pathlib.Path(__file__).parent.absolute())+'/..'
dir_KLens = dir_repo + '/KLens'
sys.path.append(dir_KLens)
from tfCube2 import Parameters, GalaxyImage

class ImageFit():
    def __init__(self, image, par_init=None):
        self.image = image

        if par_init is None:
            self.par_init = Parameters().fid
        else:
            self.par_init = par_init

        self.psf = self._set_psf(self.par_init)

    def _set_psf(self, pars):
        psf = galsim.Gaussian(fwhm=pars['psfFWHM'])
        psf = psf.shear(g1=pars['psf_g1'], g2=pars['psf_g2'])
        return psf

    def forward_model(self, pars):

        disk = galsim.Sersic(n=1, half_light_radius=pars['r_hl_image'], flux=pars['flux'], trunc=4*pars['r_hl_image'])

        e = cal_e_int(sini=pars['sini'], q_z=pars['aspect'])
        g1_int = e/2.
        disk = disk.shear(g1=g1_int, g2=0.0)
        disk = disk.rotate(pars['theta_int']*galsim.radians)
        disk = disk.shear(g1=pars['g1'], g2=pars['g2'])

        galObj = galsim.Convolution([disk, self.psf])

        image0 = galsim.Image(pars['ngrid'], pars['ngrid'], scale=pars['subGridPixScale'])
        image = galObj.drawImage(image=image0)

        return image.array
    
    def simple_model(self, e, half_light_radius, theta_int, flux, g1=0., g2=0.):
        '''Simple image model, only involves key parameters to describe an image
            This function is used to find the major axis direction of image only data 
        '''
        disk = galsim.Sersic(n=1, half_light_radius=half_light_radius, flux=flux, trunc=4*half_light_radius)
        disk = disk.shear(g1=e/2., g2=0.0)
        disk = disk.rotate(theta_int*galsim.radians)
        disk = disk.shear(g1=g1, g2=g2)
        galObj = galsim.Convolution([disk, self.psf])
        _image = galsim.Image(self.image.ngrid, self.image.ngrid, scale=self.image.pixScale)
        newImage = galObj.drawImage(image=_image)
        return newImage.array
    
    def cal_chi2(self, model):
        return np.sum((self.image.array-model)**2/self.image.array_var)
    

    def _init_MCMC(self, active_par_key=['e_obs', 'r_hl_image', 'theta_int', 'flux'], par_fix={'g1':0., 'g2':0.}):
        self.active_par_key = active_par_key
        self.par_fix = par_fix

        self.Pars = Parameters(par_in=self.par_init, line_species='Halpha')
        
        if par_fix is not None:
            self.par_base = self.Pars.gen_par_dict(active_par=list(self.par_fix.values()), active_par_key=list(self.par_fix.keys()), par_ref=self.par_init)
        else:
            self.par_base = self.par_init
        
        self.par_lim = self.Pars.set_par_lim()
        self.par_std = self.Pars.set_par_std()

    def cal_loglike(self, active_par):

        pars = self.Pars.gen_par_dict(active_par=active_par, active_par_key=self.active_par_key, par_ref=self.par_base)

        for item in self.active_par_key:
            if (pars[item] < self.par_lim[item][0] or pars[item] > self.par_lim[item][1]):
                return -np.inf
        
        modelImg = self.simple_model(e=pars['e_obs'], half_light_radius=pars['r_hl_image'], theta_int=pars['theta_int'], flux=pars['flux'], g1=pars['g1'], g2=pars['g2'])

        logL_img = -0.5*self.cal_chi2(model=modelImg)

        return logL_img

    def run_MCMC(self, Nwalker, Nsteps, active_par_key=['e_obs', 'r_hl_image', 'theta_int', 'flux'], par_fix={'g1': 0., 'g2': 0.}):

        self._init_MCMC(active_par_key=active_par_key, par_fix=par_fix)
        Ndim = len(self.active_par_key)

        starting_point = [self.par_init[item] for item in self.active_par_key]
        std = [self.par_std[item] for item in self.active_par_key]
        p0_walkers = emcee.utils.sample_ball(starting_point, std, size=Nwalker)

        sampler = emcee.EnsembleSampler(Nwalker, Ndim, self.cal_loglike, a=5.0)
        posInfo = sampler.run_mcmc(p0_walkers, 2)
        p0_walkers = posInfo.coords
        sampler.reset()

        Tstart = time.time()
        posInfo = sampler.run_mcmc(p0_walkers, Nsteps, progress=True)
        Time_MCMC = (time.time()-Tstart)/60.
        print("Total MCMC time (mins):", Time_MCMC)

        chainInfo = {}
        chainInfo['acceptance_fraction'] = np.mean(sampler.acceptance_fraction)  # good range: 0.2~0.5
        chainInfo['lnprobability'] = sampler.lnprobability
        chainInfo['chain'] = sampler.chain
        chainInfo['par_key'] = self.active_par_key
        chainInfo['par_fid'] = self.par_init
        chainInfo['par_fix'] = self.par_fix

        return chainInfo



if __name__ == '__main__':
    dir_binnedFit = str(pathlib.Path(__file__).parent.absolute())
    sys.path.append(dir_binnedFit+'/tests')
    from get_pars import get_pars0

    from gen_mocks import gen_mock_tfCube

    pars, _ = get_pars0()
    dataInfo = gen_mock_tfCube(pars, 'Halpha', slits='major', noise_mode=1)

    image = dataInfo['image']

    # ====================
    ImFit = ImageFit(image=image, par_init=dataInfo['par_fid'])
    model = ImFit.forward_model(pars=dataInfo['par_fid'])

    print('chi2: ', ImFit.cal_chi2(model))
    image.display(xlim=[-2.5, 2.5], model=model)
    # ====================




    


