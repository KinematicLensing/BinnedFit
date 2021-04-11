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
            self.par_init = Parameters()
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
    
    def cal_chi2(self, model):
        return np.sum((self.image.array-model)**2/self.image.array_var)
        
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




    


