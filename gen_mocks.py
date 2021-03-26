import numpy as np
import astropy.units as u
import sys
import pathlib
dir_repo = str(pathlib.Path(__file__).parent.absolute()) + '/..'
dir_tfCube = dir_repo + '/KLens'
sys.path.append(dir_tfCube)

dir_binnedFit = dir_repo + '/BinnedFit'
sys.path.append(dir_binnedFit)

from binnedFit_utilities import cal_e_int, cal_theta_obs, Spec2D
from KLtool import find_flux_norm
from tfCube2 import TFCube, Parameters


def gen_mock_tfCube(pars=None, line_species='Halpha', slits='both', noise_mode=0):

    if pars is None:
        pars = Parameters()
    
    pars['flux'] = find_flux_norm(pars, R=1.5)
    
    eint_thy = cal_e_int(sini=pars['sini'], q_z=pars['aspect'])
    theta_obs = cal_theta_obs(g2=pars['g2'], e_int=eint_thy, theta_int=pars['theta_int'])
    
    slitAng_major_p = theta_obs
    slitAng_minor_p = theta_obs + np.pi / 2.
    
    if slits == 'major':
        pars['slitAngles'] = np.array([slitAng_major_p])
        
    elif slits == 'minor':
        pars['slitAngles'] = np.array([slitAng_minor_p])
        
    elif slits == 'both':
        pars['slitAngles'] = np.array([slitAng_major_p, slitAng_minor_p])
        
    # ------ generate mock data ------
    TF = TFCube(pars=pars, line_species=line_species)
    dataInfo = TF.gen_mock_data(noise_mode=noise_mode)

    # ------ modify the output of dataInfo to be compatible with binnedFit ------
    del dataInfo['par_meta']

    # change dataInfo['spec'] = [2Darr_1, 2Darr_2] from a list of np.array to be a list of Spec2D objects
    # making Spec2D objects
    for j in range(len(dataInfo['spec'])):
        dataInfo['spec'][j] = Spec2D(array=dataInfo['spec'][j], spaceGrid=dataInfo['spaceGrid'], lambdaGrid=dataInfo['lambdaGrid'], array_var=dataInfo['spec_variance'][j])
    
    dataInfo['lambda0'] = pars['linelist']['lambda'][pars['linelist']['species']==line_species][0]
    
    return dataInfo



if __name__ == '__main__':
    line_species = 'Halpha'

    pars = {}
    pars['g1'] = 0.0  # 0.05
    pars['g2'] = 0.0  # 0.05
    pars['sini'] = 0.5  # 0.5

    pars['redshift'] = 0.4

    pars['aspect'] = 0.2
    pars['r_hl_image'] = 0.5
    pars['r_hl_spec'] = 0.5

    pars['theta_int'] = 0.  # np.pi/3.

    pars['slitWidth'] = 0.12
    pars['ngrid'] = 256

    pars['Resolution'] = 5000.  # 6000.
    pars['expTime'] = 60.*30.  # 60.*30.
    pars['pixScale'] = 0.1185
    pars['nm_per_pixel'] = 0.033
    pars['throughput'] = 0.29
    pars['psfFWHM'] = 0.5  # 0.5

    pars['area'] = 3.14 * (1000./2.)**2

    pars['vcirc'] = 200.


    linelist = np.empty(5, dtype=[('species', np.str_, 16),
                                ('lambda', float),
                                ('flux', float)])
    linelist['species'] = ['OIIa', 'OIIb', 'OIIIa', 'OIIIb', 'Halpha']
    linelist['lambda'] = [372.7092, 372.9875, 496.0295, 500.8240, 656.461]
    fiber_SDSS = np.pi * 1.5**2
    refSDSSspec = 3.*1e-17 * u.erg/u.second/u.Angstrom/u.cm**2
    refSDSSspec = refSDSSspec.to(u.erg/u.second/u.nm/u.cm**2)
    linelist['flux'] = refSDSSspec.value / fiber_SDSS  # [unit: erg/s/cm2/nm/arcsec2]
    pars['linelist'] = linelist


    dataInfo = gen_mock_tfCube(pars=pars, line_species=line_species, slits='both', noise_mode=0)

