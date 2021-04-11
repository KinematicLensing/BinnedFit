import numpy as np
import sys
import pathlib
dir_repo = str(pathlib.Path(__file__).parent.absolute()) + '/..'
dir_KLens = dir_repo + '/KLens'
sys.path.append(dir_KLens)

dir_binnedFit = dir_repo + '/BinnedFit'
sys.path.append(dir_binnedFit)

dir_TNGcube = dir_repo + '/TNGcube'
sys.path.append(dir_TNGcube)

from binnedFit_utilities import cal_e_int, cal_theta_obs
from spec2D import Spec2D
from TNGcube import Image
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
        dataInfo['spec'][j] = Spec2D(array=dataInfo['spec'][j], array_var=dataInfo['spec_variance'][j],spaceGrid=dataInfo['spaceGrid'], lambdaGrid=dataInfo['lambdaGrid'], line_species=line_species, z=dataInfo['par_fid']['redshift'], auto_cut=False)
    
    # make Image object
    dataInfo['image'] = Image(dataInfo['image'], dataInfo['spaceGrid'], array_var=dataInfo['image_variance'])
    
    return dataInfo


if __name__ == '__main__':
    sys.path.append(dir_binnedFit+'/tests')
    from get_pars import get_pars0

    pars, line_species = get_pars0()
    dataInfo = gen_mock_tfCube(pars, line_species, slits='major', noise_mode=0)

    #dataInfo['image'].display(xlim=[-2,2])
