from get_pars import get_pars0
from tfCube2 import Parameters
import sys
import pathlib
import numpy as np
import time
import emcee
from multiprocessing import Pool

dir_repo = str(pathlib.Path(__file__).parent.absolute())+'/../..'
dir_KLens = dir_repo + '/KLens'
sys.path.append(dir_KLens)
dir_binnedFit = dir_repo + '/BinnedFit'
sys.path.append(dir_binnedFit)
sys.path.append(dir_binnedFit+'/tests')
from get_pars import get_pars0
from gen_mocks import gen_mock_tfCube
from gamma import GammaInference

pars, _ = get_pars0()
dataInfo = gen_mock_tfCube(pars, 'Halpha', slits='both', noise_mode=0)

# ----------------------
GI = GammaInference(dataInfo, active_par_key=[
    'vcirc', 'sini', 'vscale', 'r_0', 'v_0', 'g1', 'g2',  'r_hl_image', 'theta_int', 'flux'], par_fix=None, vTFR_mean=200.)

chainInfo = GI.run_MCMC(Nwalker=20, Nsteps=4, outfile_MCMC="./chain_Ha_noise0.pkl", save_step_size=2)
