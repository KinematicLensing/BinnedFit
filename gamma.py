import numpy as np
import time
import emcee
import sys
import pathlib

from imageFit import ImageFit
from rotCurveFit import RotFitSingle, RotFitDouble
from binnedFit_utilities import load_pickle, save_pickle

dir_repo = str(pathlib.Path(__file__).parent.absolute())+'/..'
dir_KLens = dir_repo + '/KLens'
sys.path.append(dir_KLens)
from tfCube2 import Parameters

import os
os.environ["OMP_NUM_THREADS"] = "1"   # suggested to set this for the parallel of emcee
from multiprocessing import Pool
from multiprocessing import cpu_count
from schwimmbad import MPIPool

class GammaInference():
    '''The main binnedFit pipeline that performs parameter inferences given data.
        Args:
            dataInfo: a dictionary that stores the data information.

            active_par_key: a list of parameters to be vary during the MCMC sampling process.

                e.g. active_par_key=['vcirc', 'sini', 'vscale', 'r_0', 'v_0', 'g1', 'g2',  'r_hl_image', 'theta_int', 'flux']

            par_fix: dict 
                To force some parameters to be fixed to certain values during the MCMC sampling.
                e.g par_fix={'v_0': 0., 'theta_int': 0.}
            
            vTFR_mean: double
                Expected rotational velocity for the input galaxy from tully-fisher relation.
    '''

    def __init__(self, dataInfo, active_par_key=['vcirc', 'sini', 'vscale', 'r_0', 'v_0', 'g1', 'g2',  'r_hl_image', 'theta_int', 'flux'], par_fix=None, vTFR_mean=None):

        self.sigma_TF_intr = 0.08

        if vTFR_mean is None:
            self.vTFR_mean = 200.
        else:
            self.vTFR_mean = vTFR_mean
                
        self.Pars = Parameters(par_in=dataInfo['par_fid'], line_species=dataInfo['line_species'])

        self.active_par_key = active_par_key

        self.par_fid = dataInfo['par_fid']
        self.par_fix = par_fix

        if self.par_fix is not None:
            self.par_base = self.Pars.gen_par_dict(active_par=list(self.par_fix.values()), active_par_key=list(self.par_fix.keys()), par_ref=self.par_fid)
        else:
            self.par_base = self.par_fid.copy()
        
        self._is_singlet = dataInfo['spec'][0]._is_singlet
        self.Nspec = len(dataInfo['spec'])
        
        self.ImgFit = ImageFit(dataInfo['image'], par_init=self.par_base)

        self.RFs = []
        for j in range(self.Nspec):
            if self._is_singlet:
                self.RFs.append(RotFitSingle(dataInfo['spec'][j]))
            else:
                self.RFs.append(RotFitDouble(dataInfo['spec'][j]))

        self.par_lim = self.Pars.set_par_lim()  # defined in tfCube2.Parameters.set_par_lim()
        self.par_std = self.Pars.set_par_std()

        self._parpare_storage_info()

    def _cal_loglike_image(self, pars):
        modelImg = self.ImgFit.forward_model(pars)
        logL_img = -0.5*self.ImgFit.cal_chi2(modelImg)
        return logL_img, modelImg
    
    def _cal_bestfit_specStats(self, pars, IDspec):
        pars['slitAngle'] = pars['slitAngles'][IDspec]
        cenLambda, amp, sigma = self.RFs[IDspec].fit_spec2D(pars)
        return cenLambda, amp, sigma
    
    def _cal_loglike_spec(self, pars, IDspec):
        pars['slitAngle'] = pars['slitAngles'][IDspec]
        modelSpec_i = self.RFs[IDspec].forward_model(pars)
        logL_spec_i = -0.5*self.RFs[IDspec].cal_chi2(modelSpec_i)
        return logL_spec_i, modelSpec_i

    def cal_loglike(self, active_par):
        
        pars = self.Pars.gen_par_dict(active_par=active_par, active_par_key=self.active_par_key, par_ref=self.par_base)
        
        for item in self.active_par_key:
            if (pars[item] < self.par_lim[item][0] or pars[item] > self.par_lim[item][1]):
                return -np.inf
        
        logL_img, _ = self._cal_loglike_image(pars)
        
        logPrior_vcirc = self.Pars.logPrior_vcirc(vcirc=pars['vcirc'], sigma_TF_intr=self.sigma_TF_intr, vTFR_mean=self.vTFR_mean)
        logL_spec = 0.
        for j in range(self.Nspec):
            logL_spec += self._cal_loglike_spec(pars, j)[0]
        
        loglike = logL_img+logL_spec+logPrior_vcirc
        
        return loglike

    def _parpare_storage_info(self):

        self.chainInfo = {}
        self.chainInfo['par_fid'] = self.par_fid.copy()
        self.chainInfo['par_fix'] = self.par_fix
        self.chainInfo['par_name'] = self.Pars.set_par_name()
        self.chainInfo['par_key'] = self.active_par_key
    
    def save_chain(self, sampler, outfile_MCMC):

        self.chainInfo['acceptance_fraction'] = np.mean(sampler.acceptance_fraction)  # good range: 0.2~0.5
        self.chainInfo['lnprobability'] = sampler.lnprobability
        self.chainInfo['chain'] = sampler.chain
        save_pickle(filename=outfile_MCMC, info=self.chainInfo)
        
        return self.chainInfo
    
    def run_MCMC(self, Nwalker, Nsteps, outfile_MCMC=None, save_step_size=2):

        Ndim = len(self.active_par_key)

        starting_point = [self.par_fid[item] for item in self.active_par_key]
        std = [self.par_std[item] for item in self.active_par_key]

        p0_walkers = emcee.utils.sample_ball(starting_point, std, size=Nwalker)
        
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(Nwalker, Ndim, self.cal_loglike, a=2.0, pool=pool)

            step_size = save_step_size
            steps_taken = 0

            ##### start long sampling
            Tstart = time.time()

            while steps_taken < Nsteps:
                posInfo = sampler.run_mcmc(p0_walkers, step_size, progress=True)
                p0_walkers = posInfo.coords
                steps_taken+=step_size
                print("steps_taken", steps_taken)

                self.chainInfo = self.save_chain(sampler=sampler, outfile_MCMC=outfile_MCMC)

            Time_MCMC = (time.time()-Tstart)/60.
            print("Total MCMC time (mins):", Time_MCMC)

        return self.chainInfo
    
    def run_MCMC_mpi(self, Nwalker, Nsteps, outfile_MCMC=None, save_step_size=2):

        Ndim = len(self.active_par_key)

        starting_point = [self.par_fid[item] for item in self.active_par_key]
        std = [self.par_std[item] for item in self.active_par_key]

        p0_walkers = emcee.utils.sample_ball(starting_point, std, size=Nwalker)
        
        with MPIPool() as pool:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
                
            sampler = emcee.EnsembleSampler(Nwalker, Ndim, self.cal_loglike, a=2.0, pool=pool)

            step_size = save_step_size
            steps_taken = 0

            ##### start long sampling
            Tstart = time.time()

            while steps_taken < Nsteps:
                posInfo = sampler.run_mcmc(p0_walkers, step_size, progress=True)
                p0_walkers = posInfo.coords
                steps_taken+=step_size
                print("steps_taken", steps_taken)

                self.chainInfo = self.save_chain(sampler=sampler, outfile_MCMC=outfile_MCMC)

            Time_MCMC = (time.time()-Tstart)/60.
            print("Total MCMC time (mins):", Time_MCMC)

        return self.chainInfo
        

if __name__ == '__main__':
    dir_binnedFit = str(pathlib.Path(__file__).parent.absolute())
    sys.path.append(dir_binnedFit+'/tests')
    from get_pars import get_pars0

    from gen_mocks import gen_mock_tfCube

    pars, _ = get_pars0()
    dataInfo = gen_mock_tfCube(pars, 'Halpha', slits='both', noise_mode=0)

    # ----------------------
    GI = GammaInference(dataInfo, active_par_key=[
                        'vcirc', 'sini', 'vscale', 'r_0', 'v_0', 'g1', 'g2',  'r_hl_image', 'theta_int', 'flux'], par_fix=None, vTFR_mean=200.)
    
    active_par_fid = [GI.par_fid[key] for key in GI.active_par_key]
    loglike_at_fid = GI.cal_loglike(active_par_fid)

    # -- check fit status --
    pars = GI.Pars.gen_par_dict(
        active_par=active_par_fid, active_par_key=GI.active_par_key, par_ref=GI.par_base)
    dataInfo['image'].display(xlim=[-2.5, 2.5], model=GI.ImgFit.forward_model(pars))

    j = 0
    pars['slitAngle'] = pars['slitAngles'][j]
    dataInfo['spec'][j].display(xlim=[-2.5, 2.5], model=GI.RFs[j].forward_model(pars))

    j = 1
    pars['slitAngle'] = pars['slitAngles'][j]
    dataInfo['spec'][j].display(xlim=[-2.5, 2.5], model=GI.RFs[j].forward_model(pars))








        


        

        




        





    



    


        
