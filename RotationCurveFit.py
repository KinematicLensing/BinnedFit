import numpy as np
import emcee
from multiprocessing import Pool
from binnedFit_utilities import velocity_to_lambda
from spec2D import Spec2D
from gaussFit import GaussFit
import sys
import pathlib

dir_repo = str(pathlib.Path(__file__).parent.absolute())+'/..'
dir_KLens = dir_repo + '/KLens'
sys.path.append(dir_KLens)
from tfCube2 import Parameters

class RotationCurveFit():

    def __init__(self, data_info, active_par_key=['vcirc', 'sini', 'vscale', 'r_0', 'v_0', 'g1', 'g2', 'theta_int'], par_fix=None, vTFR_mean=None, thresholdSNR=0, is_weightSNR=1.):
        '''
            e.g. 
            active_par_key = ['vscale', 'r_0', 'sini', 'v_0'] # 'redshift'
            par_fix = {'redshift': 0.598}
        '''

        self.sigma_TF_intr = 0.08

        if vTFR_mean is None:
            self.vTFR_mean = 200.
        else:
            self.vTFR_mean = vTFR_mean

        self.thresholdSNR = thresholdSNR

        self.is_weightSNR = is_weightSNR

        self.Pars = Parameters(par_in=data_info['par_fid'], line_species=data_info['line_species'])

        self.par_fid = data_info['par_fid']
        self.par_fix = par_fix

        if self.par_fix is not None:
            self.par_base = self.Pars.gen_par_dict(active_par=list(self.par_fix.values()), active_par_key=list(self.par_fix.keys()), par_ref=self.par_fid)
        else:
            self.par_base = self.par_fid.copy()

        self.active_par_key = active_par_key
        self.Ntot_active_par = len(self.active_par_key)
        self.Nspec = len(data_info['spec'])

        self.spec = data_info['spec']
        self.lambda0 = data_info['lambda0']
        
        self.spec_stats = self.spec_statstics()

        self.par_lim = self.Pars.set_par_lim() # defined in tfCube2.Parameters.set_par_lim()
        self.par_std = self.Pars.set_par_std()

    def spec_statstics(self):
        
        spec_stats = []
        for j in range(self.Nspec):
            print(self.spec[j].array.shape)
            GF = GaussFit(spec2D=self.spec[j], thresholdSNR=self.thresholdSNR)
            self.spec[j] = GF.spec2D
            print(self.spec[j].array.shape)
            stats = {}
            stats['peakLambda'], stats['amp'], stats['sigma'] = GF.fit_spec2D()
            spec_stats.append(stats)
        
        return spec_stats


    def getPhi_faceOn(self, theta, sini):
        '''
            find the phi angle, when the disk is face on, given the inclination angle (sini), and theta in the image plane of the disk.
        '''
        if sini == 1 or sini == -1: # disk is edge on.
            phi = np.pi
            return phi
        else:
            cosi = np.sqrt(1-sini**2)
            phi = np.arctan2(np.tan(theta), cosi)
            #phi = np.arctan2(np.sin(theta)/cosi, np.cos(theta))
            return phi
    
    def getTheta_0shear(self, g1, g2, slitAngle):

        tan_slitAng = np.tan(slitAngle)
        theta_0shear = np.arctan2(-g2+(1+g1)*tan_slitAng, (1-g1)-g2*tan_slitAng)

        return theta_0shear

    def model_arctan_rotation(self, r, vcirc, sini, vscale, r_0, v_0, g1, g2, theta_int, redshift, slitAngle):
        '''
            arctan rotation curve in unit of lambda_obs, given cosmological redshift
        '''
        theta_0shear = self.getTheta_0shear(g1=g1, g2=g2, slitAngle=slitAngle)
        theta_0rotate = theta_0shear - theta_int
        #theta_0rotate = slitAngle
        phi = self.getPhi_faceOn(theta=theta_0rotate, sini=sini)

        if (theta_0rotate > 1.5707963267948966) or (theta_0rotate < 0.):
            R = np.flip(r)
        else :
            R = r

        peak_V = v_0 + 2/np.pi * vcirc * sini * np.cos(phi) * np.arctan((R - r_0)/vscale)
        model_lambda = velocity_to_lambda(v_peculiar=peak_V, lambda0=self.lambda0, redshift=redshift)

        return model_lambda
    
    def cal_chi2(self, active_par):

        par = self.Pars.gen_par_dict(active_par=active_par, active_par_key=self.active_par_key, par_ref=self.par_base)
        
        chi2_tot = 0.
        for j in range(self.Nspec):
            
            model = self.model_arctan_rotation(r=self.spec[j].spaceGrid, vcirc=par['vcirc'], sini=par['sini'], vscale=par['vscale'], r_0=par['r_0'], v_0=par['v_0'], g1=par['g1'], g2=par['g2'], theta_int=par['theta_int'], redshift=par['redshift'], slitAngle=par['slitAngles'][j])
            #print(model)

            diff = self.spec_stats[j]['peakLambda'] - model
            if self.is_weightSNR:
                # normalizing the weighting factor such that weight_SNR.mean() = 1.0
                weight_SNR = self.spec[j].SNR_pos / self.spec[j].SNR_pos.mean() 
                chi2 = np.sum(weight_SNR*(diff/self.spec_stats[j]['sigma'])**2)
            else:
                chi2 = np.sum((diff/self.spec_stats[j]['sigma'])**2)
            chi2_tot += chi2

        return chi2_tot

    def cal_loglike(self, active_par):
        
        par = self.Pars.gen_par_dict(active_par=active_par, active_par_key=self.active_par_key, par_ref=self.par_base)

        for item in self.active_par_key:
            if ( par[item] < self.par_lim[item][0] or par[item] > self.par_lim[item][1] ):
                return -np.inf
        
        logPrior_vcirc = self.Pars.logPrior_vcirc(vcirc=par['vcirc'], sigma_TF_intr=self.sigma_TF_intr, vTFR_mean=self.vTFR_mean)

        chi2 = self.cal_chi2(active_par)

        loglike = -0.5*chi2 + logPrior_vcirc

        return loglike

    def run_MCMC(self, Nwalker, Nsteps):

        Ndim = self.Ntot_active_par
        starting_point = [ self.par_fid[item] for item in self.active_par_key ]
        std = [ self.par_std[item] for item in self.active_par_key ]

        p0_walkers = emcee.utils.sample_ball(starting_point, std, size=Nwalker)

        sampler = emcee.EnsembleSampler(Nwalker, Ndim, self.cal_loglike, a=2.0)
        # emcee a parameter: Npar < 4 -> better set a > 3  ( a = 5.0 )
                    #                  : Npar > 7 -> better set a < 2  ( a = 1.5 ) 
        posInfo = sampler.run_mcmc(p0_walkers,5)
        p0_walkers = posInfo.coords
        sampler.reset()
            
        Tstart=time.time()
        posInfo = sampler.run_mcmc(p0_walkers, Nsteps, progress=True)
        Time_MCMC=(time.time()-Tstart)/60.
        print ("Total MCMC time (mins):",Time_MCMC)

        chain_info = {}
        chain_info['acceptance_fraction'] = np.mean(sampler.acceptance_fraction) # good range: 0.2~0.5
        chain_info['lnprobability'] = sampler.lnprobability
        chain_info['par_fid'] = self.par_fid
        chain_info['chain'] = sampler.chain
        chain_info['par_key'] = self.active_par_key
        chain_info['par_fix'] = self.par_fix

        return chain_info


