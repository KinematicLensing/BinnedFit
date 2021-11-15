import numpy as np
from binnedFit_utilities import velocity_to_lambda
from spec2D import Spec2D
from scipy.optimize import curve_fit
from gaussFit import GaussFit
import sys
import pathlib

dir_repo = str(pathlib.Path(__file__).parent.absolute())+'/..'
dir_KLens = dir_repo + '/KLens'
sys.path.append(dir_KLens)
from tfCube2 import Parameters


class RotFitSingle:

    def __init__(self, spec2D):

        self.spec2D = spec2D

        self.init_amp = self.spec2D.peak_flux
        self.thresholdSNR = np.max(self.spec2D.SNR_pos[0:5])+0.2
    
    def gaussian(self, x, x0, amp, sigma):
        return amp*np.exp(-(x-x0)**2 / (2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
    
    def _getPhi_faceOn(self, theta, sini):
        '''
            find the phi angle, when the disk is face on, given the inclination angle (sini), and theta in the image plane of the disk.
        '''
        if sini == 1 or sini == -1:  # disk is edge on.
            phi = np.pi
            return phi
        else:
            cosi = np.sqrt(1-sini**2)
            phi = np.arctan2(np.tan(theta), cosi)
            return phi
    
    def _getTheta_0shear(self, g1, g2, slitAngle):

        tan_slitAng = np.tan(slitAngle)
        theta_0shear = np.arctan2(-g2+(1+g1)*tan_slitAng, (1-g1)-g2*tan_slitAng)

        return theta_0shear

    def model_arctan_rotation(self, pars):
        '''arctan rotation curve in unit of lambdaObs'''
        theta_0shear = self._getTheta_0shear(g1=pars['g1'], g2=pars['g2'], slitAngle=pars['slitAngle'])
        theta_0rotate = theta_0shear - pars['theta_int']
        phi = self._getPhi_faceOn(theta=theta_0rotate, sini=pars['sini'])

        R = self.spec2D.spaceGrid
        
        peakV = pars['v_0'] + 2/np.pi*pars['vcirc']*pars['sini']*np.cos(phi) * np.arctan((R - pars['r_0'])/pars['vscale'])

        cenLambda = velocity_to_lambda(peakV, lambda0=self.spec2D.lambda0, z=pars['redshift'])

        return cenLambda
    
    def _fit_per_bin(self, pos_id, x0i):
        '''
            Perform fitting on the spec2D.array at each position bin, 
            with gaussian x0 being fixed at x0i
        '''
        x0fixed_gaussian = lambda x, amp, sigma: self.gaussian(x, x0i, amp, sigma)

        if self.spec2D.SNR_pos[pos_id] >= self.thresholdSNR:
            init_pt = [self.init_amp[pos_id], 0.8]
        else:
            init_pt = [self.init_amp[pos_id], 3.0]

        try:
            best_vals, covar = curve_fit(x0fixed_gaussian, self.spec2D.lambdaGrid, self.spec2D.array[pos_id], p0=init_pt, sigma=self.spec2D.array_var[pos_id], maxfev=10000)
        except RuntimeError:
            print(f'RuntimeError for pos_id: {pos_id}. Set best-fit gaussian amp. = 0')
            best_vals = (0., 10.)
        
        return best_vals
    
    def fit_spec2D(self, pars):

        ngrid_pos = self.spec2D.ngrid_pos
        cenLambda = self.model_arctan_rotation(pars)
        amp = np.zeros(ngrid_pos)
        sigma = np.zeros(ngrid_pos)

        for j in range(ngrid_pos):
            amp[j], sigma[j] = self._fit_per_bin(j, cenLambda[j])
        
        return cenLambda, amp, sigma
    
    def forward_model(self, pars):

        cenLambda, amp, sigma = self.fit_spec2D(pars)

        model_arr = np.zeros([self.spec2D.ngrid_pos, self.spec2D.ngrid_spec])

        for j in range(self.spec2D.ngrid_pos):
            model_arr[j] = self.gaussian(self.spec2D.lambdaGrid, cenLambda[j], amp[j], sigma[j])
        
        return model_arr

    def cal_chi2(self, model):
        return np.sum((self.spec2D.array-model)**2/self.spec2D.array_var)


class RotFitDouble(RotFitSingle):

    def __init__(self, spec2D):
        super().__init__(spec2D)
        self.lambdaDoublets = spec2D.lineLambda0[spec2D.line_species]
        self.lambda0 = self.spec2D.lambda0

    def gaussian(self, x, x_cen, amp, sigma):

        x0_lo = x_cen/self.lambda0*self.lambdaDoublets[0]
        x0_up = x_cen/self.lambda0*self.lambdaDoublets[1]

        return amp / np.sqrt(2*np.pi*sigma**2) * (np.exp(-(x-x0_lo)**2/(2*sigma**2)) + np.exp(-(x-x0_up)**2/(2*sigma**2)))
    


if __name__ == '__main__':
    dir_binnedFit = str(pathlib.Path(__file__).parent.absolute())
    sys.path.append(dir_binnedFit+'/tests')
    from get_pars import get_pars0

    from gen_mocks import gen_mock_tfCube

    pars, _ = get_pars0()
    dataInfo1 = gen_mock_tfCube(pars, 'Halpha', slits='major', noise_mode=1)
    dataInfo2 = gen_mock_tfCube(pars, 'OII', slits='major', noise_mode=1)

    spec2D_1 = dataInfo1['spec'][0]
    spec2D_2 = dataInfo2['spec'][0]

    # ====================
    RF1 = RotFitSingle(spec2D_1)
    parFid1 = dataInfo1['par_fid'] ; parFid1['slitAngle'] = parFid1['slitAngles'][0]
    model1 = RF1.forward_model(parFid1)
    print('chi2: ', RF1.cal_chi2(model1))

    spec2D_1.display(xlim=[-2., 2.,], model=model1)

    # ====================
