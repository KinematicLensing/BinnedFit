import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import time

class GaussFit():
    def __init__(self, spec2D):

        self.spec2D = spec2D
        self._model_arr = None

        self.cenLambda = None
        self._init_cenLambda = None
        self._init_amp = None

        self.thresholdSNR = np.max(self.spec2D.SNR_pos[0:5])+0.2

    @staticmethod
    def gaussian(x, x0, amp, sigma):
        return amp*np.exp(-(x-x0)**2 / (2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
    
    @property
    def init_cenLambda(self):
        '''initial guess for cenLambda (before running optimizer)'''
        if self._init_cenLambda is None:
            self._init_cenLambda = self.spec2D.peak_lambda
        return self._init_cenLambda
    
    @property
    def init_amp(self):
        '''initial guess for flux amp. (before running optimizer)'''
        if self._init_amp is None:
            self._init_amp = self.spec2D.peak_flux
        return self._init_amp

    def _fit_per_bin(self, pos_id):
        '''
            perform fitting on the spec2D.array at each position bin (given pos_id: self.array[pos_id])
            use peak information (spec2D.peak_lambda, spec2D.peak_flux) as an initial starting point before running optimizer
        '''
        # initial guess on the velocity dispersion at fixed position (here the unit is in nm)
        # init_sigma need to be in the same unit as self.lambdaGrid        
        
        if self.spec2D.SNR_pos[pos_id] >= self.thresholdSNR:
            init_pt = [self.init_cenLambda[pos_id], self.init_amp[pos_id], 0.8] # [x0,amp,sigma]
        else:
            init_pt = [self.init_cenLambda[pos_id], self.init_amp[pos_id], 3.0]

        try:
            best_vals, covar = curve_fit(self.gaussian, self.spec2D.lambdaGrid, self.spec2D.array[pos_id], 
            p0=init_pt, sigma=self.spec2D.array_var[pos_id], maxfev=1000)
        except RuntimeError:
            print(f'RuntimeError for pos_id: {pos_id}. Set best-fit gaussian amp. = 0')
            best_vals = (self.init_cenLambda[pos_id], 0., 10.)
            
        return best_vals

    def fit_spec2D(self):
        '''
            loop over each position stripe to get fitted cenLambda, amp, sigma
            fitted cenLambda unit: same as self.lambdaGrid
        '''
        if self.cenLambda is None:
            ngrid_pos = self.spec2D.ngrid_pos
            self.amp = np.zeros(ngrid_pos)
            self.cenLambda = np.zeros(ngrid_pos)
            self.sigma = np.zeros(ngrid_pos)
            
            start_time = time.time()
            for j in range(ngrid_pos):
                self.cenLambda[j], self.amp[j], self.sigma[j] = self._fit_per_bin(pos_id=j)
            end_time = time.time()
            print("time cost in gaussFit_spec2D:", (end_time-start_time), "(secs)")

        return self.cenLambda, self.amp, self.sigma

    @property
    def model(self):
        '''
            generate model 2D spectrum based on best fitted parameters derived from self.fit_spec2D
        '''

        if self._model_arr is None:
            self.cenLambda, self.amp, self.sigma = self.fit_spec2D()
            self._model_arr = np.zeros([self.spec2D.ngrid_pos, self.spec2D.ngrid_spec])
            for j in range(self.spec2D.ngrid_pos):
                self._model_arr[j] = self.gaussian(self.spec2D.lambdaGrid, self.cenLambda[j], self.amp[j], self.sigma[j])
        
        return self._model_arr

    @property
    def chi2(self):
        return np.sum((self.spec2D.array-self.model)**2/self.spec2D.array_var)
  
    @property
    def reduced_chi2(self):
        Npars = self.spec2D.spaceGrid.size*3
        dof = self.spec2D.array.size - Npars
        return self.chi2/dof


class GaussFitDouble(GaussFit):
    
    def __init__(self, spec2D):
        super().__init__(spec2D)

        self.lambdaDoublets = spec2D.lineLambda0[spec2D.line_species]
        self.lambda0 = self.spec2D.lambda0

        self._spec2Dsm = gaussian_filter(spec2D.array, sigma=1.5)
        self._grad1map = np.gradient(self._spec2Dsm, axis=1)
        self._grad2map = np.gradient(self._grad1map, axis=1)
        self.init_cenID = self.gen_init_cenID()
    
    def cal_peak_lambda(self, cenLambda, mode='lo'):
        '''compute peak lambda (of the lower or upper doublet) given cenLambda
            Args:
                mode = 'lo' or 'up'
        '''
        if mode == 'lo':
            return cenLambda/self.lambda0 * self.lambdaDoublets[0]
        elif mode == 'up':
            return cenLambda/self.lambda0 * self.lambdaDoublets[1]
        else:
            raise ValueError('mode needs to be set as lo or up')
        

    def gaussian(self, x, x_cen, amp, sigma):

        x0_lo = self.cal_peak_lambda(x_cen, mode='lo')
        x0_up = self.cal_peak_lambda(x_cen, mode='up')

        return amp / np.sqrt(2*np.pi*sigma**2) * ( np.exp(-(x-x0_lo)**2/(2*sigma**2)) + np.exp(-(x-x0_up)**2/(2*sigma**2)) )
    
    def _find_cenID_per_bin(self, pos_id):
        '''identify central lambdaGrid ID where the local min is located for a doublet'''
        grad1 = self._grad1map[pos_id, :]
        grad2 = self._grad2map[pos_id, :]

        threshold = np.max(self._spec2Dsm[pos_id, :])*0.45

        takeoutID = self._spec2Dsm[pos_id, :] > threshold   # high signal
        takeoutID *= grad2 > 0                              # 2nd derivative > 0.

        if any(takeoutID): # if there are Ture entries in takeoutID
            pickID = np.abs(grad1[takeoutID]).argmin()          # 1st der -> to 0 (within the takeoutID points)
            cenID = np.array(range(self.spec2D.ngrid_spec))[takeoutID][pickID]
            return cenID
        else:
            return 999
        
    
    def gen_init_cenID(self):
        '''initial guess for cenID (before running optimizer)'''
        ngrid_pos = self.spec2D.ngrid_pos
        init_cenID = np.zeros(ngrid_pos, dtype=int)

        for pos_id in range(ngrid_pos):
            init_cenID[pos_id] = self._find_cenID_per_bin(pos_id)

        return init_cenID
    
    @property
    def init_cenLambda(self):
        '''initial guess for cenLambda (before running optimizer)'''
        if self._init_cenLambda is None:
            lambdaV0 = (1.+self.spec2D.z) * self.spec2D.lambda0
            self._init_cenLambda = np.array([self.spec2D.lambdaGrid[ID] if ID != 999 else lambdaV0 for ID in self.init_cenID])
        return self._init_cenLambda



if __name__ == '__main__':
    import sys
    import pathlib
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
    GF1 = GaussFit(spec2D_1)
    GF2 = GaussFitDouble(spec2D_2)

    model1 = GF1.model
    model2 = GF2.model
