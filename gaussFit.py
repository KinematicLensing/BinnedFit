import numpy as np
from scipy.optimize import curve_fit
import time

class GaussFit():
    def __init__(self, spec2D):

        self.spec2D = spec2D
        self._model_arr = None
        self.peakLambda = None

    @staticmethod
    def gaussian(x, x0, amp, sigma):
        return amp*np.exp(-(x-x0)**2 / (2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)

    def _fit_per_bin(self, function, pos_id):
        '''
            perform fitting on the spec2D.array at each position bin (given pos_id: self.array[pos_id])
            use peak information (spec2D.peak_lambda, spec2D.peak_flux) as an initial starting point before running optimizer
        '''
        # initial guess on the velocity dispersion at fixed position (here the unit is in nm)
        # init_sigma need to be in the same unit as self.lambdaGrid        
        
        if self.spec2D.SNR_pos[pos_id] >= 17.0:
            init_pt = [self.spec2D.peak_lambda[pos_id], self.spec2D.peak_flux[pos_id], 0.8] # [x0,amp,sigma]
        else:
            init_pt = [self.spec2D.peak_lambda[pos_id], self.spec2D.peak_flux[pos_id], 3.0]

        try:
            best_vals, covar = curve_fit(function, self.spec2D.lambdaGrid, self.spec2D.array[pos_id], p0=init_pt, maxfev=1000)
        except RuntimeError:
            print(f'RuntimeError for pos_id: {pos_id}. Set best-fit gaussian amp. = 0')
            best_vals = (self.spec2D.peak_lambda[pos_id], 0., 10.)
            
            #init_pt = [self.spec2D.peak_lambda[pos_id], 1., 100.]
            #try:
            #    best_vals, covar = curve_fit(function, self.spec2D.lambdaGrid, self.spec2D.array[pos_id], p0=init_pt, maxfev=1000)
            #except RuntimeError:
            #    print(f'2nd RuntimeError for pos_id: {pos_id}. Simply set best-fit gaussian amp=0.')
                
    
        return best_vals

    def fit_spec2D(self, function):
        '''
            loop over each position stripe to get fitted_amp, fitted_peakLambda, fitted_sigma
            fitted_peakLambda unit: same as self.lambdaGrid
        '''
        if self.peakLambda is None:
            ngrid_pos = self.spec2D.ngrid_pos
            self.amp = np.zeros(ngrid_pos)
            self.peakLambda = np.zeros(ngrid_pos)
            self.sigma = np.zeros(ngrid_pos)
            
            start_time = time.time()
            for j in range(ngrid_pos):
                self.peakLambda[j], self.amp[j], self.sigma[j] = self._fit_per_bin(function, pos_id=j)
            end_time = time.time()
            print("time cost in gaussFit_spec2D:", (end_time-start_time), "(secs)")

        return self.peakLambda, self.amp, self.sigma

    @property
    def model(self):
        '''
            generate model 2D spectrum based on best fitted parameters derived from self.fit_spec2D
        '''

        if self._model_arr is None:
            self.peakLambda, self.amp, self.sigma = self.fit_spec2D(function=GaussFit.gaussian)
            self._model_arr = np.zeros([self.spec2D.ngrid_pos, self.spec2D.ngrid_spec])
            for j in range(self.spec2D.ngrid_pos):
                self._model_arr[j] = GaussFit.gaussian(self.spec2D.lambdaGrid, self.peakLambda[j], self.amp[j], self.sigma[j])
        
        return self._model_arr



    @property
    def chi2(self):
        return np.sum((self.spec2D.array-self.model)**2/self.spec2D.array_var)
    
    @property
    def reduced_chi2(self):
        Npars = self.spec2D.spaceGrid.size*3
        dof = self.spec2D.array.size - Npars
        return self.chi2/dof
