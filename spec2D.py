import numpy as np
from matplotlib import pyplot as plt
from gaussFit import GaussFit, GaussFitDouble
from binnedFit_utilities import lambda_to_velocity, velocity_to_lambda

class Spec2D:
    '''Slit Spectrum data class'''
    lineLambda0 = {'Halpha': 656.461, 'OII': [372.7092, 372.9875], 'OIII': [496.0295,500.8240]}

    def __init__(self, array, array_var, spaceGrid, lambdaGrid, line_species, z, auto_cut=False):
        '''
            Args:
                array: 2D array, shape (ngrid_pos, ngrid_spec)
                array_var: 2D array, storing the variance of self.array
                    e.g. The sky slit spectrum genertate by tfCube.skySpec2D()
                spaceGrid: 1D array
                lambdaGrid: 1D array
                
                line_species: str, 'Halpha', 'OII', 'OIII'
                z : redshift of the spectrum
        '''
        self.array = array
        self.spaceGrid = spaceGrid
        self.lambdaGrid = lambdaGrid
        self.line_species = line_species
        self.z = z

        self.array_var = array_var
        
        self.lambda0 = np.mean(Spec2D.lineLambda0[self.line_species])

        self.center = (0., (1.+self.z)*self.lambda0)

        if auto_cut:
            self.auto_cutout()

        self._SNR = None

        if self.line_species in ['OII', 'OIII']:
            self._is_singlet = False
        else:
            self._is_singlet = True

        if self._is_singlet:
            self.GF = GaussFit(spec2D=self)
        else:
            self.GF = GaussFitDouble(spec2D=self)
    
    @property
    def pixScale(self):
        return self.spaceGrid[2]-self.spaceGrid[1]
    
    @property
    def nm_per_pixel(self):
        return self.lambdaGrid[2]-self.lambdaGrid[1]

    @property
    def ngrid_pos(self):
        return len(self.spaceGrid)

    @property
    def ngrid_spec(self):
        return len(self.lambdaGrid)

    @property
    def peak_flux(self):
        '''the peak flux (# photons) for each stripe of the spatial grid'''
        return np.amax(self.array, axis=1)

    @property
    def peak_id(self):
        '''ID in lambdaGrid where the peak flux is located, for each  spatial stripe'''
        return np.argmax(self.array, axis=1)

    @property
    def peak_lambda(self):
        '''the lambda position where the flux is peaked at, for each spatial stripe'''
        return self.lambdaGrid[self.peak_id]

    def auto_cutout(self):
        '''exclude spatial bins where the peak_flux(x) is negative'''
        id_x = np.where(self.peak_flux > 0.)[0]

        self.array = self.array[id_x, :]
        self.spaceGrid = self.spaceGrid[id_x]
        self.array_var = self.array_var[id_x, :]


    def cutout(self, xlim=None, lambdalim=None, thresholdSNR=0.):
        '''return a Spec2D object with smaller array size in given space limit and lambda limit
            Args:
                xlim = [-2.5, 2.5]
        '''

        if xlim is not None:
            id_x = np.where((self.spaceGrid >= xlim[0]) & (
                self.spaceGrid <= xlim[1]))[0]
        else:
            id_x = range(self.ngrid_pos)

        if lambdalim is not None:
            id_lambda = np.where((self.lambdaGrid >= lambdalim[0]) & (
                self.lambdaGrid <= lambdalim[1]))[0]
        else:
            id_lambda = range(self.ngrid_spec)

        if thresholdSNR > 0.:
            id_x_highSNR = np.where(self.SNR_pos >= thresholdSNR)[0]
            # take commond position IDs within xlim & highSNR
            id_x = list(set(id_x).intersection(set(id_x_highSNR)))
            id_x.sort()
        
        return Spec2D(self.array[id_x, :][:, id_lambda], self.array_var[id_x, :][:, id_lambda], self.spaceGrid[id_x], self.lambdaGrid[id_lambda], self.line_species, self.z)


    @property
    def SNR_pos(self):
        '''Compute the total SNR for each position stripe of the spec2D array'''

        if self._SNR is None:
            self._SNR = np.zeros(self.ngrid_pos)
            for j in range(self.ngrid_pos):
                self._SNR[j] = np.sqrt(np.sum(self.array[j]**2/self.array_var[j]))
    
        return self._SNR
    
    def _get_mesh(self, mode='corner'):
        '''generate coordiante mesh
            Args: 
                mode: 'corner' or 'center'
                    corner mode: mesh coordinate refers to the bottom left corner of each pixel grid
                    center mode: mesh coordinate refers to the center of each pixel grid
        '''
        if mode == 'corner':
            spaceGrid_plt = self.spaceGrid - self.pixScale/2.
            lambdaGrid_plt = self.lambdaGrid - self.nm_per_pixel/2.
            Xmesh, Lmesh = np.meshgrid(spaceGrid_plt, lambdaGrid_plt)
        else:
            Xmesh, Lmesh = np.meshgrid(self.spaceGrid, self.lambdaGrid)
        return Xmesh, Lmesh

    def display(self, xlim=None, ylim=None, vlim=None, filename=None, title='slit spectrum', mark_cen=True, 
                mark_peak=False, mark_fit=False, model=None):
        '''display the spec2D array'''

        fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
        plt.rc('font', size=16)

        if model is None:
            Xmesh, Lmesh = self._get_mesh(mode='corner')
            splot = ax.pcolormesh(Xmesh, Lmesh, self.array.T, cmap=plt.cm.jet, vmin = self.array.min(), vmax = self.array.max())
        else:
            Xmesh, Lmesh = self._get_mesh(mode='center')
            splot = ax.contourf(Xmesh, Lmesh, self.array.T, cmap=plt.cm.jet)

            mplot = ax.contour(Xmesh, Lmesh, model.T, levels=splot.levels, colors='yellow')
            ax.clabel(mplot, inline=1, fontsize=10)

        if mark_cen:
            ax.axvline(x=self.center[0], ls='--', color='lightgray', alpha=0.7)
            ax.axhline(y=self.center[1], ls='--', color='lightgray', alpha=0.7)

        ax.set_xlabel('x [arcsec]', fontsize=16)
        ax.set_ylabel('$\lambda$ [nm]', fontsize=16)
        ax.tick_params(labelsize=14)

        ax.set_title(title, fontsize=16)

        cbr = fig.colorbar(splot, ax=ax, pad=0.22)
        cbr.ax.tick_params(labelsize=14)

        if xlim is not None:
            ax.set_xlim((xlim[0], xlim[1]))
        else:
            ax.set_xlim((self.spaceGrid.min(), self.spaceGrid.max()))

        if ylim is not None:
            ax.set_ylim((ylim[0], ylim[1]))
        elif vlim is not None:
            ylim = velocity_to_lambda(v=vlim, lambda0=self.lambda0, z=self.z)
            ax.set_ylim((ylim[0], ylim[1]))
        else:
            ax.set_ylim((self.lambdaGrid.min(), self.lambdaGrid.max()))
        
        ax2 = ax.twinx()
        ax2.yaxis.set_label_position("right")
        _v_lim = lambda_to_velocity(ax.get_ylim(), lambda0=self.lambda0, z=self.z)
        ax2.set_ylim(_v_lim)
        ax2.tick_params(labelsize=14)
        ax2.set_ylabel('v [km/s]', fontsize=16)
        
        if mark_peak:
            ax.scatter(self.spaceGrid, self.peak_lambda, color='orange', s=3, alpha=0.6)
        
        if mark_fit:
            cenLambda, amp, sigma = self.GF.fit_spec2D()

            if self._is_singlet:
                ax.errorbar(self.spaceGrid[1::3], cenLambda[1::3], sigma[1::3], color='dimgray',
                        marker='o', markersize=2, ls='none', label='gaussFit summary')
            else:
                peakLambda_lo = self.GF.cal_peak_lambda(cenLambda, mode='lo')
                peakLambda_up = self.GF.cal_peak_lambda(cenLambda, mode='up')
                ax.errorbar(self.spaceGrid[1::3], peakLambda_lo[1::3], sigma[1::3], color='dimgray',
                            marker='o', markersize=2, ls='none', label='gaussFit summary')
                ax.errorbar(self.spaceGrid[2::3], peakLambda_up[2::3], sigma[2::3], color='dimgray',
                            marker='o', markersize=2, ls='none')
                ax.plot(self.spaceGrid, cenLambda, ls='-.', color='lightgray')

            #ax.legend(loc="best", prop={'size': 12})

        if filename is not None:
            fig.savefig(filename, bbox_inches='tight')
        else:
            fig.tight_layout()
            fig.show()
        
        return fig, ax

if __name__ == '__main__':

    import sys
    import pathlib
    dir_binnedFit = str(pathlib.Path(__file__).parent.absolute())
    sys.path.append(dir_binnedFit+'/tests')
    from get_pars import get_pars0

    from gen_mocks import gen_mock_tfCube

    pars, line_species = get_pars0()
    dataInfo = gen_mock_tfCube(pars, line_species, slits='major', noise_mode=0)

    # ========= example inputs to construct a Spec2D obj. ========= #

    spec_array = dataInfo['spec'][0].array
    spec_var = dataInfo['spec_variance'][0]
    spaceGrid = dataInfo['spaceGrid']
    lambdaGrid = dataInfo['lambdaGrid']
    
    # ========= create Spec2D object ========= #

    spec2D = Spec2D(array=spec_array, spaceGrid=spaceGrid, lambdaGrid=lambdaGrid, line_species=line_species, z=dataInfo['par_fid']['redshift'], array_var=spec_var, auto_cut=True)

    fig, ax=spec2D.display(mark_peak=True, xlim=[-2.5, 2.5])
    ax.axvline(x=-1.5)
    fig.show()
