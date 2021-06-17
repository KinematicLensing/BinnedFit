# BinnedFit
A fitting pipeline to extract gravational lensing signals utilizing galaxy's morphology (photometric) and kinematic (spectroscopic) information. The theoritical formalism of this <b>kinematic lensing</b> idea is presented in [arXiv:1311.1489](https://arxiv.org/abs/1311.1489v2).



## Requirements

This package depends on the following Python libraries:

* [KLens](https://github.com/emhuff/KLens.git)
* [GalSim](https://github.com/GalSim-developers/GalSim)
* [astropy](https://www.astropy.org)
* [emcee](https://emcee.readthedocs.io/en/stable/)
* [ChainConsumer](https://samreay.github.io/ChainConsumer/index.html)

## Overview of this repository

1. The galaxy slit spectra data are stored in the data class <b>Spec2D</b> defined in [spec2D.py](https://github.com/hungjinh/BinnedFit/blob/master/spec2D.py). <b>Spec2D</b> provides many useful methods to display the spectra data, and to easily overplot data with model for visualization. [This notebook](https://github.com/hungjinh/BinnedFit/blob/master/notebook/%5Bdemo%5D%20Spec2D%20(tfCube%20mock).ipynb) demonstrates how <b>Spec2D</b> works. 


2. Models to fit galaxy images are managed by the <b>ImageFit</b> class defined in [imageFit.py](https://github.com/hungjinh/BinnedFit/blob/master/imageFit.py). A demo for fitting a galaxy image with ImageFit is available at [this notebook](https://github.com/hungjinh/BinnedFit/blob/master/notebook/%5Bdemo%5D%20ImageFit.ipynb).
   
    
3. A simple strategy to model the galaxy slit spectrum is to fit a single Gaussian profile $G_i$($\lambda$ | $A_i$, $\sigma_i$, $\mu_i$) = $\frac{A_i}{2 \pi \sigma_i^2}$ exp($\frac{-(\lambda-\mu_i)^2}{2 \sigma_i^2}$) across each positional bin $i$ of a slit spectrum. The best-fitted $\mu_i$ specifies the peak $\lambda$ value of bin $i$, and $\sigma_i$ is the 1$\sigma$ scatter of line width at bin $i$. This positional Gaussian-fit procedure is managed by the <b>GaussFit</b> class defined in [gaussFit.py](https://github.com/hungjinh/BinnedFit/blob/master/gaussFit.py). For a spectral data array with $N$ position bins, <b>GaussFit</b> essentially uses a total of 3$\times$$N$ degrees of freedom to model the spectrum array ($A_1$, $A_2$, ... $A_N$), ($\sigma_1$, $\sigma_2$, ... $\sigma_N$), ($\mu_1$, $\mu_2$, ... $\mu_N$). 
   
   We also have the <b>GaussFitDouble</b> class to handle doublet emission lines with double Gaussian profiles. An example notebook for the usage of <b>GaussFit</b> and <b>GaussFitDouble</b> is available [here](https://github.com/hungjinh/BinnedFit/blob/master/notebook/%5Bdemo%5D%20GaussFit%2C%20GaussFitDouble.ipynb).

 
4. Typically, the rotation curve from an observed galaxy slit spectrum can be well-described via an arctan function. We can then use this contraint to decrease the degrees of freedom (d.o.f) used in <b>GaussFit</b> to model the spectra. i.e. replacing the d.o.f. provided in ($\mu_1$, $\mu_2$, ... $\mu_N$) with few parameters that specify an arctan rotaion curve. This d.o.f reduced model is coded up in the <b>RotFitSingle</b> and <b>RotFitDouble</b> classes (for siglet and doublet emission lines respectively) in [rotCurveFit.py](https://github.com/hungjinh/BinnedFit/blob/master/rotCurveFit.py). This is essentially how binnedFit deals with the spectra data. [This notebook](https://github.com/hungjinh/BinnedFit/blob/master/notebook/%5Bdemo%5D%20RotFitSingle%2C%20RotFitDouble.ipynb) demonstrates how to apply <b>RotFitSingle</b> and <b>RotFitDouble</b> on mock data.


5. Finally, connecting all the above modeling pieces together to perform parameter inferences on galaxy image+spectra data, we have the <b>GammaInference</b> class defined in [gamma.py](https://github.com/hungjinh/BinnedFit/blob/master/gamma.py) to manage the overall MCMC sampling process. An example tutorial on how to run the fitting pipeline is demonstrated in [this notebook](https://github.com/hungjinh/BinnedFit/blob/master/notebook/%5Bdemo%5D%20GammaInference.ipynb). 

6. To analyze the MCMC chains generated from the output of <b>GammaInference</b>, one can find methods provided in the <b>ChainTool</b> class written in [chainTool.py](https://github.com/hungjinh/BinnedFit/blob/master/chainTool.py). Check [this notebook](https://github.com/hungjinh/BinnedFit/blob/master/notebook/%5Bcheck%5D%20chain%20result%20(single).ipynb) for the usage of <b>ChainTool</b>, and [this notebook](https://github.com/hungjinh/BinnedFit/blob/master/notebook/%5Bcheck%5D%20chain%20results%20(multi-chains).ipynb) for displaying multiple chains.  

