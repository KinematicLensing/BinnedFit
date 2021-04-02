import numpy as np
import astropy.units as u
import sys
import pathlib
dir_repo = str(pathlib.Path(__file__).parent.absolute()) + '/../..'
dir_KLens = dir_repo + '/KLens'
sys.path.append(dir_KLens)

from tfCube2 import Parameters


def get_pars0():
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
    pars['ngrid'] = 128

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
    linelist['flux'] = refSDSSspec.value / \
        fiber_SDSS  # [unit: erg/s/cm2/nm/arcsec2]
    pars['linelist'] = linelist

    return pars, line_species


if __name__ == '__main__':
    pars, line_species = get_pars0()
