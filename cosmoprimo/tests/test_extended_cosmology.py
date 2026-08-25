import os
import tempfile

import pytest
import numpy as np

from cosmoprimo import (Cosmology, Background, Thermodynamics, Primordial,
                        Harmonic, Fourier, CosmologyError, CosmologyInputError, CosmologyComputationError,
                        constants)



def test_isitgr(plot=False):
    cosmo_camb = Cosmology(engine='camb')
    try:
        cosmo = Cosmology(engine='isitgr')
    except ImportError:
        return

    k = np.linspace(0.01, 1., 200)
    z = np.linspace(0., 2., 10)
    assert np.allclose(cosmo_camb.get_fourier().pk_interpolator()(k=k, z=z), cosmo.get_fourier().pk_interpolator()(k=k, z=z), atol=0., rtol=5e-3)

    cosmo = Cosmology(engine='isitgr', MG_parameterization='mueta', E11=-0.5, E22=-0.5, extra_params=dict(AccuracyBoost=1.1))
    assert not np.allclose(cosmo_camb.get_fourier().pk_interpolator()(k=k, z=z), cosmo.get_fourier().pk_interpolator()(k=k, z=z), atol=0., rtol=5e-3)
    cosmo.comoving_radial_distance(z)

    from cosmoprimo.fiducial import DESI
    cosmo = DESI(engine='isitgr')

    if plot:
        z = 1.
        k = np.linspace(0.001, 0.2, 100)
        from matplotlib import pyplot as plt
        ax = plt.gca()
        for kwargs in [{}, {'mu0': -0.5, 'Sigma0': 0.}, {'mu0': -0.5, 'Sigma0': 1.}]:
            pk = Cosmology(engine='isitgr', MG_parameterization='muSigma', **kwargs).get_fourier().pk_interpolator(of='delta_cb').to_1d(z=z)
            #ax.plot(k,  k * pk(k), label=str(kwargs))
            k = pk.k
            ax.loglog(k,  pk(k), label=str(kwargs))
        ax.legend()
        plt.show()


def test_mgcamb(plot=False):
    cosmo_camb = Cosmology(engine='camb')
    try:
        cosmo = Cosmology(engine='mgcamb')
    except ImportError:
        return

    k = np.linspace(0.01, 1., 200)
    z = np.linspace(0., 2., 10)
    assert np.allclose(cosmo_camb.get_fourier().pk_interpolator()(k=k, z=z), cosmo.get_fourier().pk_interpolator()(k=k, z=z), atol=0., rtol=5e-3)

    cosmo = Cosmology(engine='mgcamb', MG_flag=1, **{'mu0': -0.5, 'sigma0': 1.}, extra_params=dict(AccuracyBoost=1.1))
    assert not np.allclose(cosmo_camb.get_fourier().pk_interpolator()(k=k, z=z), cosmo.get_fourier().pk_interpolator()(k=k, z=z), atol=0., rtol=5e-3)
    cosmo.comoving_radial_distance(z)

    from cosmoprimo.fiducial import DESI
    cosmo = DESI(engine='mgcamb')

    if plot:
        z = 1.
        k = np.linspace(0.001, 0.2, 100)
        from matplotlib import pyplot as plt
        ax = plt.gca()
        for kwargs in [{}, {'mu0': -0.5, 'sigma0': 0.}, {'mu0': -0.5, 'sigma0': 1.}]:
            pk = Cosmology(engine='mgcamb', MG_flag=1, **kwargs).get_fourier().pk_interpolator(of='delta_cb').to_1d(z=z)
            #ax.plot(k,  k * pk(k), label=str(kwargs))
            k = pk.k
            ax.loglog(k,  pk(k), label=str(kwargs))
        ax.legend()
        plt.show()


def test_axiclass():

    cosmo_class = Cosmology(engine='class')
    try:
        cosmo = Cosmology(engine='axiclass')
    except ImportError:
        return

    k = np.linspace(0.01, 1., 200)
    z = np.linspace(0., 2., 10)
    assert np.allclose(cosmo_class.get_fourier().pk_interpolator()(k=k, z=z), cosmo.get_fourier().pk_interpolator()(k=k, z=z), atol=0., rtol=1e-4)

    params = {'scf_potential': 'axion', 'n_axion': 2.6, 'log10_axion_ac': -3.531, 'fraction_axion_ac': 0.1, 'scf_parameters': [2.72, 0.0], 'scf_evolve_as_fluid': False,
              'scf_evolve_like_axionCAMB': False, 'attractor_ic_scf': False, 'compute_phase_shift': False, 'include_scf_in_delta_m': True, 'include_scf_in_delta_cb': True}
    cosmo = Cosmology(engine='axiclass', **params)
    assert not np.allclose(cosmo_class.get_fourier().pk_interpolator()(k=k, z=z), cosmo.get_fourier().pk_interpolator()(k=k, z=z), atol=0., rtol=1e-4)
    cosmo.comoving_radial_distance(z)

    from cosmoprimo.fiducial import DESI
    cosmo = DESI(engine='axiclass', **params)
    cosmo['log10_axion_ac']


def test_mochiclass():
    cosmo_class = Cosmology(engine='class')
    try:
        cosmo = Cosmology(engine='mochiclass')
    except ImportError:
        return

    k = np.linspace(0.01, 1., 200)
    z = np.linspace(0., 2., 10)
    assert np.allclose(cosmo_class.get_fourier().pk_interpolator()(k=k, z=z), cosmo.get_fourier().pk_interpolator()(k=k, z=z), atol=0., rtol=1e-4)

    params = {'Omega_Lambda': 0, 'Omega_fld': 0, 'Omega_smg': -1, 'gravity_model': 'brans dicke', 'parameters_smg': [0.7, 50, 1., 1.e-1],
              'skip_stability_tests_smg': 'no', 'a_min_stability_test_smg': 1e-6}
    cosmo = Cosmology(engine='mochiclass', **params)
    assert not np.allclose(cosmo_class.get_fourier().pk_interpolator(of='theta_cb')(k=k, z=z), cosmo.get_fourier().pk_interpolator(of='theta_cb')(k=k, z=z), atol=0., rtol=1e-4)
    cosmo.comoving_radial_distance(z)

    from cosmoprimo.fiducial import DESI
    cosmo = DESI(engine='mochiclass', **params)
    cosmo['parameters_smg']
    cosmo.get_fourier().pk_interpolator(of='theta_cb')


def test_negnuclass():
    cosmo_class = Cosmology(engine='class')
    try:
        cosmo = Cosmology(engine='negnuclass')
    except ImportError:
        return

    k = np.linspace(0.01, 1., 200)
    z = np.linspace(0., 2., 10)
    assert np.allclose(cosmo_class.get_fourier().pk_interpolator()(k=k, z=z), cosmo.get_fourier().pk_interpolator()(k=k, z=z), atol=0., rtol=1e-4)

    params = {'m_ncdm': -0.4}
    cosmo = Cosmology(engine='negnuclass', **params)
    assert not np.allclose(cosmo_class.get_fourier().pk_interpolator(of='theta_cb')(k=k, z=z), cosmo.get_fourier().pk_interpolator(of='theta_cb')(k=k, z=z), atol=0., rtol=1e-4)
    cosmo.comoving_radial_distance(z)

    from cosmoprimo.fiducial import DESI
    from matplotlib import pyplot as plt
    ax = plt.gca()
    for m_ncdm in [-0.04, 0.06]:
        params.update(m_ncdm=m_ncdm)
        cosmo = DESI(engine='negnuclass', **params)
        pk = cosmo.get_fourier().pk_interpolator(of='theta_cb')
        ax.loglog(pk.k, pk(pk.k, z=1.))
    plt.show()

def test_decnuclass():
    cosmo_class = Cosmology(engine='class')
    try:
        cosmo = Cosmology(engine='decnuclass')
    except ImportError:
        return

    k = np.linspace(0.01, 1., 200)
    z = np.linspace(0., 2., 10)
    assert np.allclose(cosmo_class.get_fourier().pk_interpolator()(k=k, z=z), cosmo.get_fourier().pk_interpolator()(k=k, z=z), atol=0., rtol=1e-4)

    params = {'m_ncdm': 0.4, 'Gamma_ncdm': 1e2}
    cosmo = Cosmology(engine='decnuclass', **params)
    assert not np.allclose(cosmo_class.get_fourier().pk_interpolator(of='theta_cb')(k=k, z=z), cosmo.get_fourier().pk_interpolator(of='theta_cb')(k=k, z=z), atol=0., rtol=1e-4)
    cosmo.comoving_radial_distance(z)

    from cosmoprimo.fiducial import DESI
    from matplotlib import pyplot as plt
    ax = plt.gca()
    for Gamma_ncdm in [1e2,1e3,1e4]:
        params.update(m_ncdm=m_ncdm)
        cosmo = DESI(engine='decnuclass', **params)
        pk = cosmo.get_fourier().pk_interpolator(of='theta_cb')
        ax.loglog(pk.k, pk(pk.k, z=1.))
    plt.show()


if __name__ == "__main__":
    test_isitgr()
    test_mgcamb()
    test_axiclass()
    test_mochiclass()
    test_negnuclass()
    test_decnuclass()
    