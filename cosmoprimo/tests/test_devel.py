import os
import tempfile

import pytest
import numpy as np

from cosmoprimo import (Cosmology, Background, Thermodynamics, Primordial,
                        Harmonic, Fourier, CosmologyError, CosmologyInputError, CosmologyComputationError,
                        constants)

def test_params():
    from cosmoprimo.cosmology import BaseCosmoParams
    default_params = BaseCosmoParams().get_default_params(of='cosmology')
    cosmo = Cosmology()
    with pytest.raises(CosmologyError):
        cosmo = Cosmology(sigma8=1., A_s=1e-9)
    params = {'Omega_cdm': 0.3, 'Omega_b': 0.02, 'h': 0.8, 'n_s': 0.96}
    cosmo = Cosmology(**params)
    assert cosmo['omega_cdm'] == 0.3 * 0.8**2
    assert len(cosmo['z_pk']) == 30
    assert cosmo['sigma8'] == 0.8
    for neutrino_hierarchy in ['normal', 'inverted', 'degenerate']:
        cosmo = Cosmology(m_ncdm=0.1, neutrino_hierarchy=neutrino_hierarchy)
        assert len(cosmo['m_ncdm']) == 3
        assert np.allclose(sum(cosmo['m_ncdm']), 0.1)
        assert np.allclose(cosmo['m_ncdm_tot'], 0.1)
    m_ncdm = [0.01, 0.02, 0.05]
    cosmo = Cosmology(m_ncdm=m_ncdm)
    Background(cosmo, engine='class')
    Fourier(cosmo)

    with tempfile.TemporaryDirectory() as tmp_dir:
        fn = os.path.join(tmp_dir, 'cosmo.json')
        cosmo.write(fn)
        cosmo = Cosmology.read(fn)

    assert np.allclose(cosmo['m_ncdm'], m_ncdm)
    assert cosmo.engine.__class__.__name__ == 'ClassEngine'
    Fourier(cosmo)

    with pytest.raises(CosmologyInputError):
        cosmo = Cosmology(tau=0.05, tau_reio=0.06)
    cosmo = Cosmology(ombh2=0.05, omch2=0.1)
    assert np.allclose(cosmo['omega_b'], 0.05) and np.allclose(cosmo['omega_cdm'], 0.1)
    tf = Cosmology(engine='eisenstein_hu_nowiggle_variants',Omega_ncdm = 0.2).get_transfer()
    k = np.array([5.])
    with pytest.raises(CosmologyError):
        tf.transfer_kz(k, z=0., of='delta_ncdm')


if __name__ == '__main__':
    test_params()