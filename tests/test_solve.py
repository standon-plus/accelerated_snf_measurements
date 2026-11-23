import numpy as np
from snf.solve import omp, lstsq

def test_omp_recovers_sparse():
    rng = np.random.default_rng(0)
    M, K, s = 120, 400, 10
    Psi = rng.normal(size=(M, K)) + 1j * rng.normal(size=(M, K))
    x = np.zeros(K, complex)
    x[rng.choice(K, s, replace=False)] = rng.normal(size=s) + 1j * rng.normal(size=s)
    assert np.allclose(omp(Psi, Psi @ x, tol=1e-10), x)

def test_lstsq_overdetermined():
    rng = np.random.default_rng(1)
    Psi = rng.normal(size=(50, 20)) + 1j * rng.normal(size=(50, 20))
    x = rng.normal(size=20) + 0j
    assert np.allclose(lstsq(Psi, Psi @ x), x)
