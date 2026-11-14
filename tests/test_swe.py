import numpy as np
from math import factorial as fact, comb
from snf.swe import wigner_d_rows

def d_hansen(n, mu, m, theta): # Explicit Edmonds sum used by Hansen
    pre = np.sqrt(fact(n + mu) * fact(n - mu) / (fact(n + m) * fact(n - m)))
    tot = 0
    for sigma in range(max(0, -mu - m), min(n - mu, n - m) + 1):
        tot += (comb(n + m, n - mu - sigma) * comb(n - m, sigma) * (-1)**(n - mu - sigma)
                * np.cos(theta / 2)**(2 * sigma + mu + m) * np.sin(theta / 2)**(2 * n - 2 * sigma - mu - m))
    return pre * tot

def test_matches_hansen():
    N, theta = 6, np.array([0.3, 1.2, 2.9, -0.7, 4.0])
    for mus in (np.array([-1, 1]), np.array([0])):
        d = wigner_d_rows(N, mus, theta)
        for n in range(1, N + 1):
            for i, t in enumerate(theta):
                for k, mu in enumerate(mus):
                    ref = [d_hansen(n, mu, m, t) for m in range(-n, n + 1)]
                    assert np.allclose(d[n - 1][i, k], ref, atol=1e-12)

def test_identity_and_orthogonal():
    N = 5
    for n in range(1, N + 1):
        mus = np.arange(-n, n + 1)
        D0 = wigner_d_rows(n, mus, np.array([0.0]))[n - 1][0]
        D = wigner_d_rows(n, mus, np.array([1.1]))[n - 1][0]
        assert np.allclose(D0, np.eye(2 * n + 1))
        assert np.allclose(D @ D.T, np.eye(2 * n + 1))
