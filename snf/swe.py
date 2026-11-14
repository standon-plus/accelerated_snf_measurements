import spherical
from math import factorial as fact
import numpy as np
from scipy.special import spherical_jn, spherical_yn

def j_index(s, m, n):
    '''Zero-based Hansen index for mode (s, m, n).'''
    return 2 * (n * (n + 1) + m - 1) + s - 1

def wigner_d_rows(N, mus, theta):
    '''
    Calculate small Wigner-d d^n_{mu m}(theta) in Hansen's (Edmonds) convention for all N.

    N (int): Total number of polar modes
    mus (np.array): Probe azimuthal index (typically +/- 1 for ideal probe). Dimensions (Mu)
    theta (np.array): Euler angle for rotation. Dimensions (I)

    Returns d; dimensions (N, I, Mu, 2n+1)
    '''
    th_unique, inv = np.unique(np.round(theta, 12), return_inverse=True)
    wigner = spherical.Wigner(N)
    # spherical's (mu, m) entry is Hansen's (m, mu), so swap the order
    idx = [np.array([[wigner.dindex(n, m, mu) for m in range(-n, n+1)] for mu in mus]) for n in range(1, N + 1)]
    d = [np.empty((len(th_unique), len(mus), 2 * n + 1)) for n in range(1, N + 1)]
    for k, t in enumerate(th_unique):
        d_vals = wigner.d(np.exp(1j*t))
        for n in range(1, N + 1):
            d[n - 1][k] = d_vals[idx[n - 1]]
    return [d_n[inv] for d_n in d]

def z3(p, kA):
    return spherical_jn(p, kA) + 1j * spherical_yn(p, kA)

def C(s, n, sigma, mu, v, kA):
    '''
    Return translation coefficient for given parameters. 

    s (int): polarization index
    n (int): polar mode index
    sigma (int): probe polarization index
    mu (int): probe azimuthal index
    v (int): probe polar index
    kA (float): wavenumber times distance probe is from AUT
    '''
    prefactor = np.sqrt((2 * n + 1)*(2 * v + 1)/(n * (n + 1) * v * (v + 1))) * ((-1)**int(mu)) * (1j**(n - v)) / 2
    tot = 0j
    for p in range(abs(n - v), n + v + 1, 2):
        a = (2 * p + 1) * spherical.Wigner3j(n, v, p, 0, 0, 0) * spherical.Wigner3j(n, v, p, mu, -mu, 0)
        if sigma == s:
            b = n * (n + 1) + v * (v + 1) - p * (p + 1)
        else:
            b = 2j * mu * kA
        tot += 1j**(-p) * b * a * z3(p, kA)
    return prefactor * tot

def probe_response(N, kA, R, Np=1, mus=np.array((-1, 1)),):
    '''
    Calculate the probe response constants. 

    N (int): Total number of polar modes.
    kA (float): Wavenumber times distance probe is from AUT
    R (np.array): Probe recieve coefficients. Dimensions 2Np(Np + 2)
    Np (int): Total number of polar modes for the probe.
    mus (np.array): Probe azimuthal index (typically +/- 1 for ideal probe). Dimensions (Mu)

    Returns P; dimensions (2, Mu, N)
    '''
    P = np.zeros((2, len(mus), N), dtype=np.complex128)
    for i_mu, mu in enumerate(mus):
        for n in range(max(1, abs(mu)), N+1):
            for s in (1, 2):
                P[s-1, i_mu, n-1] = sum(C(s, n, sigma, mu, v, kA) * R[j_index(sigma, mu, v)] for sigma in (1, 2) for v in range(max(1, abs(mu)), Np + 1))
    return P / 2

def psi(chi, theta, phi, R, N=10, kA=25, Np=1, mus=np.array((-1, 1)), weighted=False):
    '''
    Create the psi matrix used in the linear problem to calculate spherical near-field coefficients. 

    chi, theta, phi (np.array): Euler angles at which probe measurements are taken. Dimensions (I)
    R (np.array): Probe recieve coefficients, arranged as in Hansen formulation. Dimensions 2Np(Np+2)
    N (int): Total number of polar modes. 
    kA (float): Wavenumber times distance probe is from AUT
    Np (int): Polar modes of probe. Ideal probe has Np=1
    mus (np.array): Probe azimuthal index (typically +/- 1 for ideal probe). Dimensions (Mu)
    weighted (bool): Weight rows according to sqrt(sin theta) as done in Valdez NIST paper. When using this, ensure multiply w by weights before solving.

    Returns Psi; dimensions (I, 2N(N+2))
    '''
    d = wigner_d_rows(N, mus, theta) # (N, I, Mu, 2n+1)
    P = probe_response(N, kA, R, Np, mus) # (2, Mu, N)
    Psi = np.empty((len(theta), 2 * N * (N + 2)), dtype=np.complex128)

    e_mu = np.exp(1j * np.outer(chi, mus)) # (I, Mu)
    for n in range(1, N + 1):
        ms = np.arange(-n, n + 1) # (2n + 1)
        e_m = np.exp(1j * np.outer(phi, ms)) # (I, 2n+1)
        for s in (1, 2):
            inner = np.einsum('iu,ium,u->im', e_mu, d[n-1], P[s-1, :, n-1]) # (I, 2n+1)
            j = j_index(s, ms, n) # (2n + 1)
            Psi[:, j] = inner * e_m
    sw = np.sqrt(np.abs(np.sin(theta))) if weighted else np.ones(len(theta))
    Psi *= sw[:, None]
    return (Psi, sw) if weighted else Psi
