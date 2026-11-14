import numpy as np
import pandas as pd
from scipy.constants import mu_0, c
from scipy.special import spherical_jn, spherical_yn
from snf.swe import j_index, wigner_d_rows

Z0 = mu_0 * c

def load_far_field(path): # HFSS far field export -> theta (n_th), phi (n_ph), rEtheta, rEphi (n_th, n_ph). Duplicate phi endpoint dropped
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    col = lambda prefix: [c for c in df.columns if c.startswith(prefix)][0]
    th, ph = df[col('Theta')].to_numpy(float), df[col('Phi')].to_numpy(float)
    if 'deg' in col('Theta'):
        th, ph = np.radians(th), np.radians(ph)
    order = np.lexsort((ph, th))  # theta slow, phi fast
    mag_ang = lambda name: (lambda a: a[:, 0] * np.exp(1j * a[:, 1]))(df[col(name)].str.split(expand=True).astype(float).to_numpy()[order])
    th_1d, ph_1d = np.unique(th), np.unique(ph)
    Eth = mag_ang('rEtheta').reshape(th_1d.size, ph_1d.size)
    Eph = mag_ang('rEphi').reshape(th_1d.size, ph_1d.size)
    if np.isclose(ph_1d[-1] - ph_1d[0], 2 * np.pi):
        ph_1d, Eth, Eph = ph_1d[:-1], Eth[:, :-1], Eph[:, :-1]
    return th_1d, ph_1d, Eth, Eph

def theta_weights(theta):
    '''
    Quadrature weights for integrating f(theta) sin(theta) dtheta on an equiangular grid with both poles (Clenshaw-Curtis in cos theta).
    Exact for polynomials in cos theta up to degree len(theta) - 1.
    '''
    k = np.arange(len(theta))
    A = np.cos(np.outer(k, theta))
    b = np.where(k == 1, 0.0, (1 + np.cos(k * np.pi)) / (1 - np.where(k == 1, 0, k)**2))
    return np.linalg.solve(A, b)

def sw_functions(N, theta, phi, kr=None):
    '''
    Hansen spherical wave functions F^(3)_smn, or far-field functions K_smn if kr is None.
    Built from Hansen's Wigner d (mu = -1, 0, 1) so the convention matches swe.psi.

    theta, phi (np.array): Dimensions (I)
    kr (np.array): wavenumber times radius. Dimensions (I)

    Yields (n, j1, j2, F1, F2) per n; F dimensions (I, 2n+1, 3) in (r, theta, phi) components; j1, j2 are Hansen indices of s=1,2
    '''
    d = wigner_d_rows(N, np.array([-1, 0, 1]), theta) # (N, I, 3, 2n+1)
    for n in range(1, N + 1):
        ms = np.arange(-n, n + 1)
        sn = np.sqrt((2 * n + 1) / 2) * np.sqrt(n * (n + 1)) / 2
        A = -sn * (d[n-1][:, 0] + d[n-1][:, 2])  # (-m/|m|)^m m Pbar / sin(theta)
        B = sn * (d[n-1][:, 0] - d[n-1][:, 2])   # (-m/|m|)^m dPbar / dtheta
        P0 = np.sqrt((2 * n + 1) / 2) * d[n-1][:, 1] # (-m/|m|)^m Pbar
        e = np.exp(1j * np.outer(phi, ms)) / np.sqrt(2 * np.pi * n * (n + 1))
        if kr is None:
            r1, r2, rr = (-1j)**(n + 1), (-1j)**n, 0
        else:
            h = spherical_jn(n, kr) + 1j * spherical_yn(n, kr)
            h_prev = spherical_jn(n - 1, kr) + 1j * spherical_yn(n - 1, kr)
            r1, r2, rr = h[:, None], (h_prev - n * h / kr)[:, None], (n * (n + 1) * h / kr)[:, None]
        F1 = np.stack([np.zeros_like(A), 1j * A, -B], axis=-1) * (r1 * e)[..., None]
        F2 = np.stack([rr * P0, r2 * B, r2 * 1j * A], axis=-1) * e[..., None]
        yield n, j_index(1, ms, n), j_index(2, ms, n), F1, F2

def ff_to_T(theta, phi, Eth, Eph, N, P_in=1.0, conj=True):
    '''
    Project a far field onto Hansen transmission coefficients T_smn (per unit incident amplitude, |v|^2 / 2 = P_in).
    rE e^{-ikr} = sqrt(Z0) sum Q_smn K_smn, Q = sqrt(2 P_in) T.

    theta (np.array): equiangular, including both poles. Dimensions (n_th)
    phi (np.array): uniform over [0, 2pi), no duplicate endpoint. Dimensions (n_ph)
    Eth, Eph (np.array): r E_theta, r E_phi. Dimensions (n_th, n_ph)
    conj (bool): conjugate first (HFSS e^{+jwt} -> Hansen e^{-iwt})

    Returns T; dimensions 2N(N+2)
    '''
    if conj:
        Eth, Eph = Eth.conj(), Eph.conj()
    w = theta_weights(theta) * 2 * np.pi
    T = np.zeros(2 * N * (N + 2), complex)
    for n, j1, j2, K1, K2 in sw_functions(N, theta, np.zeros_like(theta)): # K(theta, 0); phi dependence is e^{im phi}
        ms = np.arange(-n, n + 1)
        e = np.exp(-1j * np.outer(phi, ms)) / len(phi) # phi average of E e^{-im phi}
        Em_th, Em_ph = Eth @ e, Eph @ e # (n_th, 2n+1)
        for j, K in ((j1, K1), (j2, K2)):
            T[j] = w @ (Em_th * K[..., 1].conj() + Em_ph * K[..., 2].conj())
    return T / np.sqrt(Z0) / np.sqrt(2 * P_in)

def T_to_ff(T, theta, phi, P_in=1.0):
    '''
    Far field r E e^{-ikr} (Hansen convention) from T at points (theta, phi). Dimensions (I)

    Returns Eth, Eph; dimensions (I)
    '''
    N = int(round(np.sqrt(len(T) / 2 + 1))) - 1
    E = np.zeros((len(theta), 3), complex)
    for n, j1, j2, K1, K2 in sw_functions(N, theta, phi):
        E += np.einsum('j,ijc->ic', T[j1], K1) + np.einsum('j,ijc->ic', T[j2], K2)
    E *= np.sqrt(Z0) * np.sqrt(2 * P_in)
    return E[:, 1], E[:, 2]

def nf_from_T(T, xyz, k, origin=np.zeros(3), P_in=1.0):
    '''
    E and H (Hansen convention, Cartesian) radiated by T at points xyz, with the expansion centered at origin.
    E = k sqrt(Z0) sum Q F_smn, H = -i k / sqrt(Z0) sum Q F_(3-s)mn.

    xyz (np.array): Dimensions (I, 3)
    k (float): wavenumber

    Returns E, H; dimensions (I, 3)
    '''
    N = int(round(np.sqrt(len(T) / 2 + 1))) - 1
    p = xyz - origin
    r = np.linalg.norm(p, axis=1)
    th, ph = np.arccos(np.clip(p[:, 2] / r, -1, 1)), np.arctan2(p[:, 1], p[:, 0])
    E, H = np.zeros((len(r), 3), complex), np.zeros((len(r), 3), complex)
    for n, j1, j2, F1, F2 in sw_functions(N, th, ph, k * r):
        E += np.einsum('j,ijc->ic', T[j1], F1) + np.einsum('j,ijc->ic', T[j2], F2)
        H += np.einsum('j,ijc->ic', T[j1], F2) + np.einsum('j,ijc->ic', T[j2], F1)
    Q = np.sqrt(2 * P_in)
    E, H = k * np.sqrt(Z0) * Q * E, -1j * k / np.sqrt(Z0) * Q * H
    st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
    basis = np.stack([np.column_stack([st * cp, st * sp, ct]),          # r hat
                      np.column_stack([ct * cp, ct * sp, -st]),         # theta hat
                      np.column_stack([-sp, cp, np.zeros_like(ph)])], axis=1) # phi hat; (I, 3, 3)
    return np.einsum('ic,icx->ix', E, basis), np.einsum('ic,icx->ix', H, basis)

def T_to_R(T):
    '''Receive coefficients of a reciprocal antenna from its transmission coefficients: R_smn = (-1)^m T_s,-m,n.'''
    N = int(round(np.sqrt(len(T) / 2 + 1))) - 1
    R = np.zeros_like(T)
    for n in range(1, N + 1):
        ms = np.arange(-n, n + 1)
        for s in (1, 2):
            R[j_index(s, ms, n)] = (-1.0)**ms * T[j_index(s, -ms, n)]
    return R
