import numpy as np
from scipy.special import assoc_legendre_p_all
from snf.farfield import Z0, theta_weights, sw_functions, ff_to_T, T_to_ff, nf_from_T, T_to_R
from snf.swe import j_index

theta = np.radians(np.arange(0, 181)) # HFSS far field grid
phi = np.radians(np.arange(0, 360))
TH, PH = (a.flatten() for a in np.meshgrid(theta, phi, indexing='ij'))

def random_T(N, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=2 * N * (N + 2)) + 1j * rng.normal(size=2 * N * (N + 2))

def test_theta_weights():
    w = theta_weights(theta)
    for p in range(0, 150, 7):
        assert np.isclose(w @ np.cos(theta)**p, (1 + (-1)**p) / (p + 1))

def test_orthonormal():
    N = 12
    K = np.zeros((len(TH), 2 * N * (N + 2), 3), complex)
    for n, j1, j2, K1, K2 in sw_functions(N, TH, PH):
        K[:, j1], K[:, j2] = K1, K2
    w = np.repeat(theta_weights(theta), len(phi)) * 2 * np.pi / len(phi)
    G = np.einsum('i,iac,ibc->ab', w, K, K.conj())
    assert np.allclose(G, np.eye(G.shape[0]), atol=1e-10)

def test_round_trip_and_power():
    N = 40
    T = random_T(N)
    Eth, Eph = T_to_ff(T, TH, PH)
    Eth, Eph = Eth.reshape(len(theta), len(phi)), Eph.reshape(len(theta), len(phi))
    T2 = ff_to_T(theta, phi, Eth, Eph, N, conj=False)
    assert np.allclose(T2, T, atol=1e-10)
    P = theta_weights(theta) @ np.mean(np.abs(Eth)**2 + np.abs(Eph)**2, axis=1) * 2 * np.pi / (2 * Z0)
    assert np.isclose(P, np.sum(np.abs(T)**2)) # P_in = 1 W: radiated = P_in sum |T|^2

def test_K_matches_legendre(): # Hansen's explicit K_smn from normalized Legendre functions (no Condon-Shortley phase)
    N, th, ph = 6, np.linspace(0.1, 3.0, 20), np.linspace(-2, 3, 20)
    P, dP = assoc_legendre_p_all(N, N, np.cos(th), norm=True, diff_n=1)
    for n, j1, j2, K1, K2 in sw_functions(N, th, ph):
        for i, m in enumerate(range(-n, n + 1)):
            f = (-np.sign(m))**abs(m) if m else 1
            Pb = (-1.0)**abs(m) * P[n, abs(m)]
            dPb = -(-1.0)**abs(m) * dP[n, abs(m)] * np.sin(th)
            c = f * np.exp(1j * m * ph) / np.sqrt(2 * np.pi * n * (n + 1))
            K1_ref = np.column_stack([0 * th, 1j * m * Pb / np.sin(th), -dPb]) * ((-1j)**(n + 1) * c)[:, None]
            K2_ref = np.column_stack([0 * th, dPb, 1j * m * Pb / np.sin(th)]) * ((-1j)**n * c)[:, None]
            assert np.allclose(K1[:, i], K1_ref) and np.allclose(K2[:, i], K2_ref)

def test_dipole(): # x-directed Hertzian dipole -> only s=2, n=1, m=+/-1; R matches the ideal probe in diagrams
    Eth = np.outer(np.cos(theta), np.cos(phi))
    Eph = np.outer(np.ones_like(theta), -np.sin(phi))
    T = ff_to_T(theta, phi, Eth, Eph, 5, conj=False)
    big = np.abs(T) > 1e-9 * np.abs(T).max()
    assert set(np.flatnonzero(big)) == {j_index(2, -1, 1), j_index(2, 1, 1)}
    R = T_to_R(T)[:6]
    R_ideal = np.zeros(6); R_ideal[j_index(2, 1, 1)] = -1; R_ideal[j_index(2, -1, 1)] = 1
    assert np.allclose(R / R[j_index(2, -1, 1)], R_ideal)

def test_nf_far_limit():
    N, k, r = 8, 300.0, 1e4
    T = random_T(N)
    th, ph = np.linspace(0.2, 3.0, 15), np.linspace(-3, 3, 15)
    xyz = r * np.column_stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    E, H = nf_from_T(T, xyz, k)
    Eth, Eph = T_to_ff(T, th, ph)
    that = np.column_stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)])
    phat = np.column_stack([-np.sin(ph), np.cos(ph), 0 * ph])
    E_ff = (Eth[:, None] * that + Eph[:, None] * phat) * np.exp(1j * k * r) / r
    assert np.allclose(E, E_ff, rtol=1e-3, atol=1e-3 * np.abs(E_ff).max())
    assert np.allclose(H, np.cross(xyz / r, E) / Z0, atol=1e-3 * np.abs(H).max())

def test_nf_maxwell(): # curl E = i w mu H (e^{-iwt}), checked by central differences near the source
    N, k = 5, 300.0
    T = random_T(N, 1)
    p0 = np.array([0.03, -0.02, 0.05])
    h = 1e-7
    dE = [(nf_from_T(T, (p0 + h * e)[None], k)[0][0] - nf_from_T(T, (p0 - h * e)[None], k)[0][0]) / (2 * h) for e in np.eye(3)]
    curl = np.array([dE[1][2] - dE[2][1], dE[2][0] - dE[0][2], dE[0][1] - dE[1][0]])
    _, H = nf_from_T(T, p0[None], k)
    assert np.allclose(curl, 1j * k * Z0 * H[0], rtol=1e-5)
