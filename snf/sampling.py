import numpy as np

def equiangular_bernoulli(num_points, dth=np.radians(9), n_phi=25): # Equiangular grid, randomly subsampled
    theta = np.arange(dth / 2, np.pi, dth)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta, phi = np.meshgrid(theta, phi)
    theta, phi = theta.flatten(), phi.flatten()
    idx = np.random.choice(len(theta), size=num_points)
    return theta[idx], phi[idx]

def spiral_random_phi(num_points): # Constant-ish theta spacing with random phi
    theta = np.arccos((2 * np.arange(num_points) - num_points + 1) / (num_points - 1))
    phi = np.random.rand(num_points) * 2 * np.pi
    return theta, phi

def constant_velocity(num_points, theta_rpm=44, phi_rpm=44 / np.pi, sample_period=0.25): # Both axes spin at constant rpm, sampled periodically
    theta = np.linspace(0, theta_rpm * 2 * np.pi * sample_period * num_points / 60, num_points, endpoint=False)
    phi = np.linspace(0, phi_rpm * 2 * np.pi * sample_period * num_points / 60, num_points, endpoint=False)
    return theta, phi

def randomized_constant_velocity(num_points, sample_points=40000, per_rot=6): # Constant velocity, Bernoulli sampling along phi. per_rot = average samples per rotation
    idx = np.cumsum(np.random.geometric(p=per_rot / sample_points, size=num_points))
    max_rad = np.max(idx) * 2 * np.pi / sample_points
    phi_full = np.linspace(0, max_rad, np.max(idx))
    phi = phi_full[idx - 1]
    theta = phi / np.pi
    return theta, phi

def sphere_position(theta, phi): # Fold Euler angles onto theta in [0, pi], phi in [0, 2pi)
    th_pos = np.arccos(np.cos(theta))
    ph_pos = np.where(np.sin(theta) < 0, phi + np.pi, phi) % (2 * np.pi)
    return th_pos, ph_pos

def add_chi(theta, phi, chis=(0, np.pi / 2)): # Repeat each (theta, phi) for every chi
    chi = np.repeat(chis, theta.size)
    return chi, np.tile(theta, len(chis)), np.tile(phi, len(chis))

def coherence(A):
    '''
    Returns mutual coherence of matrix A.

    A (np.array): Dimensions (M, N)
    '''
    norms = np.linalg.norm(A, axis=0)
    if np.any(norms == 0):
        raise ValueError(f"{np.sum(norms == 0)} zero columns; coherence is undefined")
    A_norm = A / norms
    G = np.abs(A_norm.conj().T @ A_norm)        # (K, K) Gram matrix magnitudes
    np.fill_diagonal(G, 0)
    mu = G.max()
    return mu

def welch_bound(M, K): # Lower bound on coherence of an (M, K) matrix
    return np.sqrt((K - M) / (M * (K - 1))) if K > M else 0.0
