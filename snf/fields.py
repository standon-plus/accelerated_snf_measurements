import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

def load_field(path): # XYZ positions and field components
    df = pd.read_csv(path, skipinitialspace=True)
    xyz = df.iloc[:, 1:4].to_numpy(float)
    comps = []
    for col in df.columns[4:7]:
        mag_ang = df[col].str.split(expand=True).astype(float).to_numpy()
        comps.append(mag_ang[:, 0] * np.exp(1j * mag_ang[:, 1]))
    return xyz, np.column_stack(comps)

def grid_axes(xyz): # theta, phi, and R of vector field
    r = np.linalg.norm(xyz, axis=1)
    theta = np.arccos(np.clip(xyz[:, 2] / r, -1, 1))
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    n_th = np.unique(np.round(theta, 9)).size
    n_ph = len(xyz) // n_th
    th_1d = theta.reshape(n_th, n_ph)[:, 0]
    ph_1d = np.linspace(phi[n_ph], phi[2 * n_ph - 1], n_ph)  # from a non-pole ring
    return th_1d, ph_1d, r.mean()

def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotate_field(xyz, F, R): # Rotate field according to R matrix. Interpolate to keep same points
    th_1d, ph_1d, r = grid_axes(xyz)
    n_th, n_ph = th_1d.size, ph_1d.size
    F_grid = F.reshape(n_th, n_ph, 3)

    interp = RegularGridInterpolator((th_1d, ph_1d), F_grid, method="linear")

    # Find where each output point came from before the rotation
    src = xyz @ R # Unrotate
    th_s = np.clip(np.arccos(np.clip(src[:, 2] / r, -1, 1)), th_1d[0], th_1d[-1])
    ph_s = np.clip(np.arctan2(src[:, 1], src[:, 0]), ph_1d[0], ph_1d[-1])

    F_src = interp(np.column_stack([th_s, ph_s])) # original vectors at source points
    return F_src @ R.T

def rotate_by_angles(xyz, F, theta, phi, chi=0, degrees=True): # Rotate field by R = Rz(phi) Ry(theta) Rz(chi); chi is about the probe's own axis
    if degrees:
        theta, phi, chi = np.radians(theta), np.radians(phi), np.radians(chi)
    return rotate_field(xyz, F, Rz(phi) @ Ry(theta) @ Rz(chi))

def s21_reciprocity(xyz, Ea, Ha, Eb, Hb, P1=1.0, P2=1.0): # S21 = (1/(4 sqrt(P1 P2))) * surface integral of (E1 x H2 - E2 x H1)
    th_1d, ph_1d, R = grid_axes(xyz)
    n_th, n_ph = th_1d.size, ph_1d.size
    n = xyz / np.linalg.norm(xyz, axis=1)[:, None]

    # Integrand on the sphere
    S_n = np.sum(np.cross(Ea, Hb) * n, axis=1) - np.sum(np.cross(Eb, Ha) * n, axis=1)
    integrand = S_n.reshape(n_th, n_ph) * np.sin(th_1d)[:, None]

    # Integrate (R^2 S_n sin(theta) dphi dtheta)
    inner = np.trapezoid(integrand, ph_1d, axis=1)
    w = R**2 * np.trapezoid(inner, th_1d)

    return w / (4 * np.sqrt(P1 * P2))

def load_hfss_s21(path, probe_wp, aut_wp):
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df.sort_values('TH_PROBE [deg]')
    thetas = df['TH_PROBE [deg]'].to_numpy(float)

    s_name = f'S(Probe1_PROBE_WP:{probe_wp},AUT_PC1_AUT_WP:{aut_wp})'
    find = lambda prefix: [c for c in df.columns if c.startswith(prefix + '(' + s_name)]
    mag_dB = df[find('dB')[0]].to_numpy(float)
    phase_deg = df[find('ang_deg')[0]].to_numpy(float)
    return thetas, mag_dB, phase_deg
