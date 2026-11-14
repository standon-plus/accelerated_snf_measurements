import numpy as np
from snf.fields import rotate_field, rotate_by_angles, s21_reciprocity, Rz, Ry

def sphere_grid(r=0.15, n_th=73, n_ph=145): # Same ordering as HFSS export: theta slow, phi fast
    th, ph = np.meshgrid(np.linspace(0, np.pi, n_th), np.linspace(-np.pi, np.pi, n_ph), indexing='ij')
    th, ph = th.flatten(), ph.flatten()
    return r * np.column_stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])

def test_chi_zero_matches_theta_phi():
    xyz = sphere_grid()
    F = np.random.default_rng(0).normal(size=xyz.shape) + 0j
    assert np.allclose(rotate_by_angles(xyz, F, 30, 45), rotate_field(xyz, F, Rz(np.radians(45)) @ Ry(np.radians(30))))

def test_chi_swaps_polarization(): # chi = 90 deg turns an x-polarized field into y
    xyz = sphere_grid()
    F = np.tile([1, 0, 0], (len(xyz), 1)) + 0j
    assert np.allclose(rotate_by_angles(xyz, F, 0, 0, chi=90), np.tile([0, 1, 0], (len(xyz), 1)))

def test_s21_antisymmetric():
    xyz = sphere_grid()
    rng = np.random.default_rng(1)
    Ea, Ha, Eb, Hb = (rng.normal(size=xyz.shape) + 1j * rng.normal(size=xyz.shape) for _ in range(4))
    assert np.isclose(s21_reciprocity(xyz, Ea, Ha, Eb, Hb), -s21_reciprocity(xyz, Eb, Hb, Ea, Ha))
