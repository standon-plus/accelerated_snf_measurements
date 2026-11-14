# Accelerated spherical near-field measurements

Simulating SNF measurements with reciprocity (two HFSS simulations) and recovering the SWEC with compressed sensing.

```
snf/fields.py     # load HFSS fields, rotate probe (Rz(phi) Ry(theta) Rz(chi)), reciprocity S21
snf/swe.py        # Hansen SWE: Wigner d, translation coefficients, probe response, Psi matrix
snf/sampling.py   # sampling patterns, coherence, Welch bound
snf/plotting.py   # E/H cuts, measurement sphere
snf/farfield.py   # HFSS far field -> Hansen T (and R), SWE far/near field evaluation
notebooks/        # calculate_s21 (reciprocity vs HFSS), diagrams (Psi + coherence), farfield_swec (SWEC + checks)
tests/
```

Setup: `pip install -e .[test]`, put the HFSS exports in `data/` (see `data/README.md`), run `pytest`.

## Conventions
- HFSS uses e^{+jωt}, Hansen uses e^{-iωt}: conjugate HFSS fields/S-params before Hansen code. The reciprocity integral works in HFSS's convention directly.
- SWE: rE e^{-ikr} = sqrt(Z0) sum Q K_smn, Q = sqrt(2 P_in) T, radiated power = P_in sum |T|^2. Receive: R_smn = (-1)^m T_s,-m,n.
- Wigner d follows Hansen (Edmonds). The `spherical` package's (mu, m) entry is Hansen's (m, mu); `wigner_d_rows` swaps them.
