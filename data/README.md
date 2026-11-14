# Data (not tracked)

Notebooks read from this directory:

```
SNF_fields_ports/{AUT,probe}/wp{1,2}/{E,H}.csv   # HFSS field calculator export
S_params.csv                                     # HFSS S21 sweep over TH_PROBE
FF_fields/FF_{AUT,probe}_wp{1,2}.csv             # HFSS far field: Phi(rad), Theta(rad), rEphi, rEtheta (mag ang)
```

E/H columns: Variation, X, Y, Z, Ex, Ey, Ez, each component "mag ang" (ang in rad). Rows: theta slowest, phi fastest.
