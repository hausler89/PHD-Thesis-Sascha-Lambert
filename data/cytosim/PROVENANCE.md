# Cytosim force-curve data

**Authors of the data: Kristian A. Pajanonot and Komal Bhattacharyya.** They designed and ran the Cytosim
simulations and provided the force curves; this repository only uses them (section 3.6). Please credit them when using
these data.

Simulations of two perpendicular filaments in Cytosim with Bell–Evans bonds.

| File | Bond | r_u0 | x_u | Stiffness k | Recording |
|---|---|---|---|---|---|
| `stiff{80,40,20}_1.csv` | 1 | 0.01 | 0.8 | 80, 40, 20 | censored (peak force > 2 pN) |
| `stiff{80,40,20}_2.csv` | 2 | 0.15 | 0.2 | 80, 40, 20 | censored (peak force > 2 pN) |
| `stiff{80,40,20}_4_all.csv` | 3 | 0.15 | 0.8 | 80, 40, 20 | uncensored |

Format: one column per force curve (`curve0, curve1, ...`, force in pN, NaN after the bond broke),
last column: time in s. k_BT = 4.112 pN nm.
