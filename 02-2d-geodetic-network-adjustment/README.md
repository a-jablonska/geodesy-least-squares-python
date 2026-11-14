# 2D Angle-Distance Geodetic Network Adjustment

Iterative parametric least-squares adjustment (LSA) of a 2D horizontal geodetic network consisting of **2 fixed control points** (`P1`, `P2`), **3 unknown points** (`P11`, `P12`, `P13`), and **23 terrestrial observations** (3 distances, 10 horizontal directions from 2 eccentric stations, and 10 horizontal angles).

## Mathematical Model

Non-linear observation equations are linearized via Taylor series expansion around approximate coordinates `(X⁰, Y⁰)` and solved iteratively until convergence (`max|dX| < 10⁻⁸ m`):

```text
V = A * dX + L,      P = diag(1 / m_i²)
dX = -(Aᵀ * P * A)⁻¹ * (Aᵀ * P * L)
σ₀ = sqrt((Vᵀ * P * V) / (n - r))
C_X = σ₀² * (Aᵀ * P * A)⁻¹
