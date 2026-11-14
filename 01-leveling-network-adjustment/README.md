# 1D Leveling Network Least-Squares Adjustment

Parametric least-squares adjustment (LSA) of a geodetic leveling network consisting of **3 fixed benchmarks** (`H101`, `H102`, `H103`), **7 unknown points** (`P1`–`P7`), and **14 observed elevation differences**.

## Mathematical Model

The functional and stochastic models follow standard parametric adjustment:

```text
V = A * X + L,      P = diag(1 / m_i²)
X = -(Aᵀ * P * A)⁻¹ * (Aᵀ * P * L)
σ₀ = sqrt((Vᵀ * P * V) / (n - r))
C_X = σ₀² * (Aᵀ * P * A)⁻¹
