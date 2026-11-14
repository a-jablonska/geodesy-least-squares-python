# Geodetic Computations & Least-Squares Adjustment in Python

Numerical implementations of geodetic network adjustment, covariance propagation, error ellipse computation, and accuracy assessment algorithms built in Python.

## Projects

| # | Project | Description |
| :--- | :--- | :--- |
| 01 | **[1D Leveling Network Adjustment](./01-leveling-network-adjustment/)** | Parametric least-squares adjustment (LSA) of a leveling network (14 observations, 7 unknown points, 3 benchmarks), mathematical controls, covariance matrices, correlations, and linear function error propagation. |
| 02 | **[2D Angle-Distance Network Adjustment](./02-2d-geodetic-network-adjustment/)** | Iterative non-linear least-squares adjustment of a 2D terrestrial network (distances, directions, angles), station orientation estimation, standard error ellipses, and Matplotlib network visualization. |

## Tech Stack
* **Language:** Python 3.8+
* **Libraries:** `NumPy`, `Matplotlib`

## Quick Start

```bash
pip install numpy matplotlib
python 02-2d-geodetic-network-adjustment/network_adjustment_2d.py
