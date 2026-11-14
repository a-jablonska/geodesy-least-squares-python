# Geodetic Computations & Least-Squares Adjustment in Python

Numerical implementations of geodetic network adjustment, covariance propagation, and accuracy assessment algorithms built in Python with NumPy.

## Projects

| # | Project | Description |
| :--- | :--- | :--- |
| 01 | **[1D Leveling Network Adjustment](./01-leveling-network-adjustment/)** | Parametric least-squares adjustment (LSA) of a leveling network (14 observations, 7 unknown points, 3 benchmarks), mathematical controls, covariance matrices, correlations, and linear function error propagation. |

## Tech Stack
* **Language:** Python 3.8+
* **Libraries:** `NumPy`

## Quick Start

```bash
pip install numpy
python 01-leveling-network-adjustment/leveling_network_adjustment.py
