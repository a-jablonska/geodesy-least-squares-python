import numpy as np


def adjust_leveling_network():
    """
    Performs parametric least-squares adjustment (LSA) of a 1D leveling network,
    verifies mathematical controls, and computes covariance matrices, standard
    errors, correlations, and accuracy of derived functions.
    """
    # =========================================================================
    # 1. INPUT DATA
    # =========================================================================

    # Known benchmark elevations [m]
    H101 = 221.6500
    H102 = 210.0821
    H103 = 214.9079

    # Observed elevation differences h [m]
    # Network edges:
    #  1: P4 -> P6    6: P6 -> P102  11: P5 -> P6
    #  2: P1 -> P4    7: P2 -> P6    12: P3 -> P5
    #  3: P1 -> P7    8: P2 -> P7    13: P4 -> P7
    #  4: P3 -> P103  9: P5 -> P101  14: P2 -> P3
    #  5: P5 -> P7   10: P1 -> P2
    h = np.array([
        [  3.457],  # 1:  4 -> 6
        [ 12.245],  # 2:  1 -> 4
        [ 10.810],  # 3:  1 -> 7
        [ -2.674],  # 4:  3 -> 103
        [  2.263],  # 5:  5 -> 7
        [-13.150],  # 6:  6 -> 102
        [ -1.225],  # 7:  2 -> 6
        [ -6.111],  # 8:  2 -> 7
        [  5.585],  # 9:  5 -> 101
        [ 16.926],  # 10: 1 -> 2
        [  7.162],  # 11: 5 -> 6
        [ -1.506],  # 12: 3 -> 5
        [ -1.441],  # 13: 4 -> 7
        [ -6.867]   # 14: 2 -> 3
    ])

    # Standard errors converted from millimeters to meters [m]
    std_errors = np.array([6, 6, 5, 3, 7, 7, 3, 7, 3, 5, 5, 6, 7, 8]) / 1000.0

    # =========================================================================
    # 2. SYSTEM MATRICES
    # =========================================================================

    # Design matrix A (14 observations x 7 unknowns: P1 to P7)
    A = np.array([
        [ 0,  0,  0, -1,  0,  1,  0],  # 1:  4 -> 6
        [-1,  0,  0,  1,  0,  0,  0],  # 2:  1 -> 4
        [-1,  0,  0,  0,  0,  0,  1],  # 3:  1 -> 7
        [ 0,  0, -1,  0,  0,  0,  0],  # 4:  3 -> 103
        [ 0,  0,  0,  0, -1,  0,  1],  # 5:  5 -> 7
        [ 0,  0,  0,  0,  0, -1,  0],  # 6:  6 -> 102
        [ 0, -1,  0,  0,  0,  1,  0],  # 7:  2 -> 6
        [ 0, -1,  0,  0,  0,  0,  1],  # 8:  2 -> 7
        [ 0,  0,  0,  0, -1,  0,  0],  # 9:  5 -> 101
        [-1,  1,  0,  0,  0,  0,  0],  # 10: 1 -> 2
        [ 0,  0,  0,  0, -1,  1,  0],  # 11: 5 -> 6
        [ 0,  0, -1,  0,  1,  0,  0],  # 12: 3 -> 5
        [ 0,  0,  0, -1,  0,  0,  1],  # 13: 4 -> 7
        [ 0, -1,  1,  0,  0,  0,  0]   # 14: 2 -> 3
    ], dtype=float)

    # Misclosure vector L [m]
    L = np.array([
        [-h[0, 0]],
        [-h[1, 0]],
        [-h[2, 0]],
        [H103 - h[3, 0]],
        [-h[4, 0]],
        [H102 - h[5, 0]],
        [-h[6, 0]],
        [-h[7, 0]],
        [H101 - h[8, 0]],
        [-h[9, 0]],
        [-h[10, 0]],
        [-h[11, 0]],
        [-h[12, 0]],
        [-h[13, 0]]
    ])

    # Weight matrix P = diag(1 / m_i^2)
    P = np.diag(std_errors ** (-2))

    # =========================================================================
    # 3. LEAST-SQUARES ADJUSTMENT & MATHEMATICAL CONTROLS
    # =========================================================================

    N_inv = np.linalg.inv(A.T @ P @ A)

    # Unknown parameters X (adjusted point elevations)
    X = -N_inv @ (A.T @ P @ L)

    # Residuals V and adjusted observations h_adj
    V = A @ X + L
    h_adj = h + V

    # Control 1: Normal equations orthogonality (A^T * P * V == 0)
    control_1 = A.T @ P @ V

    # Control 2: Weighted sum of squared residuals (V^T * P * V == L^T * P * V)
    vpv_1 = (V.T @ P @ V)[0, 0]
    vpv_2 = (L.T @ P @ V)[0, 0]

    # =========================================================================
    # 4. ACCURACY ASSESSMENT (COVARIANCE MATRICES)
    # =========================================================================

    dof = len(h) - A.shape[1]  # Degrees of freedom (n - r)
    sigma_0 = np.sqrt(vpv_1 / dof)  # A posteriori standard error of unit weight

    # Covariance matrix of adjusted elevations (CX)
    CX = (sigma_0 ** 2) * N_inv
    mX = np.sqrt(np.diag(CX))

    # Covariance matrix of residuals (CV)
    CV = (sigma_0 ** 2) * (np.linalg.inv(P) - A @ N_inv @ A.T)
    mV = np.sqrt(np.diag(CV))

    # Covariance matrix of adjusted observations (Ch)
    Ch = (sigma_0 ** 2) * (A @ N_inv @ A.T)
    mh = np.sqrt(np.diag(Ch))

    # =========================================================================
    # 5. CORRELATIONS & ERROR PROPAGATION FOR DERIVED FUNCTIONS
    # =========================================================================

    # Correlation coefficient between adjusted elevations H1 and H3
    rho_H1_H3 = CX[0, 2] / (mX[0] * mX[2])

    # Correlation coefficients between adjusted observations (h1, h2) and (h1, h4)
    rho_h1_h2 = Ch[0, 1] / (mh[0] * mh[1])
    rho_h1_h4 = Ch[0, 3] / (mh[0] * mh[3])

    # Standard error of linear function u1 = 2*H1 + 3*H2 - H3
    D1 = np.array([[2.0, 3.0, -1.0, 0.0, 0.0, 0.0, 0.0]])
    mu1 = np.sqrt((D1 @ CX @ D1.T)[0, 0])

    # Standard error of linear function u2 = h_23 + h_45 = (H3 - H2) + (H5 - H4)
    D2 = np.array([[0.0, -1.0, 1.0, -1.0, 1.0, 0.0, 0.0]])
    mu2 = np.sqrt((D2 @ CX @ D2.T)[0, 0])

    # =========================================================================
    # 6. OUTPUT REPORT
    # =========================================================================

    print("=== MATHEMATICAL CONTROLS ===")
    print("Control 1 (A^T * P * V ~ 0):")
    print(np.round(control_1.flatten(), 10))
    print(f"Control 2: V^T * P * V = {vpv_1:.6f} | L^T * P * V = {vpv_2:.6f}")

    print("\n=== ADJUSTMENT SUMMARY ===")
    print(f"Degrees of freedom (f) = {dof}")
    print(f"A posteriori unit weight error (m0) = {sigma_0:.4f}")

    print("\n[Adjusted Point Elevations & Standard Errors]")
    for i in range(len(X)):
        print(f"P{i + 1} = {X[i, 0]:9.4f} m  (mH = {mX[i] * 1000:5.1f} mm)")

    print("\n[Correlations & Derived Functions]")
    print(f"Correlation rho(H1, H3) = {rho_H1_H3:.4f}")
    print(f"Correlation rho(h1, h2) = {rho_h1_h2:.4f}")
    print(f"Correlation rho(h1, h4) = {rho_h1_h4:.4f}")
    print(f"Std error of u1 = 2*H1 + 3*H2 - H3: {mu1 * 1000:.1f} mm")
    print(f"Std error of u2 = h_23 + h_45:      {mu2 * 1000:.1f} mm")

    print("\n[Adjusted Observations, Residuals & Standard Errors]")
    print("Obs | h_adj [m] |  v [mm] | m_v [mm] | m_h [mm]")
    print("-" * 50)
    for i in range(len(h_adj)):
        print(f"{i + 1:3d} | {h_adj[i, 0]:9.4f} | {V[i, 0] * 1000:7.2f} | {mV[i] * 1000:8.1f} | {mh[i] * 1000:8.1f}")


def main():
    adjust_leveling_network()


if __name__ == "__main__":
    main()