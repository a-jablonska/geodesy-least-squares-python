import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def get_azimuth(dx: float, dy: float) -> float:
    """Computes geodetic azimuth in radians [0, 2*pi) from coordinate increments (dx, dy)."""
    return np.arctan2(dy, dx) % (2.0 * np.pi)


def wrap_angle(angle: float) -> float:
    """Wraps an angle in radians into the [-pi, pi] interval."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def adjust_2d_network():
    """
    Performs iterative parametric least-squares adjustment (LSA) of a 2D
    angle-distance geodetic network, assesses coordinate and observation accuracy,
    computes error ellipse parameters, and generates a network visualization.
    """
    # Unit conversion constants (grads and centesimal seconds to radians)
    RO_GRAD = 200.0 / np.pi
    RO_CC = 2_000_000.0 / np.pi

    # =========================================================================
    # 1. INPUT DATA
    # =========================================================================

    # Control points (1, 2) and initial approximate coordinates for unknown points (11, 12, 13) [m]
    X0 = {1: 2399.754, 2: 1987.380, 11: 3010.0, 12: 1830.0, 13: 2800.0}
    Y0 = {1: 2351.218, 2: 1194.429, 11: 2040.0, 12: 2010.0, 13: 1300.0}

    fixed_points = [1, 2]
    unknown_points = [11, 12, 13]

    # Distance observations: (from_pt, to_pt, distance_m, std_error_m)
    dist_obs = [
        (1, 12, 667.813, 0.036),
        (11, 1, 686.008, 0.044),
        (12, 11, 1250.414, 0.065),
    ]

    # Direction observations: (station_pt, target_pt, direction_grad, std_error_cc)
    dir_obs = [
        (11, 1, 0.00000, 18.5),
        (11, 12, 24.69093, 15.1),
        (11, 2, 61.38858, 13.0),
        (11, 13, 94.97012, 16.0),
        (11, 1, 399.99978, 18.5),
        (12, 2, 0.00000, 17.9),
        (12, 13, 51.25836, 14.8),
        (12, 11, 99.37815, 17.6),
        (12, 1, 124.78049, 16.0),
        (12, 2, 399.99972, 17.9),
    ]

    # Horizontal angle observations: (station_pt, left_pt, right_pt, angle_grad, std_error_cc)
    ang_obs = [
        (11, 1, 12, 24.69117, 14.8),
        (11, 2, 13, 33.58204, 19.1),
        (13, 11, 1, 43.47623, 10.2),
        (13, 12, 2, 48.78025, 17.0),
        (2, 13, 11, 36.03707, 11.7),
        (2, 1, 12, 33.56472, 13.3),
        (12, 2, 13, 51.25806, 17.3),
        (12, 13, 11, 48.12052, 16.0),
        (1, 12, 2, 41.65519, 19.1),
        (1, 2, 13, 46.69886, 17.6),
    ]

    # Parameter column indices: dX, dY for points 11, 12, 13 and orientation unknowns dC for stations 11, 12
    idx_xy = {11: (0, 1), 12: (2, 3), 13: (4, 5)}
    idx_c = {11: 6, 12: 7}
    num_params = 8
    num_obs = len(dist_obs) + len(dir_obs) + len(ang_obs)

    # =========================================================================
    # 2. WEIGHT MATRIX P
    # =========================================================================

    P = np.zeros((num_obs, num_obs))
    row = 0
    for _, _, _, m_d in dist_obs:
        P[row, row] = 1.0 / (m_d ** 2)
        row += 1
    for _, _, _, m_k in dir_obs:
        P[row, row] = 1.0 / ((m_k / RO_CC) ** 2)
        row += 1
    for _, _, _, _, m_a in ang_obs:
        P[row, row] = 1.0 / ((m_a / RO_CC) ** 2)
        row += 1

    # Initial station orientation parameters C0 [rad]
    C0 = {
        11: get_azimuth(X0[1] - X0[11], Y0[1] - Y0[11]),
        12: get_azimuth(X0[2] - X0[12], Y0[2] - Y0[12]),
    }

    # =========================================================================
    # 3. ITERATIVE LEAST-SQUARES ADJUSTMENT
    # =========================================================================

    max_iterations = 10
    tolerance = 1e-8
    iterations_used = 0

    for iteration in range(1, max_iterations + 1):
        iterations_used = iteration
        A = np.zeros((num_obs, num_params))
        L = np.zeros((num_obs, 1))
        row = 0

        # 3a. Distance observation equations
        for p_from, p_to, d_obs, _ in dist_obs:
            dx = X0[p_to] - X0[p_from]
            dy = Y0[p_to] - Y0[p_from]
            d0 = np.hypot(dx, dy)

            if p_from in idx_xy:
                A[row, idx_xy[p_from][0]] = -dx / d0
                A[row, idx_xy[p_from][1]] = -dy / d0
            if p_to in idx_xy:
                A[row, idx_xy[p_to][0]] = dx / d0
                A[row, idx_xy[p_to][1]] = dy / d0

            L[row, 0] = d0 - d_obs
            row += 1

        # 3b. Direction observation equations
        for p_from, p_to, k_obs, _ in dir_obs:
            dx = X0[p_to] - X0[p_from]
            dy = Y0[p_to] - Y0[p_from]
            d0_sq = dx ** 2 + dy ** 2
            a0 = get_azimuth(dx, dy)

            if p_from in idx_xy:
                A[row, idx_xy[p_from][0]] = dy / d0_sq
                A[row, idx_xy[p_from][1]] = -dx / d0_sq
            if p_to in idx_xy:
                A[row, idx_xy[p_to][0]] = -dy / d0_sq
                A[row, idx_xy[p_to][1]] = dx / d0_sq

            A[row, idx_c[p_from]] = -1.0
            L[row, 0] = wrap_angle(a0 - C0[p_from] - (k_obs / RO_GRAD))
            row += 1

        # 3c. Horizontal angle observation equations (alpha = Az_right - Az_left)
        for st, left_pt, right_pt, ang_val, _ in ang_obs:
            dx_l, dy_l = X0[left_pt] - X0[st], Y0[left_pt] - Y0[st]
            dx_r, dy_r = X0[right_pt] - X0[st], Y0[right_pt] - Y0[st]
            d0_l_sq = dx_l ** 2 + dy_l ** 2
            d0_r_sq = dx_r ** 2 + dy_r ** 2
            a0_l = get_azimuth(dx_l, dy_l)
            a0_r = get_azimuth(dx_r, dy_r)

            if right_pt in idx_xy:
                A[row, idx_xy[right_pt][0]] += -dy_r / d0_r_sq
                A[row, idx_xy[right_pt][1]] += dx_r / d0_r_sq
            if st in idx_xy:
                A[row, idx_xy[st][0]] += dy_r / d0_r_sq
                A[row, idx_xy[st][1]] += -dx_r / d0_r_sq

            if left_pt in idx_xy:
                A[row, idx_xy[left_pt][0]] -= -dy_l / d0_l_sq
                A[row, idx_xy[left_pt][1]] -= dx_l / d0_l_sq
            if st in idx_xy:
                A[row, idx_xy[st][0]] -= dy_l / d0_l_sq
                A[row, idx_xy[st][1]] -= -dx_l / d0_l_sq

            L[row, 0] = wrap_angle((a0_r - a0_l) - (ang_val / RO_GRAD))
            row += 1

        N = A.T @ P @ A
        N_inv = np.linalg.inv(N)
        dX = -N_inv @ (A.T @ P @ L)

        X0[11] += dX[0, 0]
        Y0[11] += dX[1, 0]
        X0[12] += dX[2, 0]
        Y0[12] += dX[3, 0]
        X0[13] += dX[4, 0]
        Y0[13] += dX[5, 0]
        C0[11] += dX[6, 0]
        C0[12] += dX[7, 0]

        if np.max(np.abs(dX[:6])) < tolerance:
            break

    # =========================================================================
    # 4. ACCURACY ASSESSMENT & ERROR ELLIPSES
    # =========================================================================

    V = A @ dX + L
    control_1 = A.T @ P @ V
    vpv = (V.T @ P @ V)[0, 0]
    dof = num_obs - num_params
    sigma_0 = np.sqrt(vpv / dof)

    # Covariance matrix of adjusted parameters CX
    CX = (sigma_0 ** 2) * N_inv
    mx = np.sqrt(np.diag(CX))

    # Covariance matrix of adjusted observations CL_adj
    CL_adj = (sigma_0 ** 2) * (A @ N_inv @ A.T)
    mL_adj = np.sqrt(np.diag(CL_adj))

    # =========================================================================
    # 5. CONSOLE REPORT
    # =========================================================================

    print("=== MATHEMATICAL CONTROLS ===")
    print("Normal equations check (A^T * P * V ~ 0):")
    print(np.round(control_1.flatten(), 10))
    print(f"Converged in {iterations_used} iterations | Degrees of freedom (f) = {dof}")
    print(f"A posteriori unit weight error (m0) = {sigma_0:.4f}\n")

    print("=== ADJUSTED COORDINATES & ERROR ELLIPSES ===")
    ellipses_data = {}
    for pt in unknown_points:
        i = idx_xy[pt][0]
        mp = np.hypot(mx[i], mx[i + 1])

        # 2x2 sub-covariance matrix for point (X, Y)
        C_pt = CX[i:i + 2, i:i + 2]
        eigvals, eigvecs = np.linalg.eigh(C_pt)
        semi_major = np.sqrt(eigvals[1]) * 1000.0  # [mm]
        semi_minor = np.sqrt(eigvals[0]) * 1000.0  # [mm]

        # Geodetic azimuth of semi-major axis (X is North/vertical, Y is East/horizontal)
        vx, vy = eigvecs[:, 1]
        phi_rad = get_azimuth(vx, vy)
        phi_grad = phi_rad * RO_GRAD

        ellipses_data[pt] = (semi_major, semi_minor, phi_rad)

        print(f"Point {pt:2d}: X = {X0[pt]:9.3f} m (mX = {mx[i] * 1000:5.1f} mm)")
        print(f"          Y = {Y0[pt]:9.3f} m (mY = {mx[i + 1] * 1000:5.1f} mm)")
        print(f"          Position error mP = {mp * 1000:5.1f} mm")
        print(f"          Error ellipse: a = {semi_major:5.1f} mm, b = {semi_minor:5.1f} mm, phi = {phi_grad:6.2f} grad\n")

    print("=== ADJUSTED OBSERVATION STANDARD ERRORS ===")
    print("Distances (3) [m]:   ", ", ".join(f"{mL_adj[i]:.4f}" for i in range(3)))
    print("Directions (10) [cc]:", ", ".join(f"{mL_adj[i] * RO_CC:.1f}" for i in range(3, 13)))
    print("Angles (10) [cc]:    ", ", ".join(f"{mL_adj[i] * RO_CC:.1f}" for i in range(13, 23)))

    # =========================================================================
    # 6. NETWORK GEOMETRY & ERROR ELLIPSES VISUALIZATION
    # =========================================================================

    fig, ax = plt.subplots(figsize=(9, 8))

    # Plot sightlines / network edges
    edges = [
        (1, 11), (1, 12), (1, 13), (1, 2),
        (2, 11), (2, 12), (2, 13),
        (11, 12), (11, 13), (12, 13),
    ]
    for p1, p2 in edges:
        ax.plot([Y0[p1], Y0[p2]], [X0[p1], X0[p2]], linestyle="--", color="#8c969e", linewidth=1.2, zorder=1)

    # Highlight measured distances in bold
    for p1, p2, _, _ in dist_obs:
        ax.plot([Y0[p1], Y0[p2]], [X0[p1], X0[p2]], linestyle="-", color="#d95f02", linewidth=2.0, alpha=0.8, zorder=2,
                label="Measured distance" if (p1, p2) == dist_obs[0][:2] else "")

    # Plot control points (triangles) and adjusted unknown points (circles)
    ax.scatter([Y0[p] for p in fixed_points], [X0[p] for p in fixed_points],
               marker="^", s=160, color="#1b9e77", edgecolor="black", zorder=4, label="Control points (1, 2)")
    ax.scatter([Y0[p] for p in unknown_points], [X0[p] for p in unknown_points],
               marker="o", s=130, color="#377eb8", edgecolor="black", zorder=4, label="Adjusted points (11, 12, 13)")

    # Plot scaled error ellipses (exaggerated for visibility)
    ellipse_scale = 8000.0  # 1 mm on ellipse -> 8 m on coordinate grid
    for pt in unknown_points:
        a_mm, b_mm, phi_rad = ellipses_data[pt]
        # Convert geodetic azimuth (clockwise from North/X) to Matplotlib angle (counter-clockwise from East/Y)
        mpl_angle_deg = np.degrees(np.pi / 2.0 - phi_rad)
        ell = Ellipse(
            xy=(Y0[pt], X0[pt]),
            width=2.0 * (a_mm / 1000.0) * ellipse_scale,
            height=2.0 * (b_mm / 1000.0) * ellipse_scale,
            angle=mpl_angle_deg,
            edgecolor="#e41a1c",
            facecolor="#e41a1c",
            alpha=0.18,
            linewidth=1.8,
            zorder=3,
            label="Standard error ellipse (8000x)" if pt == unknown_points[0] else ""
        )
        ax.add_patch(ell)

    # Annotate point IDs
    for pt in fixed_points + unknown_points:
        ax.annotate(f"P{pt}", (Y0[pt], X0[pt]), textcoords="offset points", xytext=(10, 8),
                    fontsize=11, fontweight="bold", color="#222222", zorder=5)

    ax.set_title("2D Geodetic Network Adjustment & Standard Error Ellipses", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Y [m] (East)", fontsize=11)
    ax.set_ylabel("X [m] (North)", fontsize=11)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.show()


def main():
    adjust_2d_network()


if __name__ == "__main__":
    main()
