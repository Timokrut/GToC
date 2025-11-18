import math
import numpy as np
import matplotlib.pyplot as plt

def count_T(V_mod: int) -> float:
    return 1 / V_mod

def count_q(V_inf: int, T: float) -> int:
    log_q = V_inf * T
    return int(2 ** log_q)

def check_orthonormality(T: float, f_0: int, N: int = 3) -> dict:
    t = np.linspace(0, T, N)
    
    phi1 = np.sqrt(2 / T) * np.cos(2 * math.pi * f_0 * t)
    phi2 = np.sqrt(2 / T) * np.sin(2 * math.pi * f_0 * t)
    
    inner_11 = np.trapezoid(phi1 * phi1, t)
    inner_22 = np.trapezoid(phi2 * phi2, t)
    inner_12 = np.trapezoid(phi1 * phi2, t)
    inner_21 = np.trapezoid(phi2 * phi1, t)
    
    return {
    '(φ1, φ1)': inner_11,
    '(φ2, φ2)': inner_22,
    '(φ1, φ2)': inner_12,
    '(φ2, φ1)': inner_21
    }

def create_signal_constellation(q: int, A: float = 1.0) -> tuple:
    L = int(math.sqrt(q))
    constellation_points = []
    i_pairs = []

    for i1 in range(L):
        for i2 in range(L):
            s_i1 = A * (1 - (2 * i1) / (L - 1))
            s_i2 = A * (1 - (2 * i2) / (L - 1))
            constellation_points.append((s_i1, s_i2))
            i_pairs.append((i1, i2))

    return constellation_points, i_pairs

def generate_signal(si1, si2, phi1, phi2):
    return si1 * phi1 + si2 * phi2

def project_onto_basis(s, phi, t):
    return np.trapezoid(s * phi, t)

def distorted_constellation(constellation_points, T, f_0, N=3):
    t = np.linspace(0, T, N)

    phi1 = np.sqrt(2 / T) * np.cos(2 * np.pi * f_0 * t)
    phi2 = np.sqrt(2 / T) * np.sin(2 * np.pi * f_0 * t)

    distorted = []

    for (s1, s2) in constellation_points:
        s = generate_signal(s1, s2, phi1, phi2)
        
        d1 = project_onto_basis(s, phi1, t)
        d2 = project_onto_basis(s, phi2, t)
        
        distorted.append((d1, d2))

    return distorted

def plot_constellations(constellation_points: list, distorted_points: list):
    plt.figure(figsize=(8, 8))

    original_x = [p[0] for p in constellation_points]
    original_y = [p[1] for p in constellation_points]

    distorted_x = [p[0] for p in distorted_points]
    distorted_y = [p[1] for p in distorted_points]

    for i in range(len(constellation_points)):
        x_o = original_x[i]
        y_o = original_y[i]

        x_d = distorted_x[i]
        y_d = distorted_y[i]

        plt.plot(x_o, y_o, 'ro', markersize=8)
        plt.text(x_o + 0.05, y_o + 0.05, f's{i} O', fontsize=10)
        
        plt.plot(x_d, y_d, 'go', markersize=8)
        plt.text(x_d + 0.05, y_d + 0.05, f's{i} D', fontsize=10)
    

    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.xlabel('s_i1 (амплитуда cos)')
    plt.ylabel('s_i2 (амплитуда sin)')
    plt.title(f'Сигнальное созвездие (теоретическое и искаженное)')
    plt.axis('equal')
    plt.show()

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def plot_decision_regions(constellation_points: list, q: int):
    plt.figure(figsize=(8, 8))

    margin = 0.5
    x_min = min(p[0] for p in constellation_points) - margin
    x_max = max(p[0] for p in constellation_points) + margin
    y_min = min(p[1] for p in constellation_points) - margin
    y_max = max(p[1] for p in constellation_points) + margin
    
    step = 0.02
    x_vals = np.arange(x_min, x_max, step)
    y_vals = np.arange(y_min, y_max, step)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = (X[i, j], Y[i, j])
            min_dist = float('inf')
            closest_point_idx = 0
            
            for idx, signal_point in enumerate(constellation_points):
                dist = distance(point, signal_point)
                if dist < min_dist:
                    min_dist = dist
                    closest_point_idx = idx

            Z[i, j] = closest_point_idx
    plt.contourf(X, Y, Z, levels=np.arange(-0.5, q, 1), alpha=0.3, cmap='tab20')

    plt.contour(X, Y, Z, levels=np.arange(-0.5, q, 1), colors='black', linewidths=0.5)
    
    for i, point in enumerate(constellation_points):
        plt.plot(point[0], point[1], 'ro', markersize=8)
        plt.text(point[0] + 0.05, point[1] + 0.05, f's{i}', fontsize=10)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('s_i1 (амплитуда cos)')
    plt.ylabel('s_i2 (амплитуда sin)')
    plt.title(f'Решающие области')
    # plt.axis('equal')
    plt.xlim(-1.5, 1.48)
    plt.ylim(-1.49, 1.49)
    plt.show()

if __name__ == "__main__":
    f_0 = 1200
    V_mod = 600
    V_inf = 2400
    A = 1.0
    T = count_T(V_mod)
    q = count_q(V_inf, T)

    print("Базисные функции:")
    print("φ1(t) = √(2/T) * cos(2πf0t)")
    print("φ2(t) = √(2/T) * sin(2πf0t)")
    print(f"f0 = {f_0} Гц")

    print(f"T = {T:.6f} сек")
    print(f"q = {q}")
    
    ortho_results = check_orthonormality(T, f_0)
    
    for key, value in ortho_results.items():
        print(f"{key} = {value:.6f}")

    print("\nТеоретические значения:")
    print("(φ1, φ1) = 1.000000")
    print("(φ2, φ2) = 1.000000")
    print("(φ1, φ2) = 0.000000")
    print("(φ2, φ1) = 0.000000")

    constellation_points, i_pairs = create_signal_constellation(q, A)
    distorted_points = distorted_constellation(constellation_points, T, f_0)

    plot_constellations(constellation_points, distorted_points)
    plot_decision_regions(constellation_points, q)