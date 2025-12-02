import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate

def count_T(V_mod: int) -> float:
    return 1 / V_mod

def count_q(V_inf: int, T: float) -> int:
    log_q = V_inf * T
    return int(2 ** log_q)

def phi_1(t, T, f_0):
    return np.sqrt(2 / T) * np.cos(2 * np.pi * f_0 * t)

def phi_2(t, T, f_0):
    return np.sqrt(2 / T) * np.sin(2 * np.pi * f_0 * t)

def count_E_avg(constellation_points) -> float:
    E_sum = 0
    for point in constellation_points:
        E_sum += point[0] ** 2 + point[1] ** 2
    return E_sum / q

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

def signal_points(constellation_points, T, f_0, N=1000):
    t = np.linspace(0, T, N)

    phi1 = phi_1(t, T, f_0) 
    phi2 = phi_2(t, T, f_0)

    distorted = []

    for (s1, s2) in constellation_points:
        s = generate_signal(s1, s2, phi1, phi2)
        
        d1 = project_onto_basis(s, phi1, t)
        d2 = project_onto_basis(s, phi2, t)
        
        distorted.append((d1, d2))

    return distorted

def Q(x):
    """
    Q(x) = (1/√(2π)) ∫ₓ^∞ e^(-z²/2) dz
    """
    def integrand(z):
        return math.exp(-z ** 2 / 2) / math.sqrt(2 * math.pi)
    
    integration_result = integrate.quad(integrand, x, np.inf)
    return integration_result[0]  # Берем только значение интеграла
    
def theoretical_pe_kam(q, E_avg_over_N0):
    """
    P_e ≤ [1 - (1 - 2(1 - 1/M) * Q(√(3E/((M²-1)N₀))))²]
    """
    M = int(math.sqrt(q))

    # Спектральная плотность мощности шума N₀ = 1 (нормировка)
    N0 = 1.0
    
    # Аргумент функции Q
    argument = math.sqrt(3 * E_avg_over_N0 / ((M ** 2 - 1) * N0))
    Q_val = Q(argument)
    Pe_exact = 1 - (1 - 2 * (1 - 1 / M) * Q_val) ** 2
    return Pe_exact

# правило минимального расстояния,макс правдоподобия
def optimal_point(r, signal_points):
    min_dist = float('inf')
    decision = 0    
    for idx, point in enumerate(signal_points):
        dist = math.sqrt((r[0] - point[0]) ** 2 + (r[1] - point[1]) ** 2)
        if dist < min_dist:
            min_dist = dist
            decision = idx
    return decision



def simulate_communication_system_full(constellation_points, q, T, f_0, gamma_dB_values, max_errors=100, N=1000):
    experimental_pe = []
    theoretical_pe = []

    t = np.linspace(0, T, N)
    
    signal_points_base = signal_points(constellation_points, T, f_0)
    E_avg = count_E_avg(constellation_points)
    phi1 = phi_1(t, T, f_0) 
    phi2 = phi_2(t, T, f_0)

    # Цикл по значениям отношения сигнал/шум в дБ
    for gamma_dB in gamma_dB_values:
        gamma = 10 ** (gamma_dB / 10) # отношение сигнал шум 
                
        # Отношение сигнал/шум: γ = E_avg / N0
        N0 = E_avg / gamma  # спектральная плотность мощности
        sigma_square = N0 / 2 # дисперсия отсчета шума

        print(f"\nSNR = {gamma_dB} dB, N0 = {N0:.6f}, sigma^2 = {sigma_square:.6f}")

        N_err = 0  # число ошибок
        N_test = 0 # число испытний
        N_errmax = 500 if gamma_dB >= 10 else max_errors

        received_points = []

        # Цикл моделирования при одном значении отношения сигнал/шум
        while N_err < N_errmax and N_test < 50000:
            # Случайный выбор номера сигнала
            i = np.random.randint(0, q)  
            
            # Генерация АБГШ во временной области
            s_i1, s_i2 = constellation_points[i]
            signal_i = generate_signal(s_i1, s_i2, phi1, phi2)
            noise_t = np.random.normal(0, 1, len(t)) * math.sqrt(N0 * len(t) / (2 * T))
            r_t = signal_i + noise_t
 
            # Вычисление вектора r с компонентами r_j = ∫r(t)φ_j(t)dt
            r1 = np.trapezoid(r_t * phi1, t)
            r2 = np.trapezoid(r_t * phi2, t)
            r_vector = (r1, r2)
            received_points.append(r_vector)
            
            # Формирование решения по правилу минимального расстояния
            i_opt = optimal_point(r_vector, signal_points_base)
            
            if i_opt != i:
                N_err += 1
            N_test += 1
 
        # Вычисление экспериментальной вероятности ошибки
        P_e_experimental = N_err / N_test
        experimental_pe.append(P_e_experimental)

        # Теоретическая вероятность ошибки
        P_e_theoretical = theoretical_pe_kam(q, gamma)
        theoretical_pe.append(P_e_theoretical)
        print(f"Испытаний: {N_test}, Ошибок: {N_err}")
        print(f"P_e(эксп) = {P_e_experimental:.6f}, P_e(теор) = {P_e_theoretical:.6f}")

        if gamma_dB in [0, 6, 12]:
            plt.figure(figsize=(12, 8))
        
            # Принятые точки
            if received_points:
                rx, ry = zip(*received_points)
                plt.plot(rx, ry, 'bo', markersize=3, alpha=0.6, label='Принятые точки')

            # Сигнальные точки           
            for idx, point in enumerate(signal_points_base):
                plt.plot(point[0], point[1], 'ro', markersize=8)
                plt.text(point[0] + 0.05, point[1] + 0.05, f's{idx}', fontsize=10)

            plt.title(f'Диаграмма рассеивания при SNR = {gamma_dB} дБ\nИспытаний: {N_test}, Ошибок: {N_err}')
            plt.xlabel('r₁ (проекция на φ₁(t))')
            plt.ylabel('r₂ (проекция на φ₂(t))')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.tight_layout()
            plt.show()

    return experimental_pe, theoretical_pe

def probability_results(gamma_dB_values, experimental_pe, theoretical_pe):
    plt.figure(figsize=(8, 6))

    # График вероятности ошибки
    plt.semilogy(gamma_dB_values, experimental_pe, 'bo-', label='Экспериментальная', markersize=6)
    plt.semilogy(gamma_dB_values, theoretical_pe, 'r--', label='Теоретическая', linewidth=2)

    plt.xlabel('Отношение сигнал/шум (Eb/N0, дБ)')
    plt.ylabel('Вероятность ошибки на символ')
    plt.title('Сравнение теоретической и экспериментальной вероятности ошибки')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(gamma_dB_values)

    plt.tight_layout()
    plt.show()
    
    results_df = pd.DataFrame({
        'SNR (dB)': gamma_dB_values,
        'P_e (эксп)': experimental_pe,
        'P_e (теор)': theoretical_pe
    })
    print("\nРезультаты моделирования:")
    print(results_df.to_string(index=False, float_format='%.6f'))


if __name__ == '__main__':
    f_0 = 1200
    V_mod = 600
    V_inf = 2400
    A = 1.0

    T = count_T(V_mod)
    q = count_q(V_inf, T)
    constellation_points, i_pairs = create_signal_constellation(q)
    
    gamma_dB_values = np.arange(0, 16, 2)
    
    experimental_pe, theoretical_pe = simulate_communication_system_full(constellation_points, q, T, f_0, gamma_dB_values, max_errors=1000)
    
    probability_results(gamma_dB_values, experimental_pe, theoretical_pe)