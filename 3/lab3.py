import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def count_T(V_mod: int) -> float:
    return 1 / V_mod

def count_q(V_inf: int, T: float) -> int:
    log_q = V_inf * T
    return int(2 ** log_q)

def check_orthonormality(T: float, f_0: int, N: int = 1000) -> dict:
    t = np.linspace(0, T, N)
    # Базисные функции для КАМ

    phi1 = np.sqrt(2 / T) * np.cos(2 * math.pi * f_0 * t)
    phi2 = np.sqrt(2 / T) * np.sin(2 * math.pi * f_0 * t)
    
    # Вычисление скалярных произведений
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

    # Вычисление координат сигнальных точек
    for i1 in range(L):
        for i2 in range(L):
            s_i1 = A * (1 - (2 * i1) / (L - 1))
            s_i2 = A * (1 - (2 * i2) / (L - 1))
            constellation_points.append((s_i1, s_i2))
            i_pairs.append((i1, i2))

    return constellation_points, i_pairs

def plot_constellation(constellation_points: list, q: int):
    # Создаем график
    plt.figure(figsize=(8, 8))
    for i, point in enumerate(constellation_points):
        x = point[0] # s_i1 - координата X
        y = point[1] # s_i2 - координата Y
        # Рисуем точку
        plt.plot(x, y, 'ro', markersize=8) # красный кружок
        # Подписываем точку
        plt.text(x + 0.05, y + 0.05, f's{i}', fontsize=10)
    # Оформление
    plt.axhline(0, color='gray', linestyle='--') # горизонтальная ось через 0
    plt.axvline(0, color='gray', linestyle='--') # вертикальная ось через 0
    plt.grid(True, alpha=0.3)
    plt.xlabel('s_i1 (амплитуда cos)')
    plt.ylabel('s_i2 (амплитуда sin)')
    plt.title(f'Созвездие КАМ - {q} сигналов')
    plt.axis('equal')#автоматическ утсанов прав пределы
    plt.show()

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def plot_decision_regions(constellation_points: list, q: int):
    # Создаем график
    plt.figure(figsize=(8, 8))

    # Определяем границы для сетки
    margin = 0.5
    x_min = min(p[0] for p in constellation_points) - margin
    x_max = max(p[0] for p in constellation_points) + margin
    y_min = min(p[1] for p in constellation_points) - margin
    y_max = max(p[1] for p in constellation_points) + margin
    
    # Создаем сетку точек
    step = 0.02
    x_vals = np.arange(x_min, x_max, step)
    y_vals = np.arange(y_min, y_max, step)
    X, Y = np.meshgrid(x_vals, y_vals)#сетка, дублируются xvals yvals
    
    # Для каждой точки сетки определяем ближайшую сигнальную точку
    Z = np.zeros(X.shape)#номера сигнальн точек ближ
    for i in range(X.shape[0]):#все точки сетки
        for j in range(X.shape[1]):
            point = (X[i, j], Y[i, j])
            min_dist = float('inf')
            closest_point_idx = 0
            
            # Находим ближайшую сигнальную точку для кажд точки сигнальн простр
            for idx, signal_point in enumerate(constellation_points):
                dist = distance(point, signal_point)
                if dist < min_dist:
                    min_dist = dist
                    closest_point_idx = idx

            Z[i, j] = closest_point_idx
    plt.contourf(X, Y, Z, levels=np.arange(-0.5, q, 1), alpha=0.3, cmap='tab20')

    # И contour для тонких границ
    plt.contour(X, Y, Z, levels=np.arange(-0.5, q, 1), colors='black', linewidths=0.5)
    
    # Рисуем сигнальные точки
    for i, point in enumerate(constellation_points):
        plt.plot(point[0], point[1], 'ro', markersize=8)
        plt.text(point[0] + 0.05, point[1] + 0.05, f's{i}', fontsize=10)
    
    # Оформление
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.xlabel('s_i1 (амплитуда cos)')
    plt.ylabel('s_i2 (амплитуда sin)')
    plt.title(f'Решающие области (Вороного) - {q} сигналов')
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    f_0 = 1200 # Hz
    V_mod = 600 # Baud
    V_inf = 2400 # bit/sec
    A = 1.0

    print("ГЕОМЕТРИЧЕСКОЕ ПРЕДСТАВЛЕНИЕ СИГНАЛОВ КАМ")

    # 1. Выбор базисных функций
    print("\n1. ВЫБОР БАЗИСНЫХ ФУНКЦИЙ")
    print("Базисные функции для КАМ:")
    print("φ1(t) = √(2/T) * cos(2πf₀t)")
    print("φ2(t) = √(2/T) * sin(2πf₀t)")
    print(f"где f₀ = {f_0} Гц")

    # Вычисление параметров
    T = count_T(V_mod)
    q = count_q(V_inf, T)
    
    print(f"\nПараметры:")
    print(f"Период T = {T:.6f} сек")
    print(f"Количество сигналов q = {q}")
    
    # 2. Проверка ортонормированности базисных функций
    print("\n2. ПРОВЕРКА ОРТОНОРМИРОВАННОСТИ БАЗИСНЫХ ФУНКЦИЙ")
    ortho_results = check_orthonormality(T, f_0)
    
    print("Результаты проверки условий (3.1):")
    for key, value in ortho_results.items():
        print(f"{key} = {value:.6f}")

    print("\nТеоретически ожидаемые значения:")
    print("(φ1, φ1) = 1.000000")
    print("(φ2, φ2) = 1.000000")
    print("(φ1, φ2) = 0.000000")
    print("(φ2, φ1) = 0.000000")

    # 3. Построение сигнального созвездия
    print("\n3. ПОСТРОЕНИЕ СИГНАЛЬНОГО СОЗВЕЗДИЯ")
    constellation_points, i_pairs = create_signal_constellation(q, A)

    # Создание DataFrame
    data = []
    for i, ((i1, i2), (s1, s2)) in enumerate(zip(i_pairs, constellation_points)):
        data.append([i, i1, i2, f"{s1:.3f}", f"{s2:.3f}"])
    
    df = pd.DataFrame(data, columns=['№', 'i1', 'i2', 's_i1', 's_i2'])
    
    print("Координаты сигнальных точек:")
    print(df.to_string(index=False))
    
    # Визуализация созвездия
    plot_constellation(constellation_points, q)
    
    # 4. Построение решающих областей
    print("\n4. ПОСТРОЕНИЕ РЕШАЮЩИХ ОБЛАСТЕЙ (ОБЛАСТЕЙ ВОРОНОГО)")
    print("Решающие области определяются по правилу (3.7):")
    print("R_i = {r: d(r, s_i) < d(r, s_k) для всех k ≠ i}")
    print("где d(r, s_i) - евклидово расстояние между точкой r и сигнальной точкой s_i")
    plot_decision_regions(constellation_points, q)