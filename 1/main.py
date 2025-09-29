import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def count_T(V_mod: int) -> float:
    return 1 / V_mod

def count_q(V_inf: int, T: float) -> float:
    log_q = V_inf * T
    return 2 ** log_q

def create_i_pairs(q: float) -> list[tuple]:
    length = int(math.sqrt(q))
    result = []
    for i in range(length):
        for j in range(length):
            result.append((i, j))
    return result

def count_si_values(i_pair: tuple, q: float, A: float = 1.0) -> tuple:
    i_1, i_2 = i_pair
    si_1 = A * (1 - (2 * i_1) / (math.sqrt(q) - 1))
    si_2 = A * (1 - (2 * i_2) / (math.sqrt(q) - 1))
    return si_1, si_2

def calculate_signal(s_i: tuple, T: float, f_0: int, t: np.ndarray) -> np.ndarray:
    si_1, si_2 = s_i
    signal = (si_1 * math.sqrt(2 / T) * np.cos(2 * math.pi * f_0 * t) +
              si_2 * math.sqrt(2 / T) * np.sin(2 * math.pi * f_0 * t))
    return signal

def calculate_energy_numerical(signal: np.ndarray, dt: float):
    return np.sum(signal ** 2) * dt

def calculate_energy_analytical(s_i: tuple[float, float]) -> float: 
    si_1, si_2 = s_i 
    return si_1**2 + si_2**2 


if __name__ == '__main__':
    f_0 = 1200  # Hz
    V_mod = 600  # Baud
    V_inf = 2400  # bit/sec

    A = 1.0  # Амплитуда
    Ns = 128  # Количество samples на период несущей

    # Вычисление основных параметров
    T = count_T(V_mod)
    q = count_q(V_inf, T)

    # Дискретизация времени
    dt = 1 / (f_0 * Ns)
    t = np.arange(0, T, dt)

    print(f"Скорость модуляции V_mod: {V_mod} Бод")
    print(f"Информационная скорость V_inf: {V_inf} бит/сек")
    print(f"Период сигнала T: {T:.6f} сек")
    print(f"Количество сигналов q: {q}")
    print(f"Количество отсчетов на период: {len(t)}")
    print(f"Шаг дискретизации dt: {dt:.6f} сек")

    i_pairs = create_i_pairs(q)
    print(f"\nПары (i1, i2): {i_pairs}")

    print("\nАНАЛИТИЧЕСКИЕ ВЫРАЖЕНИЯ:")
    for i, pair in enumerate(i_pairs):
        s_i = count_si_values(pair, q, A)
        print(f"Сигнал {i}: s(t) = {s_i[0]:.3f}·√(2/T)·cos(2π·{f_0}·t) + {s_i[1]:.3f}·√(2/T)·sin(2π·{f_0}·t)")

    print("\nЭНЕРГИИ СИГНАЛОВ (численный расчет):")
    numerical_energies = []
    analytical_energies = []
    signals = []
    for i, pair in enumerate(i_pairs):
        s_i = count_si_values(pair, q, A)
        signal = calculate_signal(s_i, T, f_0, t)
        signals.append(signal)
        energy_numerical = calculate_energy_numerical(signal, dt)
        numerical_energies.append(energy_numerical)
        energy_analytical = calculate_energy_analytical(s_i) 
        analytical_energies.append(energy_analytical)
        print(f"Сигнал {i} ({pair}): E = {energy_numerical:.3f}")

    # === ТАБЛИЦА СИГНАЛОВ === 
    table = [] 
    for i, pair in enumerate(i_pairs): 
        s_i = count_si_values(pair, q, A) 
        s_expr = f"{s_i[0]:.3f}*√(2/T)*cos(2π*{f_0}*t) + {s_i[1]:.3f}*√(2/T)*sin(2π*{f_0}*t)" 
        table.append([i, pair[0], pair[1], s_i[0], s_i[1], numerical_energies[i], analytical_energies[i], s_expr]) 
 
    columns = ["i", "i1", "i2", "s_i1", "s_i2", "E_числ", "E_теор", "Формула"] 
    df = pd.DataFrame(table, columns=columns) 
    print("\nПОЛНАЯ ТАБЛИЦА СИГНАЛОВ:") 
    print(df.to_string(index=False)) 
 

    fig, axes = plt.subplots(4, 4, figsize=(15, 10))
    fig.suptitle(f'Сигналы КАМ (q={q}, V_mod={V_mod} Бод, V_inf={V_inf} бит/сек)', fontsize=16)

    for i, pair in enumerate(i_pairs):
        row = i // 4
        col = i % 4
        axes[row, col].plot(t, signals[i])
        axes[row, col].set_title(f'Сигнал {i}: ({pair[0]},{pair[1]})')
        axes[row, col].set_xlabel('Время (сек)')
        axes[row, col].set_ylabel('Амплитуда')
        axes[row, col].grid(True)

    plt.tight_layout()
    plt.show() 

    # график энергий номер сигнала по оси икс, эенергии по y 
    plt.figure(figsize=(10, 5)) 
    plt.plot(numerical_energies, 'o-', label='E (дискретно)', markersize=4, linewidth=1) 
    plt.plot(analytical_energies, '--', label='E (теоретически)', linewidth=2)
    plt.title(f"Сравнение энергий всех {q} сигналов") 
    plt.xlabel("Номер сигнала i") 
    plt.ylabel("Энергия") 
    plt.grid(True, alpha=0.3) 
    plt.legend() 
    plt.tight_layout() 
    plt.show()