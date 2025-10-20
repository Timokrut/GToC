import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def count_T(V_mod: int) -> float:
    return 1 / V_mod


def count_q(V_inf: int, T: float) -> int:
    m = V_inf * T
    return int(2 ** round(m))


def create_i_pairs(q: int) -> list[tuple]:
    length = int(math.isqrt(q))
    result = []
    for i in range(length):
        for j in range(length):
            result.append((i, j))
    return result


def count_si_values(i_pair: tuple, q: int, A: float = 1.0) -> tuple:
    i_1, i_2 = i_pair
    L = math.isqrt(q) - 1
    si_1 = A * (1 - (2 * i_1) / L)
    si_2 = A * (1 - (2 * i_2) / L)
    return si_1, si_2


def calculate_signal(s_i: tuple, T: float, f_0: int, t: np.ndarray) -> np.ndarray:
    si_1, si_2 = s_i
    signal = (si_1 * math.sqrt(2 / T) * np.cos(2 * math.pi * f_0 * t) +
              si_2 * math.sqrt(2 / T) * np.sin(2 * math.pi * f_0 * t))
    return signal


def fourier_transform_kam(f: np.ndarray, s_i: tuple, T: float, f_0: int) -> np.ndarray:
    si_1, si_2 = s_i

    def sinc(x):
        return np.where(x == 0, 1.0, np.sin(np.pi * x) / (np.pi * x))

    term1 = si_1 * (T / 2) * (sinc((f - f_0) * T) + sinc((f + f_0) * T))
    term2 = si_2 * (T / 2) * (sinc((f - f_0) * T) - sinc((f + f_0) * T)) / 1j
    S_f = (term1 + term2) * np.exp(-1j * np.pi * f * T)
    return S_f


def get_analytical_expression(s_i: tuple, T: float, f_0: int) -> str:
    si_1, si_2 = s_i
    si_1_str = f"{si_1:.3f}"
    si_2_str = f"{si_2:.3f}"
    expr = (f"[{si_1_str}*(T/2)*(sinc((f-{f_0})*T) + sinc((f+{f_0})*T)) + "
            f"{si_2_str}*(T/2)*(sinc((f-{f_0})*T) - sinc((f+{f_0})*T))/j]*exp(-j*pi*f*T)")
    return expr


def calculate_bandwidth(T: float) -> float:
    return 2 / T


def calculate_signal_sequence_spectrum(sequence_indices: list, signals_spectra: list, frequencies: np.ndarray, T: float) -> np.ndarray:
    total_spectrum = np.zeros_like(frequencies, dtype=complex)
    for l, signal_idx in enumerate(sequence_indices):
        signal_spectrum = signals_spectra[signal_idx] * \
            np.exp(-2j * np.pi * frequencies * l * T)
        total_spectrum += signal_spectrum
    total_spectrum = total_spectrum / len(sequence_indices)
    return total_spectrum


if __name__ == '__main__':
    f_0 = 1200
    V_mod = 600
    V_inf = 2400
    A = 1.0
    Ns = 16 

    T = count_T(V_mod)
    q = count_q(V_inf, T)

    print("\nОБЩИЕ ПАРАМЕТРЫ:")
    print(f"Скорость модуляции V_mod: {V_mod} Бод")
    print(f"Информационная скорость V_inf: {V_inf} бит/сек")
    print(f"Период сигнала T: {T:.6f} сек")
    print(f"Количество сигналов q: {q}")
    print(f"Несущая частота f_0: {f_0} Гц")

    i_pairs = create_i_pairs(q)

    table_data = []
    for i, pair in enumerate(i_pairs):
        s_i = count_si_values(pair, q, A)
        fourier_expr = get_analytical_expression(s_i, T, f_0)
        table_data.append([i, pair[0], pair[1], f"{s_i[0]:.3f}", f"{s_i[1]:.3f}", fourier_expr])

    df = pd.DataFrame(table_data, columns=['№', 'i1', 'i2', 's_i1', 's_i2', 'Преобразование Фурье S_i(f)'])
    print("\nПОЛНАЯ ТАБЛИЦА СИГНАЛОВ:")
    print(df.to_string(index=False, justify='left'))

    dt = 1 / (f_0 * Ns)
    t = np.arange(0, T, dt)

    f_max = 6 * f_0
    N_freq = 4096
    frequencies = np.linspace(-f_max, f_max, N_freq)

    signals_spectras = []
    for i, pair in enumerate(i_pairs):
        s_i = count_si_values(pair, q, A)
        spectrum = fourier_transform_kam(frequencies, s_i, T, f_0)
        signals_spectras.append(spectrum)

    n_cols = 4
    n_rows = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 24))
    fig.suptitle(f'АМПЛИТУДНЫЕ СПЕКТРЫ ВСЕХ СИГНАЛОВ КАМ (q={q})', fontsize=16, y=0.98)
    axes = axes.flatten()

    for i in range(q):
        if i < len(axes):
            amplitude_spectrum = np.abs(signals_spectras[i])
            axes[i].plot(frequencies, amplitude_spectrum, 'b-', linewidth=1.5)
            axes[i].set_title(f'Сигнал {i} (i1={i_pairs[i][0]}, i2={i_pairs[i][1]})', fontsize=10)
            axes[i].set_xlabel('Частота (Hz)', fontsize=9)
            axes[i].set_ylabel('|S(f)|', fontsize=9)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(-f_max / 2, f_max / 2)

    plt.tight_layout(pad=9, h_pad=9, w_pad=9)
    plt.subplots_adjust(top=0.90)
    plt.show()

    bw = calculate_bandwidth(T)
    print(f"\nШИРИНА ПОЛОСЫ ЧАСТОТ ДЛЯ ОДИНОЧНЫХ СИГНАЛОВ:")
    print(f"Ширина полосы (2/T): {bw:.1f} Hz")
    print(f"Общая ширина полосы множества сигналов: {bw:.1f} Hz")

    sequences: list[list] = [
        [0, 1, 2, 3],
        [0, 1, 2, 3, 4, 5, 6, 7],
        list(range(16))
    ]

    sequence_lengths = [len(x) for x in sequences]
    sequence_spectra = []

    for seq_idx, sequence in enumerate(sequences):
        spectrum_sequence = calculate_signal_sequence_spectrum(sequence, signals_spectras, frequencies, T)
        sequence_spectra.append(spectrum_sequence)
        sequence_bw = 2 / (sequence_lengths[seq_idx] * T)
        print(f"Последовательность длиной {sequence_lengths[seq_idx]}: теоретическая ширина полосы = {sequence_bw:.1f} Hz")

    plt.figure(figsize=(12, 5))
    colors = ['blue', 'red', 'green']
    for i, (spectrum, length) in enumerate(zip(sequence_spectra, sequence_lengths)):
        amplitude = np.abs(spectrum)
        plt.plot(frequencies, amplitude, color=colors[i], label=f'Длина {length}', linewidth=2)
    plt.title('Спектры последовательностей различной длины')
    plt.xlabel('Частота (Hz)')
    plt.ylabel('|S(f)|')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(-3000, 3000)
    plt.tight_layout()
    plt.show()