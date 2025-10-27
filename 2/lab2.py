import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def count_T(V_mod: int) -> float:
    return 1 / V_mod

def count_q(V_inf: int, T: float) -> int:
    log_q = V_inf * T
    return int(2 ** log_q)

def create_i_pairs(q: int) -> list[tuple]:
    length = int(math.isqrt(q))
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

def fourier_transform_qam(f: np.ndarray, s_i: tuple, T: float, f_0: int) -> np.ndarray:
    si_1, si_2 = s_i

    def sinc(x):
        return np.where(x == 0, 1.0, np.sin(np.pi * x) / (np.pi * x))

    term1 = si_1 * math.sqrt(T / 2) * (sinc((f - f_0) * T) + sinc((f + f_0) * T))
    term2 = si_2/1j * math.sqrt(T / 2) * (sinc((f - f_0) * T) - sinc((f + f_0) * T))
    S_f = (term1 + term2) * np.exp(-1j * np.pi * f * T)
    return S_f

def count_W(T: float) -> float:
    return 2 / T

def count_seq(sequence_indices: list, signals_spectra: list, frequencies: np.ndarray, T: float) -> np.ndarray:
    total_spectrum = np.zeros_like(frequencies, dtype=complex)
    for l, signal_idx in enumerate(sequence_indices):
        signal_spectrum = signals_spectra[0] * np.exp(-2j * np.pi * frequencies * l * T)
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

    print(f"V_mod: {V_mod} Бод")
    print(f"V_inf: {V_inf} бит/с")
    print(f"T: {T:.6f} сек")
    print(f"q: {q}")
    print(f"f_0: {f_0} Гц")

    i_pairs = create_i_pairs(q)

    dt = 1 / (f_0 * Ns)
    t = np.arange(0, T, dt)

    f_max = 6 * f_0
    N_freq = 4096 
    frequencies = np.linspace(-f_max, f_max, N_freq)

    signals_spectras = []
    for i, pair in enumerate(i_pairs):
        s_i = count_si_values(pair, q, A)
        spectrum = fourier_transform_qam(frequencies, s_i, T, f_0)
        signals_spectras.append(spectrum)

    n_cols = 4
    n_rows = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 24))
    fig.suptitle(f'Амплитудные спектры сигналов', fontsize=16, y=0.98)
    axes = axes.flatten()

    for i in range(q):
        if i < len(axes):
            amplitude_spectrum = np.abs(signals_spectras[i])
            axes[i].plot(frequencies, amplitude_spectrum, 'b-', linewidth=1.5)
            axes[i].set_title(f'Сигнал {i} ({i_pairs[i][0]}, {i_pairs[i][1]})', fontsize=10)
            axes[i].set_xlabel('Частота, Гц', fontsize=9)
            axes[i].set_ylabel('|S(f)|', fontsize=9)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(-f_max / 2, f_max / 2)

    plt.tight_layout(pad=9, h_pad=9, w_pad=9)
    plt.subplots_adjust(top=0.90)
    plt.show()

    bw = count_W(T)
    print(f"Ширина полосы W: {bw:.1f} Гц")
    print(f"Общая ширина полосы множества сигналов: {bw:.1f} Гц")

    sequences: list[list] = [
        list(range(1)),
        list(range(4)),
        list(range(8)),
        list(range(16))
    ]

    sequence_lengths = [len(x) for x in sequences]
    sequence_spectra = []

    for seq_idx, sequence in enumerate(sequences):
        spectrum_sequence = count_seq(sequence, signals_spectras, frequencies, T)
        sequence_spectra.append(spectrum_sequence)
        sequence_bw = 2 / (sequence_lengths[seq_idx] * T)
        print(f"Последовательность длиной {sequence_lengths[seq_idx]}: ширина полосы = {sequence_bw:.1f} Гц")

    plt.figure(figsize=(12, 5))
    for i, (spectrum, length) in enumerate(zip(sequence_spectra, sequence_lengths)):
        amplitude = np.abs(spectrum)
        plt.plot(frequencies, amplitude, label=f'Длина {length}', linewidth=2)
    plt.title('Спектры последовательностей различной длины')
    plt.xlabel('Частота, Гц')
    plt.ylabel('|S(f)|')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(-3000, 3000)
    plt.tight_layout()
    plt.show()
