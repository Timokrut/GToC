import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(100)

def theoretical_pe_func(q, SNR):
    Pe = 0.0
    for l in range(1, q):
        comb = math.comb(q - 1, l)
        Pe += comb * ((-1) ** (l + 1)) / (1 + l) * math.exp((-l / (l + 1)) * SNR)
    
    return Pe

def simulate(frequencies, T, E, gamma_dBs, max_errors=50, Nt=1000):
    t = np.linspace(0, T, Nt)
    
    # Базисные функции
    phi_cos = []
    phi_sin = []

    # Сигналы
    sig_cos = []
    sig_sin = []
        
    for f in frequencies:
        phi_cos.append(np.sqrt(2 / T) * np.cos(2 * np.pi * f * t))
        phi_sin.append(np.sqrt(2 / T) * np.sin(2 * np.pi * f * t))

        sig_cos.append(np.sqrt((2 * E) / T) * np.cos(2 * np.pi * f * t))
        sig_sin.append(np.sqrt((2 * E) / T) * np.sin(2 * np.pi * f * t))
    
    experimental_pe = []
    theoretical_pe = []
    
    for gamma_dB in gamma_dBs:
        SNR = 10 ** (gamma_dB / 10)
        N0 = E / SNR      # спектральная плотность шума
        sigma_t = math.sqrt(N0 * Nt / (2 * T))
    
        N_err = 0
        N_test = 0
        
        while N_err < max_errors:
            # Случайный выбор сигнала
            i = np.random.randint(0, q)
            
            # Случайная фаза
            theta = np.random.uniform(0, 2 * np.pi) # равномерное распределение
            
            # Формирование сигнала на выходе канала (без шума)
            # r(t) = cos(theta) * s_i_cos(t) + sin(theta) * s_i_sin(t)
            r_t = np.cos(theta) * sig_cos[i] + np.sin(theta) * sig_sin[i]
            
            # Генерация шума
            noise = np.random.normal(0, sigma_t, size=t.shape)
            r = r_t + noise
            
            # Моделирование приемника
            r_c = np.zeros(q)
            r_s = np.zeros(q)

            for j in range(q):
                r_c[j] = np.trapezoid(r * phi_cos[j], t)
                r_s[j] = np.trapezoid(r * phi_sin[j], t)
            
            # Формирование решения
            i_hat = np.argmax(r_c ** 2 + r_s ** 2)
            
            # Фиксация результата
            if i_hat != i:
                N_err += 1
            
            N_test += 1
            
        # Экспериментальная вероятность
        pe_exp = N_err / N_test
        experimental_pe.append(pe_exp)
        
        # Теоретическая вероятность
        pe_theor = theoretical_pe_func(q, SNR)
        theoretical_pe.append(pe_theor)
        print(f"Испытаний: {N_test}, Ошибок: {N_err}")
        print(f"P_e(эксп) = {pe_exp:.6f}, P_e(теор) = {pe_theor:.6f}")
    return experimental_pe, theoretical_pe

# Функция построения графиков
def plot_results(gamma_dB_values, experimental_pe, theoretical_pe):
    plt.figure(figsize=(10, 6))
    plt.semilogy(gamma_dB_values, experimental_pe, 'bo-', label='Экспериментальная')# лог шкала
    plt.semilogy(gamma_dB_values, theoretical_pe, 'r--', label='Теоретическая')
    plt.xlabel('Отношение сигнал/шум (E/N0), дБ')
    plt.ylabel('Вероятность ошибки')
    plt.title(f'Вероятность ошибки ')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(gamma_dB_values)
    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    # Параметры
    q = 2
    T = 5 * 10**(-6) # 5 мкс
    f0 = 10**6       # 10 МГц
    delta_f = 1 / T  # шаг по частоте
    E = 1

    frequencies = []
    for i in range(q):
        freq = f0 + i * delta_f
        frequencies.append(freq)
    
    # Диапазон SNR
    gamma_dBs = np.arange(0, 12, 2)
    print("МОДЕЛИРОВАНИЕ ПЕРЕДАЧИ ЧМ СИГНАЛОВ ПО КАНАЛУ СО СЛУЧАЙНОЙ ФАЗОЙ")
    print(f"Параметры: q = {q}, T = {T * 1e6:.2f} мкс, f0 = {f0 / 1e6:.2f} МГц, delta_f = {delta_f / 1e3:.2f} кГц")
    print(f"Частоты (МГц): {[f / 1e6 for f in frequencies]}")
    
    # Запуск моделирования
    experimental_pe, theoretical_pe = simulate(frequencies, T, E, gamma_dBs, max_errors=100)
    
    # Построение графиков
    plot_results(gamma_dBs, experimental_pe, theoretical_pe)
