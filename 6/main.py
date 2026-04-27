import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(100)

def theoretical_pe_func(q, SNR, epsilon):
    Pe = 0.0
    for l in range(1, q):
        sign = (-1) ** (l + 1)
        comb = math.comb(q - 1, l)
        
        denumerator = 1 + l + l * (1 - epsilon) * SNR
        numerator = math.exp(-(l * epsilon * SNR) / (1 + l + l * (1 - epsilon) * SNR))
        Pe += comb * sign * numerator / denumerator
    return Pe

def simulate(frequencies, T, E, gamma_dBs, epsilon, max_errors=50, Nt=1000):
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
        sigma_t = math.sqrt((N0 / 2) *  (Nt / T))
    
        N_err = 0
        N_test = 0
        
        while N_err < max_errors:
            # Случайный выбор сигнала
            i = np.random.randint(0, q)
            
            # Случайная фаза
            theta = np.random.uniform(0, 2 * np.pi) # равномерное распределение
            
            # Генерация коэффициента передачи канала μ
            M = math.sqrt(epsilon / 2)
            D = math.sqrt((1 - epsilon) / 2)
            x = np.random.normal(M, D)
            y = np.random.normal(M, D)
            mu = math.sqrt(x ** 2 + y ** 2)
        
            # Формирование сигнала с учётом замираний
            r_t = mu * (np.cos(theta) * sig_cos[i] + np.sin(theta) * sig_sin[i])
            
            # Генерация шума
            noise = np.random.normal(0, sigma_t, Nt)
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
        pe_theor = theoretical_pe_func(q, SNR, epsilon)
        theoretical_pe.append(pe_theor)

        print(f"Испытаний: {N_test}, Ошибок: {N_err}, SNR={SNR}")
        print(f"P_e(эксп) = {pe_exp:.6f}, P_e(теор) = {pe_theor:.6f}")
    return experimental_pe, theoretical_pe
    
def plot_results(gamma_dBs, all_experimental, all_theoretical, epsilon_values):
    plt.figure(figsize=(12, 7))

    colors = ['blue', 'red', 'green']

    for idx, eps in enumerate(epsilon_values):
        color = colors[idx]
        plt.semilogy(gamma_dBs, all_experimental[idx],
                     color=color, marker='o', linestyle='-',
                     linewidth=2, markersize=6,
                     label=f'ε = {eps} (эксп)')
        
        plt.semilogy(gamma_dBs, all_theoretical[idx],
                     color=color, linestyle='--',
                     linewidth=1.5, alpha=0.7,
                     label=f'ε = {eps} (теор)')

    plt.xlabel('SNR, дБ', fontsize=12)
    plt.ylabel('Pe', fontsize=12)
    plt.title('Передача ЧМ сигналов по каналу с замираниями', fontsize=14)
    plt.legend()
    plt.xticks(gamma_dBs)
    plt.grid()
    plt.ylim(1e-6, 1)
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
    
    # Построение графиков
    all_experimental = []
    all_theoretical = []

    # параметр замираний e
    # ε = 1 - канал без замираний
    # ε ∈ (0, 1) - канал Райса, есть прямая и рассеянная компоненты
    # ε = 0 - канал Релея, только рассеянная компонента
    epsilon_values = [0, 0.5, 1.0]

    for eps in epsilon_values:
        print(f"\nМоделирование для ε = {eps}")
        experimental_pe, theoretical_pe = simulate(frequencies, T, E, gamma_dBs, eps, max_errors=100)
        
        all_experimental.append(experimental_pe)
        all_theoretical.append(theoretical_pe)

    # Построение общего графика
    plot_results(gamma_dBs, all_experimental, all_theoretical, epsilon_values)
