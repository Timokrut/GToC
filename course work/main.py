import numpy as np
import matplotlib.pyplot as plt
np.random.seed(50)

# МОДЕЛИРОВАНИЕ ОФМ-2
# Канал: АБГШ + случайная фаза
# Параметры моделирования
Nerr_max = 100                    # максимальное число ошибок
gamma_dBs = np.arange(0, 10, 2)   # диапазон SNR в dB

# Параметры случайной фазы канала
alpha = 0.002
beta = 0.0005

# Число состояний ОФМ-2
q = 2

# Параметры системы
E = 1
T = 5 * 10**(-6) # 5 мкс
f0 = 10 * 10**6  # 10 МГц

# Дискретизация
Nt = 1000
t = np.linspace(0, T, Nt, endpoint=False)

# Базисные функции
Sc = np.sqrt(2 * E / T) * np.cos(2 * np.pi * f0 * t)
Ss = np.sqrt(2 * E / T) * np.sin(2 * np.pi * f0 * t)

# Массивы результатов
Pe_exp = []
Pe_theory = []

# ЦИКЛ ПО ЗНАЧЕНИЯМ SNR
for gamma_dB in gamma_dBs:
    # Перевод из dB в линейный масштаб
    gamma = 10 ** (gamma_dB / 10)

    # Для нормированной энергии Eb = 1:
    # sigma^2 = N0/2 = 1/(2*gamma)
    sigma2 = 1 / (2 * gamma)
    sigma = np.sqrt(sigma2)

    # Счетчики - число ошибок / число испытаний
    Nerr = 0
    l = 0

    # Начальные фазы
    theta_prev = 0.0
    Theta_prev = 0.0

    # Начальная фаза канала
    phi_prev = np.random.uniform(0, 2 * np.pi)

    # ЦИКЛ МОДЕЛИРОВАНИЯ
    while Nerr < Nerr_max:
        # ПЕРЕДАТЧИК

        # Случайный бит
        i = np.random.randint(0, 2)

        # ОФМ-2:
        # 0 -> 0
        # 1 -> pi
        delta_theta = i * np.pi

        # Текущая фаза сигнала
        theta = theta_prev + delta_theta 

        # Сохраняем фазу
        theta_prev = theta

        # СЛУЧАЙНАЯ ФАЗА КАНАЛА
        eps = np.random.uniform(-np.pi, np.pi)
        phi = (phi_prev + alpha * eps + beta) % (2 * np.pi)
        phi_prev = phi

        # КАНАЛ + ШУМ

        # сигнал r без шума
        r_clean = np.cos(theta + phi) * Sc + np.sin(theta + phi) * Ss

        # Добавление АБГШ
        dt = T / Nt
        noise = sigma * np.random.randn(Nt) / np.sqrt(dt)
        r = r_clean + noise

        # ПРИЕМНИК
        rc = np.trapezoid(r * Sc, t)
        rs = np.trapezoid(r * Ss, t)

        # Оценка фазы принятого сигнала
        Theta = np.arctan2(rs, rc)

        # Разность фаз
        dTheta = Theta - Theta_prev
        dTheta = (dTheta + np.pi) % (2 * np.pi) - np.pi # нормализация в диапазон [-pi, pi]

        # ПРИНЯТИЕ РЕШЕНИЯ

        # Если ближе к 0 -> бит 0
        # Если ближе к pi -> бит 1
        if abs(dTheta) < np.pi / 2:
            i_hat = 0
        else:
            i_hat = 1

        # Сохраняем фазу для следующего символа
        Theta_prev = Theta

        # ПОДСЧЕТ ОШИБОК
        if i != i_hat:
            Nerr += 1

        l += 1

    # Экспериментальная вероятность ошибки
    Pe = Nerr / l
    Pe_exp.append(Pe)

    # Теоретическая вероятность ошибки
    Pe_t = 0.5 * np.exp(-gamma)
    Pe_theory.append(Pe_t)

    print(f"SNR = {gamma_dB:2d} dB | Pe_exp = {Pe:.6f} | Pe_theory = {Pe_t:.6f}")

# ПОСТРОЕНИЕ ГРАФИКА
plt.figure(figsize=(8, 6))
plt.semilogy(gamma_dBs, Pe_exp, 'o-', linewidth=2,
             label='Экспериментальная')
plt.semilogy(gamma_dBs, Pe_theory, 's--', linewidth=2,
             label='Теоретическая')
plt.grid(True, which='both')
plt.xlabel('Eb/N0, dB')
plt.ylabel('Вероятность ошибки Pe')
plt.title('ОФМ-2 в канале со случайной фазой')
plt.legend()
plt.show()