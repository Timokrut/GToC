import math
import matplotlib.pyplot as plt

def count_period(V_mod: int) -> float:
    return 1/V_mod

def find_amount_of_signals(V_inf: int, T: int) -> int:
    log_q = T * V_inf
    return 2 ** log_q 

def create_i_pairs(q: int) -> list[tuple]:
    length = int(q ** 0.5)
    result = []

    for i in range(length):
        for j in range(length):
            result.append((i, j))
    
    return result

def calculate_signal(S_i: tuple[float], T: float, f_0: int) -> tuple[list[float], list[float]]:
    Si_1, Si_2 = S_i 
    x = []
    y = []
    for t in range(1, 2400):
        part1 = Si_1 * (2 / T) ** 0.5 * math.cos(2 * math.pi * f_0 * ((T / 2400) * t))
        part2 = Si_2 * (2 / T) ** 0.5 * math.sin(2 * math.pi * f_0 * ((T / 2400) * t))
        x.append(T / 2400 * t)
        y.append(part1 + part2)

    return x, y 

def count_si_wo_A(i_pair: tuple, q: int) -> tuple:
    i_1, i_2 = i_pair

    si_1 = 1 - ((2 * i_1) / (q ** 0.5 - 1))
    si_2 = 1 - ((2 * i_2) / (q ** 0.5 - 1))
    
    return (si_1, si_2)

def count_energy_of_signal(S_i: tuple[float]):
    Si_1, Si_2 = S_i 
    return Si_1 ** 2 + Si_2 ** 2


if __name__ == "__main__":
    f_0 = 1800 # Hz 
    V_mod = 2400 #Boad
    V_inf = 12000 # bit/sec 

    T = count_period(V_mod)
    q = find_amount_of_signals(V_inf, T)
    i_pairs = create_i_pairs(q)
    all_x_array = []
    all_y_array = []
    for enum, i in enumerate(i_pairs):
        S_i = count_si_wo_A(i, q) 

        x_array, y_array = calculate_signal(S_i, T, f_0)
        [all_x_array.append(k + T * enum) for k in x_array]
        [all_y_array.append(k) for k in y_array]

        energy_of_signal = count_energy_of_signal(S_i)

    
    fig_class, axis_class = plt.subplots()
    axis_class.set_title(f"Graphic. Energy: {energy_of_signal}")
    axis_class.set_xlabel("t")
    axis_class.set_ylabel("Si(t)")
    plt.plot(all_x_array, all_y_array)

    plt.show()


        # with open(f'temp{enum}.txt', 'w') as f:
        #     for j in calculate_signal(S_i, count_period(V_mod), f_0):
        #         f.write(str(j)+"\n")