import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class SignalParameters:
    """Класс для хранения параметров сигнала"""
    f_0: int = 1200
    V_mod: int = 600
    V_inf: int = 2400
    A: float = 1.0
    Ns: int = 16

class QAMSignal:
    """Класс для представления одного QAM сигнала"""
    
    def __init__(self, i_pair: Tuple[int, int], parameters: SignalParameters):
        self.i_pair: tuple[int, int] = i_pair
        self.params = parameters
        self.s_i: tuple[float, float] = None
        self.signal = None
        self.energy_numerical = None
        self.energy_analytical = None
        
        self._calculate_si_values()
    
    def _calculate_si_values(self) -> None:
        """Вычисление коэффициентов s_i1 и s_i2"""
        i_1, i_2 = self.i_pair
        q = self._calculate_q()
        sqrt_q = math.sqrt(q)
        
        self.s_i = (
            self.params.A * (1 - (2 * i_1) / (sqrt_q - 1)),
            self.params.A * (1 - (2 * i_2) / (sqrt_q - 1))
        )
    
    def _calculate_q(self) -> float:
        """Вычисление количества сигналов"""
        T = 1 / self.params.V_mod
        log_q = self.params.V_inf * T
        return 2 ** log_q
    
    def generate_signal(self, t: np.ndarray) -> None:
        """Генерация сигнала во временной области"""
        if self.s_i is None:
            self._calculate_si_values()
            
        T = 1 / self.params.V_mod
        si_1, si_2 = self.s_i
        
        self.signal = (
            si_1 * math.sqrt(2 / T) * np.cos(2 * math.pi * self.params.f_0 * t) +
            si_2 * math.sqrt(2 / T) * np.sin(2 * math.pi * self.params.f_0 * t)
        )
    
    def calculate_energy_numerical(self, dt: float) -> float:
        """Численный расчет энергии сигнала"""
        if self.signal is None:
            raise ValueError("Сначала сгенерируйте сигнал с помощью generate_signal()")
        
        self.energy_numerical = np.sum(self.signal ** 2) * dt
        return self.energy_numerical
    
    def calculate_energy_analytical(self) -> float:
        """Аналитический расчет энергии сигнала"""
        if self.s_i is None:
            self._calculate_si_values()
            
        si_1, si_2 = self.s_i
        self.energy_analytical = si_1**2 + si_2**2
        return self.energy_analytical
    
    def get_formula_string(self) -> str:
        """Получить строковое представление формулы сигнала"""
        if self.s_i is None:
            self._calculate_si_values()
            
        return (f"{self.s_i[0]:.3f}·√(2/T)·cos(2π·{self.params.f_0}·t) + "
                f"{self.s_i[1]:.3f}·√(2/T)·sin(2π·{self.params.f_0}·t)")

class QAMSystem:
    """Класс для управления системой QAM сигналов"""
    
    def __init__(self, parameters: Optional[SignalParameters] = None):
        self.params = parameters or SignalParameters()
        self.signals: List[QAMSignal] = []
        self.time_vector: Optional[np.ndarray] = None
        self.dt: Optional[float] = None
        
        self._initialize_system()
    
    def _initialize_system(self) -> None:
        """Инициализация системы: создание всех сигналов"""
        q = self._calculate_total_signals()
        length = int(math.sqrt(q))
        
        # Создание всех возможных пар (i1, i2)
        i_pairs = []
        for i in range(length):
            for j in range(length):
                i_pairs.append((i, j))
        
        # Создание объектов сигналов
        self.signals = [QAMSignal(pair, self.params) for pair in i_pairs]
        
        # Инициализация временного вектора
        self._initialize_time_vector()
    
    def _calculate_total_signals(self) -> float:
        """Вычисление общего количества сигналов"""
        T = 1 / self.params.V_mod
        log_q = self.params.V_inf * T
        return 2 ** log_q
    
    def _initialize_time_vector(self) -> None:
        """Инициализация вектора времени"""
        T = 1 / self.params.V_mod
        self.dt = 1 / (self.params.f_0 * self.params.Ns)
        self.time_vector = np.arange(0, T, self.dt)
    
    def generate_all_signals(self) -> None:
        """Генерация всех сигналов системы"""
        for signal in self.signals:
            signal.generate_signal(self.time_vector)
    
    def calculate_all_energies(self) -> None:
        """Расчет энергий для всех сигналов"""
        for signal in self.signals:
            signal.calculate_energy_numerical(self.dt)
            signal.calculate_energy_analytical()
    
    def print_system_info(self) -> None:
        """Вывод информации о системе"""
        q = self._calculate_total_signals()
        T = 1 / self.params.V_mod
        
        print(f"=== ИНФОРМАЦИЯ О СИСТЕМЕ QAM ===")
        print(f"Скорость модуляции V_mod: {self.params.V_mod} Бод")
        print(f"Информационная скорость V_inf: {self.params.V_inf} бит/сек")
        print(f"Период сигнала T: {T:.6f} сек")
        print(f"Количество сигналов q: {q}")
        print(f"Несущая частота f_0: {self.params.f_0} Гц")
        print(f"Количество отсчетов на период: {len(self.time_vector)}")
        print(f"Шаг дискретизации dt: {self.dt:.6f} сек")
        print()
    
    def print_analytical_expressions(self) -> None:
        """Вывод аналитических выражений для всех сигналов"""
        print("АНАЛИТИЧЕСКИЕ ВЫРАЖЕНИЯ:")
        for i, signal in enumerate(self.signals):
            formula = signal.get_formula_string()
            print(f"Сигнал {i}: s(t) = {formula}")
        print()
    
    def print_energies(self) -> None:
        """Вывод энергий всех сигналов"""
        print("ЭНЕРГИИ СИГНАЛОВ:")
        for i, signal in enumerate(self.signals):
            print(f"Сигнал {i} {signal.i_pair}: E = {signal.energy_numerical:.3f}")
        print()
    
    def create_dataframe(self) -> pd.DataFrame:
        """Создание DataFrame с полной информацией о сигналах"""
        table = []
        for i, signal in enumerate(self.signals):
            table.append([
                i,
                signal.i_pair[0],
                signal.i_pair[1],
                signal.s_i[0],
                signal.s_i[1],
                signal.energy_numerical,
                signal.energy_analytical,
                signal.get_formula_string()
            ])
        
        columns = ["i", "i1", "i2", "s_i1", "s_i2", "E_числ", "E_теор", "Формула"]
        return pd.DataFrame(table, columns=columns)
    
    def plot_signals(self) -> None:
        """Построение графиков сигналов"""
        if not any(signal.signal is not None for signal in self.signals):
            self.generate_all_signals()
        
        q = self._calculate_total_signals()
        n_signals = len(self.signals)
        n_cols = 4
        n_rows = math.ceil(n_signals / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        fig.suptitle(f'Сигналы КАМ (q={q}, V_mod={self.params.V_mod} Бод, '
                    f'V_inf={self.params.V_inf} бит/сек)', fontsize=16)
        
        # Если только одна строка, преобразуем axes в 2D массив
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, signal in enumerate(self.signals):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].plot(self.time_vector, signal.signal)
            axes[row, col].set_title(f'Сигнал {i}: {signal.i_pair}')
            axes[row, col].set_xlabel('Время (сек)')
            axes[row, col].set_ylabel('Амплитуда')
            axes[row, col].grid(True)
        
        # Скрыть пустые subplots
        for i in range(len(self.signals), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_energies_comparison(self) -> None:
        """Построение графика сравнения энергий"""
        if not any(signal.energy_numerical is not None for signal in self.signals):
            self.calculate_all_energies()
        
        numerical_energies = [signal.energy_numerical for signal in self.signals]
        analytical_energies = [signal.energy_analytical for signal in self.signals]
        
        plt.figure(figsize=(10, 5))
        plt.plot(numerical_energies, 'o-', label='E (дискретно)', markersize=4, linewidth=1)
        plt.plot(analytical_energies, '--', label='E (теоретически)', linewidth=2)
        plt.title(f"Сравнение энергий всех {len(self.signals)} сигналов")
        plt.xlabel("Номер сигнала i")
        plt.ylabel("Энергия")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


def main():
    """Основная функция для демонстрации работы системы"""
    # Создание системы с параметрами по умолчанию
    qam_system = QAMSystem()
    
    # Генерация всех сигналов и расчет энергий
    qam_system.generate_all_signals()
    qam_system.calculate_all_energies()
    
    # Вывод информации
    qam_system.print_system_info()
    qam_system.print_analytical_expressions()
    qam_system.print_energies()
    
    # Создание и вывод таблицы
    df = qam_system.create_dataframe()
    print("ПОЛНАЯ ТАБЛИЦА СИГНАЛОВ:")
    print(df.to_string(index=False))
    print()
    
    # Построение графиков
    qam_system.plot_signals()
    qam_system.plot_energies_comparison()


if __name__ == '__main__':
    main()
