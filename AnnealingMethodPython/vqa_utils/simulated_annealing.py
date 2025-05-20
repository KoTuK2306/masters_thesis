import numpy as np
from typing import Any, List, Tuple
from .generate_neighbor_theta import generate_neighbor_theta
from .calculate_ansatz import calculate_ansatz
from .compute_uhu import compute_uhu
from .calculate_expectation import calculate_expectation

def simulated_annealing(
    initial_theta: np.ndarray,
    pauli_operators: List[Any],
    hamiltonian_operators: List[Any],
    progress: Any,
    task: Any,
    initial_temp: float = 1000.0,
    cooling_rate: float = 0.99,
    min_temp: float = 1e-5,
    num_iterations_per_temp: int = 500,
    step_size: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """
    Алгоритм отжига (simulated annealing) для оптимизации параметров θ вариационного анзаца.

    Args:
        initial_theta (np.ndarray): Начальный вектор θ.
        pauli_operators (List[Tuple[complex, List[int]]]): Операторы Паули.
        hamiltonian_operators List[Any]: Операторы гамильтониана
        progress (Any): Индикатор прогресса (может быть None).
        task (Any): Задача для прогресс-бара.
        initial_temp (float): Начальная температура (чем выше — тем вероятнее принять ухудшающее решение).
        cooling_rate (float): Множитель охлаждения (0 < cooling_rate < 1).
        min_temp (float): Минимально допустимая температура.
        num_iterations_per_temp (int): Количество шагов на каждой температуре.
        step_size (float): Стандартное отклонение для шума θ.

    Returns:
        Tuple[np.ndarray, float]: Оптимальный найденный θ и соответствующая энергия.

    Принцип работы:
        - На каждом шаге генерируется новое состояние θ (случайным образом).
        - Если энергия уменьшилась — принимаем новое состояние.
        - Если энергия увеличилась — принимаем с вероятностью exp(-ΔE/T).
        - Температура постепенно понижается (охлаждение).

    Важно:
        - В термализации используется локальный случайный шаг (как и в основном цикле).
        - Это обеспечивает более "физичное" поведение отжига.
    """
    current_theta = initial_theta.copy()
    best_theta = current_theta.copy()
    best_energy = float("inf")
    rng = np.random.default_rng()
    temp = initial_temp
    thermalization_steps = int(num_iterations_per_temp * 0.2)

    while temp > min_temp:
        # Этап термализации: локальные случайные шаги для прогрева цепочки
        for _ in range(thermalization_steps):
            neighbor_theta = generate_neighbor_theta(current_theta, step_size)
            ansatz_dict, _, _ = calculate_ansatz(neighbor_theta, pauli_operators)
            uhu_dict = compute_uhu(ansatz_dict, hamiltonian_operators)
            current_energy = calculate_expectation(uhu_dict)
            if current_energy < best_energy:
                best_theta = neighbor_theta.copy()
                best_energy = current_energy
            current_theta = neighbor_theta.copy()
            if progress is not None:
                progress.update(task, advance=1)

        # Основной цикл отжига с возможностью принимать ухудшения
        for _ in range(num_iterations_per_temp):
            perturbation = rng.normal(0, step_size * (temp / initial_temp), current_theta.shape)
            neighbor_theta = (current_theta + perturbation) % (2 * np.pi)
            ansatz_dict, _, _ = calculate_ansatz(neighbor_theta, pauli_operators)
            uhu_dict = compute_uhu(ansatz_dict, hamiltonian_operators)
            current_energy = calculate_expectation(uhu_dict)
            energy_diff = current_energy - best_energy
            # Классическое правило Метрополиса
            if energy_diff < 0 or rng.random() < np.exp(-energy_diff / temp):
                current_theta = neighbor_theta.copy()
                if current_energy < best_energy:
                    best_theta = current_theta.copy()
                    best_energy = current_energy
            if progress is not None:
                progress.update(task, advance=1)
        temp *= cooling_rate

    return best_theta, best_energy
