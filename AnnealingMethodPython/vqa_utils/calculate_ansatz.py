import numpy as np
from typing import Tuple, List, Dict
from utils.format_ansatz import format_ansatz
from .pauli_compose import pauli_compose

def calculate_ansatz(
    theta: np.ndarray, pauli_operators: List[Tuple[complex, List[int]]]
) -> Tuple[Dict[Tuple[int, ...], complex], str, str]:
    """
    Вычисляет вариационный анзац в виде произведения экспонент операторов Паули.

    Args:
        theta (np.ndarray): Вектор параметров (обычно одного размера с числом операторов).
        pauli_operators (List[Tuple[complex, List[int]]]): Операторы Паули с коэффициентами.

    Returns:
        Tuple[
            Dict[Tuple[int, ...], complex],   # Разложение анзаца по Паули-операторам
            str,                             # Символьное представление (произведение экспонент)
            str                              # Численное разложение
        ]

    Алгоритм:
        U(θ) = prod_j exp(i * θ_j * |c_j| * σ_j)
        Реализуется по принципу покомпонентного разложения через формулу Эйлера:
            exp(i·α·σ) = cos(α)·I + i·sin(α)·σ
        С каждым новым оператором результат рекурсивно обновляется через pauli_compose.
    """
    operator_length = len(pauli_operators[0][1])
    result: Dict[Tuple[int, ...], complex] = {tuple([0] * operator_length): 1.0}

    for t, (coeff, op) in zip(theta, pauli_operators):
        angle = t * coeff
        cos_t = np.cos(angle)
        sin_t = np.sin(angle)
        new_result: Dict[Tuple[int, ...], complex] = {}
        op_tuple = tuple(op)
        for existing_op, existing_coeff in result.items():
            # cos(angle) * I (сохраняем индекс базисного оператора)
            new_result[existing_op] = new_result.get(existing_op, 0) + existing_coeff * cos_t
            # i*sin(angle)*σ (композиция Паули)
            compose_coeff, compose_op = pauli_compose(existing_op, op_tuple)
            final_coeff = existing_coeff * 1j * sin_t * compose_coeff
            new_result[compose_op] = new_result.get(compose_op, 0) + final_coeff
        result = new_result

    symbolic_str, numeric_str = format_ansatz(pauli_operators, result)
    return result, symbolic_str, numeric_str
