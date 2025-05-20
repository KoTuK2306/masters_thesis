import sys
import io
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.panel import Panel

from utils.console_and_print import console_and_print
from utils.initialize_environment import initialize_environment
from utils.read_hamiltonian_data import read_hamiltonian_data
from utils.print_hamiltonian import print_hamiltonian
from utils.print_pauli_table import print_pauli_table
from utils.print_composition_table import print_composition_table
from utils.calculate_temp_steps import calculate_temp_steps
from vqa_utils.pauli_compose import pauli_compose
from vqa_utils.generate_shifted_theta import generate_shifted_theta
from vqa_utils.simulated_annealing import simulated_annealing
from vqa_utils.calculate_ansatz import calculate_ansatz
from vqa_utils.compute_uhu import compute_uhu
from vqa_utils.calculate_expectation import calculate_expectation

from constants.file_paths import HAMILTONIAN_FILE_PATH

# Установка корректной кодировки стандартных потоков для поддержки Unicode
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

def main() -> None:
    """
    Основная функция программы. Реализует следующий цикл:
      1. Чтение гамильтониана из файла.
      2. Вывод операторов Паули и их композиции.
      3. Оптимизация анзаца методом отжига для последовательных подмножеств операторов.
      4. Поиск минимальной энергии и вывод оптимального анзаца.

    Программа построена как демонстрация вариационного квантового алгоритма
    с классическим оптимизатором (simulated annealing).
    """
    console = initialize_environment()

    # Проверка наличия файла с гамильтонианом
    if not HAMILTONIAN_FILE_PATH.exists():
        msg = (
            f"Файл [bold]{HAMILTONIAN_FILE_PATH}[/] не найден!\n"
            "Убедитесь, что рядом с EXE есть папка [bold]params[/] с файлом [bold]hamiltonian_operators.txt[/]."
        )
        console_and_print(console, Panel(msg, border_style="red"))
        return

    try:
        pauli_operators, pauli_strings = read_hamiltonian_data(HAMILTONIAN_FILE_PATH)
        print_hamiltonian(console, pauli_operators)
        print_pauli_table(console, pauli_operators)
        print_composition_table(console, pauli_compose, pauli_strings)
    except FileNotFoundError:
        console_and_print(console, Panel(f"[red]Файл {HAMILTONIAN_FILE_PATH} не найден[/red]", border_style="red"))
        return

    if len(pauli_operators) < 2:
        console_and_print(console, Panel("[red]Требуется минимум 2 оператора Паули[/red]", border_style="red"))
        return

    # Тестируемый анзац
    pauli_operators = [
        (1.0, [0, 0, 2, 1]),  # σ₀₀₂₁
        (1.0, [1, 2, 0, 0]),  # σ₁₂₀₀
        (1.0, [1, 1, 1, 2]),  # σ₁₁₁₂
        (1.0, [1, 2, 1, 1]),  # σ₁₂₁₁
    ]

    # Чтение операторов гамильтониана из файла 
    hamiltonian_operators, _ = read_hamiltonian_data(HAMILTONIAN_FILE_PATH)

    # Получение начального состояния θ
    initial_theta = generate_shifted_theta(pauli_operators)

    # Вывод прочтённого гамильтониана
    print_hamiltonian(console, hamiltonian_operators)
    
    # Вывод таблицы операторов гамильтониана (коэффициент, строка Паули)
    print_pauli_table(console, pauli_operators)

    # Вывод композиций операторов Паули
    print_composition_table(console, pauli_compose, [op for _, op in pauli_operators])

    # Параметры отжига
    SA_PARAMS = {
        "initial_temp": 100.0,
        "cooling_rate": 0.95,
        "min_temp": 1e-3,
        "num_iterations_per_temp": 100,
        "step_size": 0.1,
    }

    # Оценка общего количества шагов для прогресс-бара
    thermalization_steps = int(SA_PARAMS["num_iterations_per_temp"] * 0.2)
    temp_steps = calculate_temp_steps(
        SA_PARAMS["initial_temp"], 
        SA_PARAMS["cooling_rate"], 
        SA_PARAMS["min_temp"]
    )
    total_steps = temp_steps * (thermalization_steps + SA_PARAMS["num_iterations_per_temp"])

    # Запуск прогресс-бара с симпатичным оформлением
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("[cyan]Отжиг...", total=total_steps)

        optimized_theta, _ = simulated_annealing(
            initial_theta=initial_theta,
            pauli_operators=pauli_operators,
            hamiltonian_operators=hamiltonian_operators,
            progress=progress,
            task=task,
            **SA_PARAMS,
        )

    ansatz_dict, ansatz_symbolic, ansatz_numeric = calculate_ansatz(optimized_theta, pauli_operators)
    uhu_dict = compute_uhu(ansatz_dict, hamiltonian_operators)
    energy = calculate_expectation(uhu_dict)

    console_and_print(console, Panel(ansatz_symbolic, title="[bold]Символьное представление анзаца[/]", border_style="green"))

    console_and_print(console,Panel(ansatz_numeric, title="[bold]Численное представление анзаца[/]", border_style="purple"))

    console_and_print(console, Panel(f"{energy:.6f}", title="[bold]Энергия (<0|U†HU|0> для состояния |0...0>)[/]", border_style="green"))

    input("Нажмите Enter для выхода...")

if __name__ == "__main__":
    main()
