import os.path
import pickle
import random
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from qiskit import transpile
from qiskit.quantum_info import random_statevector, Statevector
from tqdm import tqdm

from src.permutation_circuit_generator import PermutationCircuitGeneratorQiskit, PermutationCircuitGeneratorSparse
from src.permutation_generator import PermutationGeneratorDense
from src.qiskit_utilities import remove_leading_cx_gates
from src.quantum_walks import PathFinderLinear, PathFinderSHP, PathFinderMST, PathFinderRandom, PathFinderGrayCode
from src.utilities import make_dict
from src.validation import execute_circuit, get_state_vector, get_fidelity
from src.state_preparation import CircuitGenerator, CircuitGeneratorPath, CircuitGeneratorQiskitDefault, CircuitGeneratorQiskitDense


def prepare_state(target_state: dict[str, complex], circuit_generator: CircuitGenerator, basis_gates: list[str], optimization_level: int, check_fidelity: bool,
                  fidelity_tol: float = 1e-8) -> int:
    circuit = circuit_generator.generate_circuit(target_state)
    circuit_transpiled = transpile(circuit, **make_dict(basis_gates, optimization_level))
    circuit_transpiled = remove_leading_cx_gates(circuit_transpiled)
    cx_count = circuit_transpiled.count_ops().get("cx", 0)

    if check_fidelity:
        output_state_vector = execute_circuit(circuit_transpiled)
        target_state_vector = get_state_vector(target_state)
        fidelity = get_fidelity(output_state_vector, target_state_vector)
        assert abs(1 - fidelity) < fidelity_tol, f"Failed to prepare the state. Fidelity: {fidelity}"

    return cx_count


def generate_states():
    num_qubits = np.array(list(range(4, 12)))
    num_amplitudes = num_qubits ** 2
    num_states = 1000

    for n, m in zip(num_qubits, num_amplitudes):
        out_path = f"data/qubits_{n}/m_{m}/states.pkl"
        all_inds = list(range(2 ** n))
        states = []
        for i in range(num_states):
            state_vector = random_statevector(len(all_inds)).data
            zero_inds = random.sample(all_inds, len(all_inds) - m)
            state_vector[zero_inds] = 0
            state_vector /= sum(abs(amplitude) ** 2 for amplitude in state_vector) ** 0.5
            state_dict = Statevector(state_vector).to_dict()
            states.append(state_dict)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(states, f)


def merge_state_files():
    num_qubits = np.array(list(range(3, 12)))
    num_amplitudes = 2 ** (num_qubits - 1)
    merged = {}
    for n, m in zip(num_qubits, num_amplitudes):
        file_path = f"data/qubits_{n}/m_{m}/states.pkl"
        with open(file_path, "rb") as f:
            state_list = pickle.load(f)
        merged[f"qubits_{n}_amplitudes_{m}"] = state_list
    with open("states_merged.pkl", "wb") as f:
        pickle.dump(merged, f)


def run_prepare_state():
    # path_finder = PathFinderRandom()
    path_finder = PathFinderLinear()
    # path_finder = PathFinderGrayCode()
    # path_finder = PathFinderSHP()
    # path_finder = PathFinderMST()

    # circuit_generator = CircuitGeneratorQiskitDefault()
    # circuit_generator = CircuitGeneratorPath(path_finder=path_finder, reduce_controls=True, remove_leading_cx=True, add_barriers=False)
    circuit_generator = CircuitGeneratorQiskitDense(dense_permutation_generator=PermutationGeneratorDense(), permutation_circuit_generator=PermutationCircuitGeneratorSparse())

    num_qubits_all = np.array(list(range(6, 12)))
    num_amplitudes_all = num_qubits_all
    out_col_name = "qiskit_dense"
    num_workers = 10
    check_fidelity = True
    optimization_level = 3
    basis_gates = ["rx", "ry", "rz", "h", "cx"]
    process_func = partial(prepare_state, **make_dict(circuit_generator, basis_gates, optimization_level, check_fidelity))

    for num_qubits, num_amplitudes in zip(num_qubits_all, num_amplitudes_all):
        print(f"Num qubits: {num_qubits}; num amplitudes: {num_amplitudes}")
        data_folder = f"data/qubits_{num_qubits}/m_{num_amplitudes}"
        states_file_path = os.path.join(data_folder, "states.pkl")
        with open(states_file_path, "rb") as f:
            state_list = pickle.load(f)

        results = []
        if num_workers == 1:
            for result in tqdm(map(process_func, state_list), total=len(state_list), smoothing=0, ascii=' █'):
                results.append(result)
        else:
            with Pool(num_workers) as pool:
                for result in tqdm(pool.imap(process_func, state_list), total=len(state_list), smoothing=0, ascii=' █'):
                    results.append(result)

        cx_counts_file_path = os.path.join(data_folder, "cx_counts.csv")
        df = pd.read_csv(cx_counts_file_path) if os.path.isfile(cx_counts_file_path) else pd.DataFrame()
        df[out_col_name] = results
        df.to_csv(cx_counts_file_path, index=False)
        print(f"Avg CX: {np.mean(df[out_col_name])}\n")


if __name__ == "__main__":
    # generate_states()
    # merge_state_files()
    run_prepare_state()
