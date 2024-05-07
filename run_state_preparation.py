import os.path
import pickle
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from qiskit import transpile, QuantumCircuit
from tqdm import tqdm

from src.quantum_walks import PathFinder, PathFinderLinear, PathFinderSHP, PathFinderMST
from src.validation import execute_circuit, get_state_vector, get_fidelity
from src.walks_gates_conversion import PathConverter


def prepare_state(target_state: dict[str, complex], method: str, path_finder: PathFinder, basis_gates: list[str], optimization_level: int, check_fidelity: bool = True,
                  fidelity_tol: float = 1e-10, reduce_controls: bool = True) -> int:
    if method == "qiskit":
        target_state_vector = get_state_vector(target_state)
        num_qubits = len(next(iter(target_state.keys())))
        circuit = QuantumCircuit(num_qubits)
        circuit.prepare_state(target_state_vector)
    elif method == "walks":
        path = path_finder.get_path(target_state)
        circuit = PathConverter.convert_path_to_circuit(path, reduce_controls)
    else:
        raise Exception("Unknown method")
    circuit_transpiled = transpile(circuit, basis_gates=basis_gates, optimization_level=optimization_level)
    cx_count = circuit_transpiled.count_ops().get("cx", 0)

    if check_fidelity:
        output_state_vector = execute_circuit(circuit_transpiled)
        target_state_vector = get_state_vector(target_state)
        fidelity = get_fidelity(output_state_vector, target_state_vector)
        assert abs(1 - fidelity) < fidelity_tol, "Failed to prepare the state"

    return cx_count


def run_prepare_state():
    method = "walks"
    # path_finder = PathFinderLinear()
    path_finder = PathFinderSHP()
    # path_finder = PathFinderMST()
    basis_gates = ["rx", "ry", "rz", "h", "cx"]
    optimization_level = 3
    check_fidelity = True
    reduce_controls = True
    num_qubits_all = [3]
    # num_qubits_all = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    out_col_name = "shp_reduced"
    num_workers = 20
    process_func = partial(prepare_state, method=method, path_finder=path_finder, basis_gates=basis_gates, optimization_level=optimization_level, check_fidelity=check_fidelity,
                           reduce_controls=reduce_controls)

    for num_qubits in num_qubits_all:
        print(f"Num qubits: {num_qubits}")
        results_path = f"data/qubits_{num_qubits}/cx_counts.csv"
        results_folder = os.path.split(results_path)[0]
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        if os.path.isfile(results_path):
            df = pd.read_csv(results_path)
        else:
            df = pd.DataFrame()

        with open(f"data/qubits_{num_qubits}/states.pkl", "rb") as f:
            state_list = pickle.load(f)
        results = []
        if num_workers == 1:
            for result in tqdm(map(process_func, state_list), total=len(state_list), smoothing=0, ascii=' █'):
                results.append(result)
        else:
            with Pool(num_workers) as pool:
                for result in tqdm(pool.imap(process_func, state_list), total=len(state_list), smoothing=0, ascii=' █'):
                    results.append(result)

        df[out_col_name] = results
        df.to_csv(results_path, index=False)
        print(f"Avg CX: {np.mean(df[out_col_name])}\n")


if __name__ == "__main__":
    run_prepare_state()
