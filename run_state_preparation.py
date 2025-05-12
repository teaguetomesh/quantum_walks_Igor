import os.path
import pickle
from functools import partial

import numpy as np
import pandas as pd
import qiskit
from pebble import ProcessPool
from tqdm import tqdm

from src.state_circuit_generator import StateCircuitGenerator, MHSTreeGeneratorExhaustive
from src.utilities.general import make_dict
from src.utilities.qiskit_utilities import remove_leading_cx_gates
from src.utilities.validation import execute_circuit, get_state_vector, get_fidelity


def prepare_state(target_state: dict[str, complex], circuit_generator: StateCircuitGenerator, basis_gates: list[str], optimization_level: int, check_fidelity: bool,
                  fidelity_tol: float = 1e-8) -> int:
    """ Prepares specified state using specified state generator and options. Returns CX count in the corresponding circuit. """
    circuit = circuit_generator.generate_circuit(target_state)
    circuit_transpiled = qiskit.transpile(circuit, **make_dict(basis_gates, optimization_level))
    circuit_transpiled = remove_leading_cx_gates(circuit_transpiled)
    cx_count = circuit_transpiled.count_ops().get('cx', 0)

    if check_fidelity:
        output_state_vector = execute_circuit(circuit_transpiled)
        target_state_vector = get_state_vector(target_state)
        fidelity = get_fidelity(output_state_vector[:len(target_state_vector)], target_state_vector)
        assert abs(1 - fidelity) < fidelity_tol, f'Failed to prepare the state. Fidelity: {fidelity}'

    return cx_count


def run_prepare_state():
    """ An entry point. Prepares the states from the target folder, counts CX gates in the resulting circuits and writes the results to a csv file. """
    # circuit_generator = QiskitDefaultGenerator()
    # circuit_generator = SingleEdgeGeneratorBackward(change_basis=True)
    circuit_generator = MHSTreeGeneratorExhaustive(change_basis=True, multiedge=True)
    # circuit_generator = MergingStatesGenerator()
    # circuit_generator = MultiEdgeSparseGenerator(permutation_circuit_generator=PermutationCircuitGeneratorSparse())

    num_qubits = np.array([11])
    num_amplitudes = num_qubits ** 2
    state_inds = list(range(0, 100))
    out_col_name = 'mhs_multi'
    # out_col_name = 'merging_states'
    data_folder_parent = 'data'
    num_workers = 12
    check_fidelity = True
    optimization_level = 3
    basis_gates = ['rx', 'ry', 'rz', 'h', 'cx']
    process_func = partial(prepare_state, **make_dict(circuit_generator, basis_gates, optimization_level, check_fidelity))

    # iterable = list(product(num_qubits, num_amplitudes))
    for item in zip(num_qubits, num_amplitudes):
        print(f'Current iterable: {item}')
        data_folder = f'{data_folder_parent}/qubits_{item[0]}/m_{item[1]}'
        states_file_path = os.path.join(data_folder, 'states.pkl')
        with open(states_file_path, 'rb') as f:
            state_list = pickle.load(f)
        state_list = np.array(state_list)[state_inds]

        results = []
        if num_workers == 1:
            for result in tqdm(map(process_func, state_list), total=len(state_list), smoothing=0, ascii=' █'):
                results.append(result)
        else:
            with tqdm(total=len(state_list), smoothing=0, ascii=' █') as progress_bar:
                with ProcessPool(num_workers) as pool:
                    future_iter = pool.map(process_func, state_list, timeout=2 * 3600).result()
                    while True:
                        try:
                            results.append(next(future_iter))
                        except StopIteration:
                            break
                        except TimeoutError:
                            results.append(0)
                        except Exception:
                            results.append(-1)
                        progress_bar.update()

        cx_counts_file_path = os.path.join(data_folder, 'cx_counts.csv')
        df = pd.read_csv(cx_counts_file_path) if os.path.isfile(cx_counts_file_path) else pd.DataFrame()
        df.loc[state_inds, out_col_name] = results
        df.to_csv(cx_counts_file_path, index=False)
        print(f'Avg CX: {np.mean(df[out_col_name])}\n')


if __name__ == '__main__':
    # np.random.seed(0)
    run_prepare_state()
