""" Procedures for debugging and validation of results. """
import numpy as np
from numpy import ndarray
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def execute_circuit(circuit: QuantumCircuit) -> ndarray:
    """
    Executes transpiled circuit and returns the statevector output.
    :param circuit: Transpiled qiskit circuit to execute.
    :return: State vector at the end of the circuit.
    """
    sim = AerSimulator()
    circuit.save_statevector(label="end")
    result = sim.run(circuit).result()
    state = result.data(0)["end"].data
    return state


def get_state_vector(target_state: dict[str, complex]) -> ndarray:
    """
    Converts a state described by a dictionary into the state vector representation.
    :param target_state: The state to convert.
    :return: 2^n 1D array corresponding to the target state vector in big endian notation.
    """
    num_qubits = len(list(target_state.keys())[0])
    state_vector = np.zeros(2 ** num_qubits, dtype=complex)
    for basis, amplitude in target_state.items():
        basis_ind = int(basis[::-1], 2)
        state_vector[basis_ind] = target_state[basis]
    return state_vector


def get_fidelity(state1: ndarray, state2: ndarray) -> float:
    """
    Get the fidelity between the two states.
    :param state1: First state.
    :param state2: Second state.
    :return: Fidelity of the two states.
    """
    return abs(state1.conjugate().T @ state2) ** 2
