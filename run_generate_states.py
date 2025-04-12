import os
import pickle
import random

import numpy as np
import qiskit.quantum_info as qqi
from qiskit.quantum_info import Statevector


def generate_states():
    """ Generates sets of random sparse states. Writes the results to specified folder. """
    num_qubits = np.array(list(range(8, 9)))
    num_amplitudes = num_qubits ** 2
    num_states = 1000

    for n, m in zip(num_qubits, num_amplitudes):
        out_path = f"data/qubits_{n}/m_{m}/states.pkl"
        all_inds = list(range(2 ** n))
        states = []
        for i in range(num_states):
            state_vector = qqi.random_statevector(len(all_inds)).data
            zero_inds = random.sample(all_inds, len(all_inds) - m)
            state_vector[zero_inds] = 0
            state_vector /= sum(abs(amplitude) ** 2 for amplitude in state_vector) ** 0.5
            state_dict = Statevector(state_vector).to_dict()
            states.append(state_dict)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(states, f)


if __name__ == "__main__":
    generate_states()
