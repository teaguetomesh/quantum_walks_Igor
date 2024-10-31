from abc import ABC, abstractmethod

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


class PermutationCircuitGenerator(ABC):
    """ Class that generates permutation circuits. """

    @abstractmethod
    def get_permutation_circuit(self, permutation: dict[str, str]) -> QuantumCircuit:
        """ Generates a permutation circuit for a permutation given as an old basis -> new basis mapping. """
        pass


class PermutationCircuitGeneratorQiskit(PermutationCircuitGenerator):
    """ Composes explicit unitary and lets qiskit decompose it. """

    def get_permutation_circuit(self, permutation: dict[str, str]) -> QuantumCircuit:
        num_qubits = len(next(iter(permutation)))
        unitary = np.eye(2 ** num_qubits)
        for old_basis, new_basis in permutation.items():
            old_basis_int, new_basis_int = int(old_basis, 2), int(new_basis, 2)
            unitary[:, [old_basis_int, new_basis_int]] = unitary[:, [new_basis_int, old_basis_int]]

        qc = QuantumCircuit(num_qubits)
        qc.append(Operator(unitary), range(num_qubits))
        return qc
