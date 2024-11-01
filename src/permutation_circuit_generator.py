from abc import ABC, abstractmethod
from collections import deque

import numpy as np
from numpy import ndarray
from pysat.examples.hitman import Hitman
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


class PermutationCircuitGeneratorSparse(PermutationCircuitGenerator):
    """ Uses manual CX composition that works well for sparse state permutations. """

    @staticmethod
    def find_min_control_set(other_bases: ndarray, basis: ndarray, exclude_ind: int) -> list[int]:
        """  Finds minimal set of controls necessary to distinguish basis from other_bases that does not include exclude_ind.
        other_bases is a 2D array of binary numbers where each row corresponds to a basis. """
        assert other_bases.shape[1] == len(basis), 'Dimensions mismatch'
        get_diff_inds = lambda another_basis: [ind for ind in range(len(basis)) if ind != exclude_ind and basis[ind] != another_basis[ind]]
        difference_inds = [get_diff_inds(another_basis) for another_basis in other_bases]
        hitman = Hitman()
        for ind_set in difference_inds:
            hitman.hit(ind_set)
        return hitman.get()

    def get_permutation_circuit(self, permutation: dict[str, str]) -> QuantumCircuit:
        all_bases_old = np.array([[int(val) for val in basis] for basis in permutation])
        all_bases_new = np.array([[int(val) for val in basis] for basis in permutation.values()])
        qc = QuantumCircuit(all_bases_old.shape[1])
        for j in range(all_bases_old.shape[1]):
            if all_bases_old[0, j] != all_bases_new[0, j]:
                qc.x(j)
                all_bases_old[:, j] ^= 1
        for i in range(1, all_bases_old.shape[0]):
            change_inds = deque(np.where(all_bases_old[i, :] != all_bases_new[i, :])[0])
            while len(change_inds) > 0:
                next_ind = change_inds.popleft()
                control_inds = self.find_min_control_set(all_bases_old[:i, :], all_bases_old[i, :], next_ind)
                if control_inds is None:
                    if len(change_inds) == 0:
                        raise Exception('Stuck in permutation')
                    change_inds.append(next_ind)
                    continue
                ctrl_state = ''.join([str(num) for num in all_bases_old[i, control_inds]])
                qc.mcx(control_inds, next_ind, ctrl_state=ctrl_state[::-1])
                all_bases_old[np.all(all_bases_old[:, control_inds] == all_bases_old[i, control_inds], 1), next_ind] ^= 1
        return qc.reverse_bits()
