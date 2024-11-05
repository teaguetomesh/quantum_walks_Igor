from abc import ABC, abstractmethod
from collections import deque, defaultdict
from typing import Sequence

import numpy as np
import numpy.ma as ma
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
    def find_min_control_set(other_bases: ndarray, basis: ndarray, exclude_inds: Sequence[int]) -> list[int]:
        """  Finds minimal set of controls necessary to distinguish basis from other_bases, excluding inds in exclude_inds.
        other_bases is a 2D array of binary numbers where each row corresponds to a basis. """
        assert other_bases.shape[1] == len(basis), 'Dimensions mismatch'
        difference_inds = [[ind for ind in range(len(another_basis)) if ind not in exclude_inds and another_basis[ind] != basis[ind]] for another_basis in other_bases]
        hitman = Hitman()
        for ind_set in difference_inds:
            hitman.hit(ind_set)
        return hitman.get()

    @staticmethod
    def apply_mcx(controls: Sequence[int], control_state: Sequence[int], target: int, qc: QuantumCircuit, all_bases: ndarray, differences: ndarray):
        if len(controls) == 0:
            affected_states = list(range(all_bases.shape[0]))
            qc.x(target)
        else:
            affected_states = np.where(np.all(all_bases[:, controls] == control_state, 1))[0]
            if len(affected_states) == 0:
                return
            ctrl_state = ''.join([str(num) for num in control_state])
            qc.mcx(controls, target, ctrl_state=ctrl_state[::-1])
        all_bases[affected_states, target] ^= 1
        differences[affected_states, target] ^= 1

    def get_permutation_circuit(self, permutation: dict[str, str]) -> QuantumCircuit:
        all_bases_old = np.array([[int(val) for val in basis] for basis in permutation])
        all_bases_new = np.array([[int(val) for val in basis] for basis in permutation.values()])
        difference = all_bases_old ^ all_bases_new
        qc = QuantumCircuit(all_bases_old.shape[1])
        for j in range(all_bases_old.shape[1]):
            if sum(difference[:, j]) > all_bases_old.shape[0] / 2:
                self.apply_mcx([], [], j, qc, all_bases_old, difference)

        tried_cols = np.array([False] * all_bases_old.shape[1])
        mode = 'cols'
        while True:
            if mode == 'cols':
                diff_sums = np.sum(difference, 0)
                if all(diff_sums == 0):
                    break
                target_ind = ma.masked_where(tried_cols, diff_sums).argmax()
                if diff_sums[target_ind] < 2 or all(tried_cols):
                    mode = 'rows'
                    continue
                tried_cols[target_ind] = True

                need_change_rows = difference[:, target_ind] == 1
                next_subset_rows = np.where(need_change_rows)[0]
                while True:
                    zero_count = np.sum(all_bases_old[next_subset_rows, :] == 0, 0)
                    one_count = len(next_subset_rows) - zero_count
                    largest_group_sizes = np.maximum(zero_count, one_count)
                    exclude_cols = largest_group_sizes < len(next_subset_rows)
                    exclude_cols[target_ind] = True
                    exclude_cols = np.where(exclude_cols)[0]
                    control_inds = self.find_min_control_set(all_bases_old[~need_change_rows, :], all_bases_old[next_subset_rows[0], :], exclude_cols)
                    if control_inds is not None and len(control_inds) > 0:
                        break

                    exclude_cols = largest_group_sizes == len(next_subset_rows)
                    exclude_cols[target_ind] = True
                    largest_nonmax_ind = ma.masked_where(exclude_cols, largest_group_sizes).argmax()
                    group_val = int(one_count[largest_nonmax_ind] >= zero_count[largest_nonmax_ind])
                    next_subset_rows = next_subset_rows[all_bases_old[next_subset_rows, largest_nonmax_ind] == group_val]

                    if len(next_subset_rows) == 1:
                        break

                if len(next_subset_rows) == 1:
                    continue

                self.apply_mcx(control_inds, all_bases_old[next_subset_rows[0], control_inds], target_ind, qc, all_bases_old, difference)
                tried_cols = np.array([False] * all_bases_old.shape[1])

            if mode == 'rows':
                diff_sums = np.sum(difference, 1)
                target_row = np.argmax(diff_sums)
                need_change_cols = np.where(difference[target_row, :] == 1)[0]
                target_col = need_change_cols[0]

                other_rows = np.where((np.array(range(all_bases_old.shape[0])) != target_row) & (~np.all(all_bases_old == all_bases_new[target_row, :], 1)))[0]
                for col in need_change_cols[1:]:
                    self.apply_mcx([target_col], [1], col, qc, all_bases_old, difference)
                control_inds = self.find_min_control_set(all_bases_old[other_rows, :], all_bases_old[target_row, :], [target_col])
                self.apply_mcx(control_inds, all_bases_old[target_row, control_inds], target_col, qc, all_bases_old, difference)
                for col in reversed(need_change_cols[1:]):
                    self.apply_mcx([target_col], [1], col, qc, all_bases_old, difference)
                mode = 'cols'
                tried_cols = np.array([False] * all_bases_old.shape[1])

        return qc.reverse_bits()
