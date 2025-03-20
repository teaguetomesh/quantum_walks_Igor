from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from networkx.classes import DiGraph
from numpy import ndarray
from pysat.examples.hitman import Hitman
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from src.utilities.general import array_to_str
from src.utilities.quantum import get_average_neighbors, get_different_inds, find_min_control_set_2, get_cx_cost_cx


class PermutationCircuitGenerator(ABC):
    """ Class that generates permutation circuits. """

    @abstractmethod
    def get_permutation_circuit(self, permutation: dict[str, str]) -> QuantumCircuit:
        """ Generates a permutation circuit for a permutation given as an old basis -> new basis mapping. """
        pass

    def build_cx_mcx_pattern(self, quantum_circuit: QuantumCircuit, cx_control_ind: int, cx_target_inds: ndarray, mcx_control_inds: ndarray, mcx_control_vals: ndarray):
        """ Builds the pattern of a specified MCX gate conjugated with specified set of CX gates. 0th qubit is used as ancilla for MCX. """
        shift = 1
        conjugation_qc = QuantumCircuit(quantum_circuit.num_qubits)
        for ind in cx_target_inds:
            conjugation_qc.cx(cx_control_ind + shift, ind + shift)
        quantum_circuit.compose(conjugation_qc, inplace=True)
        mcx_control_inds = list(mcx_control_inds + shift)
        mcx_control_vals = array_to_str(mcx_control_vals)
        mcx_target_ind = cx_control_ind + shift
        quantum_circuit.mcx(mcx_control_inds, mcx_target_ind, 0, 'recursion', mcx_control_vals[::-1])
        quantum_circuit.compose(conjugation_qc.reverse_ops(), inplace=True)


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


class PermutationCircuitGeneratorSparseNaive(PermutationCircuitGenerator):
    """ Generates permutation circuit made up of pairwise swaps. """

    def get_permutation_circuit(self, permutation: dict[str, str]) -> QuantumCircuit:
        existing_states = np.array([list(map(int, key)) for key in permutation])
        target_states = np.array([list(map(int, val)) for val in permutation.values()])
        qc = QuantumCircuit(existing_states.shape[1] + 1)
        for i in range(existing_states.shape[0]):
            if np.all(existing_states[i, :] == target_states[i, :]):
                continue
            diff_inds = np.where(existing_states[i, :] != target_states[i, :])[0]
            control_inds, control_vals, interaction_ind = find_min_control_set_2(existing_states, i, diff_inds)
            cx_target_inds = diff_inds[diff_inds != interaction_ind]
            self.build_cx_mcx_pattern(qc, interaction_ind, cx_target_inds, control_inds, control_vals)
            existing_states[i, :] = target_states[i, :]
        return qc.reverse_bits()


class PermutationCircuitGeneratorSparse(PermutationCircuitGenerator):
    """ Generates permutation circuit that takes into account group permutations. """

    def get_permutation_circuit(self, permutation: dict[str, str]) -> QuantumCircuit:
        @dataclass
        class SwapGate:
            control_inds: ndarray
            control_vals: ndarray
            target_ind: int
            conjugated_target_inds: ndarray

        @dataclass
        class Swap:
            selected_cols: list[int]
            row_inds: list[int]
            score: float
            gate: SwapGate

        def find_best_swap(all_bases_old: ndarray, difference: ndarray) -> Swap:
            @dataclass
            class Group:
                coords: (int, ...)
                row_inds: list[int]
                errors_by_col: ndarray
                sorted_col_inds: ndarray
                movers_by_col: ndarray
                average_degree: float

            def form_groups(all_bases_old: ndarray, difference: ndarray, selected_cols: list[int]) -> dict[(int, ...), Group]:
                groups = {}
                for key, val in pd.DataFrame(all_bases_old).groupby(selected_cols):
                    row_inds = val.index.tolist()
                    errors_by_col = np.sum(difference[np.ix_(row_inds, selected_cols)], 0)
                    sorted_col_inds = np.argsort(errors_by_col)[::-1]
                    movers_by_col = np.sum(difference[np.ix_(row_inds, selected_cols)] == 1, 0)
                    mover_inds = np.zeros(all_bases_old.shape[0], dtype=bool)
                    mover_inds[row_inds] = True
                    mover_inds &= difference[:, selected_cols[sorted_col_inds[0]]] == 1
                    average_degree = get_average_neighbors(all_bases_old[mover_inds, :])
                    groups[key] = Group(key, row_inds, errors_by_col, sorted_col_inds, movers_by_col, average_degree)
                return groups

            def build_swap_graph(groups: dict[(int, ...), Group]) -> DiGraph:
                def update_group_location(group: Group, all_coords: list[(int, ...)], graph: DiGraph):
                    def find_best_swap_gate(all_coords: list[(int, ...)], distinguish_ind: int, candidate_target_inds: ndarray) -> SwapGate:
                        control_inds, control_vals, interaction_ind = find_min_control_set_2(all_coords, distinguish_ind, candidate_target_inds)
                        return SwapGate(control_inds, control_vals, interaction_ind, candidate_target_inds)

                    if group.coords not in graph:
                        graph.add_node(group.coords, group=group)
                    for ind in range(len(group.sorted_col_inds)):
                        if ind > 0 >= group.errors_by_col[group.sorted_col_inds[ind]]:
                            break
                        subselected_cols = group.sorted_col_inds[:ind + 1]
                        destination_coords = tuple(coord ^ 1 if coord_ind in subselected_cols else coord for coord_ind, coord in enumerate(group.coords))
                        destination_group = groups.get(destination_coords)
                        if destination_coords not in graph:
                            graph.add_node(destination_coords, group=destination_group)
                        weight_forward = np.sum(group.errors_by_col[subselected_cols])
                        weight_backward = np.sum(destination_group.errors_by_col[subselected_cols]) if destination_group is not None else 0
                        weight_full = weight_forward + weight_backward
                        group_ind = all_coords.index(group.coords)
                        swap_gate = find_best_swap_gate(all_coords, group_ind, subselected_cols)
                        num_cx_gates = get_cx_cost_cx(len(swap_gate.control_inds))
                        if num_cx_gates == 0:
                            num_cx_gates = 0.001
                        num_cx_gates += 2 * (len(subselected_cols) - 1)
                        weight_full_scaled = weight_full / num_cx_gates
                        num_movers_forward = np.sum(group.movers_by_col[subselected_cols])
                        graph.add_edge(group.coords, destination_coords, weight=(weight_full_scaled, num_movers_forward, group.average_degree), gate=swap_gate)

                graph = DiGraph()
                all_coords = list(groups.keys())
                for group in groups.values():
                    update_group_location(group, all_coords, graph)
                return graph

            best_swap_overall = None
            selected_cols_final = []
            for num_cols in range(1, difference.shape[1] + 1):
                best_swap_layer = None
                for candidate_col_ind in range(difference.shape[1]):
                    if candidate_col_ind in selected_cols_final:
                        continue
                    selected_cols_trial = sorted(selected_cols_final + [candidate_col_ind])
                    groups = form_groups(all_bases_old, difference, selected_cols_trial)
                    swap_graph = build_swap_graph(groups)
                    best_edge = max(swap_graph.edges(data=True), key=lambda x: x[2]['weight'])
                    if best_swap_layer is None or best_swap_layer.score < best_edge[2]['weight']:
                        row_inds = groups[best_edge[0]].row_inds
                        if groups.get(best_edge[1]) is not None:
                            row_inds += groups[best_edge[1]].row_inds
                        best_swap_layer = Swap(selected_cols_trial[:], row_inds, best_edge[2]['weight'], best_edge[2]['gate'])
                selected_cols_final = best_swap_layer.selected_cols[:]
                if best_swap_overall is None or best_swap_overall.score < best_swap_layer.score:
                    best_swap_overall = best_swap_layer
            return best_swap_overall

        def implement_swap(swap: Swap, num_ancillas: int, qc: QuantumCircuit, all_bases_old: ndarray, difference: ndarray):
            selected_cols = np.array(swap.selected_cols)
            if len(swap.gate.control_inds) == 0:
                qc.x(selected_cols[swap.gate.target_ind] + num_ancillas)
            else:
                cx_control_ind = selected_cols[swap.gate.target_ind]
                cx_target_inds = selected_cols[swap.gate.conjugated_target_inds[swap.gate.conjugated_target_inds != cx_control_ind]]
                mcx_control_inds = selected_cols[swap.gate.control_inds]
                mcx_control_vals = swap.gate.control_vals
                self.build_cx_mcx_pattern(qc, cx_control_ind, cx_target_inds, mcx_control_inds, mcx_control_vals)
            all_bases_old[np.ix_(swap.row_inds, selected_cols[swap.gate.conjugated_target_inds])] ^= 1
            difference[np.ix_(swap.row_inds, selected_cols[swap.gate.conjugated_target_inds])] *= -1

        all_bases_old = np.array([[int(val) for val in basis] for basis in permutation])
        all_bases_new = np.array([[int(val) for val in basis] for basis in permutation.values()])
        difference = (all_bases_old ^ all_bases_new) * 2 - 1
        num_ancillas = 1
        qc = QuantumCircuit(all_bases_old.shape[1] + num_ancillas)
        while np.any(difference == 1):
            swap = find_best_swap(all_bases_old, difference)
            implement_swap(swap, num_ancillas, qc, all_bases_old, difference)
        return qc.reverse_bits()
