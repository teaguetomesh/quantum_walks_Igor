from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from networkx.classes import DiGraph
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
    """ Uses manual CX composition for sparse state permutations. """

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

            def form_groups(all_bases_old: ndarray, difference: ndarray, selected_cols: list[int]) -> dict[(int, ...), Group]:
                groups = {}
                for key, val in pd.DataFrame(all_bases_old).groupby(selected_cols):
                    row_inds = val.index.tolist()
                    errors_by_col = np.sum(difference[np.ix_(row_inds, selected_cols)], 0)
                    sorted_col_inds = np.argsort(errors_by_col)[::-1]
                    groups[key] = Group(key, row_inds, errors_by_col, sorted_col_inds)
                return groups

            def build_swap_graph(groups: dict[(int, ...), Group]) -> DiGraph:
                def update_group_location(group: Group, all_coords: list[(int, ...)], graph: DiGraph):
                    def find_best_swap_gate(all_coords: list[(int, ...)], distinguish_ind: int, candidate_target_inds: ndarray) -> SwapGate:
                        def get_diff_inds(coords1: (int, ...), coords2: (int, ...), exclude_ind: int):
                            diff_inds = [ind for ind in range(len(coords1)) if ind != exclude_ind and coords1[ind] != coords2[ind]]
                            return diff_inds

                        best_swap_gate = None
                        for target_ind in candidate_target_inds:
                            transformed_coords = np.array(all_coords)
                            secondary_target_inds = [ind for ind in candidate_target_inds if ind != target_ind]
                            transformed_coords[np.ix_(transformed_coords[:, target_ind] == 1, secondary_target_inds)] ^= 1
                            hitman = Hitman()
                            for coords in transformed_coords:
                                diff_inds = get_diff_inds(coords, transformed_coords[distinguish_ind], target_ind)
                                if len(diff_inds) == 0:
                                    continue
                                hitman.hit(diff_inds)
                            control_inds = hitman.get()
                            if best_swap_gate is None or len(best_swap_gate.control_inds) > len(control_inds):
                                best_swap_gate = SwapGate(np.array(control_inds), transformed_coords[distinguish_ind, control_inds], target_ind, candidate_target_inds)
                        return best_swap_gate

                    cx_by_num_controls = [0, 1, 6, 14, 36, 56, 80, 104]
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
                        if len(swap_gate.control_inds) < len(cx_by_num_controls):
                            num_cx_gates = cx_by_num_controls[len(swap_gate.control_inds)]
                        else:
                            num_cx_gates = cx_by_num_controls[-1] + (len(swap_gate.control_inds) - len(cx_by_num_controls) - 1) * 16
                        num_cx_gates += 2 * (len(subselected_cols) - 1)
                        weight_full_scaled = weight_full / num_cx_gates if num_cx_gates != 0 else np.inf * weight_full if weight_full != 0 else 0
                        graph.add_edge(group.coords, destination_coords, weight=weight_full_scaled, gate=swap_gate)

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
                conjugation_qc = QuantumCircuit(qc.num_qubits)
                for ind in swap.gate.conjugated_target_inds:
                    if ind == swap.gate.target_ind:
                        continue
                    conjugation_qc.cx(selected_cols[swap.gate.target_ind] + num_ancillas, selected_cols[ind] + num_ancillas)
                qc.compose(conjugation_qc, inplace=True)
                control_inds = list(selected_cols[swap.gate.control_inds] + num_ancillas)
                control_vals = ''.join([str(val) for val in swap.gate.control_vals])
                target_ind = selected_cols[swap.gate.target_ind] + num_ancillas
                qc.mcx(control_inds, target_ind, 0, 'recursion', control_vals[::-1])
                qc.compose(conjugation_qc.reverse_ops(), inplace=True)
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
