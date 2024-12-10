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

        # def find_best_path(swap_graph: DiGraph) -> (int, list[tuple[int, ...]]):
        #     def dfs(current_path: list[tuple[int, ...]]) -> (int, list[tuple[int, ...]]):
        #         def calculate_path_weight(path: list[tuple[int, ...]]):
        #             def find_closest_empty(group: Group) -> tuple[int, ...]:
        #                 for i in range(1, 2 ** len(group.errors_by_col)):
        #                     i_bin = format(i, f'0{len(group.errors_by_col)}b')
        #                     change_col_inds = group.sorted_col_inds[[int(digit) for digit in reversed(i_bin)]]
        #
        #             if swap_graph.nodes[path[-1]]['group'] is None or path[-1] == path[0]:
        #                 return sum(swap_graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
        #             else:
        #                 pass
        #
        #         visited.add(current_path[-1])
        #         if current_path[-1] is None:
        #             path_weight = 0
        #         if current_path[-1] is None or current_path[-1] in current_path[:-1] or len(swap_graph.out_edges(current_path[-1])) == 0:
        #             pass
        #         for out_edge in swap_graph.out_edges(current_path[-1]):
        #
        #     visited = set()
        #     best_weight = 0
        #     best_path = []
        #     for node in swap_graph:
        #         if node not in visited:
        #             dfs(0, [node])

        # def implement_swap_graph(swap_graph: DiGraph, bases: ndarray, difference: ndarray, circuit: QuantumCircuit):
        #     def implement_cycles(swap_graph: DiGraph, bases: ndarray, difference: ndarray, circuit: QuantumCircuit):
        #         def get_cycle_weight(cycle: list[tuple[int, ...]]) -> int:
        #             return sum(swap_graph[u][v]['weight'] for u, v in zip(cycle, cycle[1:] + [cycle[0]]))
        #
        #         cycles = list(nx.simple_cycles(swap_graph))
        #         cycle_weights = [get_cycle_weight(cycle) for cycle in cycles]
        #         cycle_inds = np.argsort(cycle_weights)[::-1]
        #         for cycle_ind in cycle_inds:
        #             cycle = cycles[cycle_ind]
        #
        #     implement_cycles(swap_graph, bases, difference, circuit)


# class PermutationCircuitGeneratorSparse(PermutationCircuitGenerator):
#     """ Uses manual CX composition that works well for sparse state permutations. """
#
#     def solve_bilinear_problem(self, matrix: ndarray) -> (ndarray, ndarray):
#         """ Solves binary bilinear problem, i.e. find binary vectors x (m x 1) and y (n x 1) such that x^T * A * y is maximized for some m x n problem matrix A.
#         The solution is found by semi-definite relaxation. """
#         x = Variable(matrix.shape[0], boolean=True)
#         y = Variable(matrix.shape[1], boolean=True)
#         z = Variable(matrix.shape, boolean=True)
#         objective = Maximize(cp.sum(multiply(matrix, z)) - cp.sum(y) / 2)
#         constraints = []
#         for i in range(z.shape[0]):
#             for j in range(z.shape[1]):
#                 constraints.append(z[i, j] <= x[i])
#                 constraints.append(z[i, j] <= y[j])
#                 constraints.append(z[i, j] >= x[i] + y[j] - 1)
#         problem = Problem(objective, constraints)
#         problem.solve()
#         return x.value.astype(bool), y.value.astype(bool)
#
#     def find_mcx_gate(self, bases: ndarray, differences: ndarray, desired_rows: ndarray, desired_cols: ndarray) -> dict:
#         """
#         Chooses the target and controls for the CX gate that tries to maximize the net number of errors fixed.
#         :param bases: 2D array where i-th row is a list of bits corresponding to i-th non-zero basis in the current state.
#         :param differences: 2D array where elem [i, j] is 1 if it needs to be flipped and -1 otherwise.
#         :param desired_rows: 1D boolean array of desired rows to be affected by this gate.
#         :param desired_cols: 1D boolean array of desired cols to be affected by this gate.
#         :return: Dict with the description of the gate.
#         """
#         def _update_solutions(criterion, next_rows, target_ind, solutions):
#             if criterion == 'error':
#                 metric = np.sum(errors_by_rows[next_rows])
#             elif criterion == 'alignment':
#                 desired_overlap = np.count_nonzero(next_rows & desired_rows)
#                 undesired_overlap = np.count_nonzero(next_rows & ~desired_rows)
#                 if desired_overlap == 0:
#                     ratio = 0
#                 elif undesired_overlap == 0:
#                     ratio = float('inf')
#                 else:
#                     ratio = desired_overlap / undesired_overlap
#                 metric = [ratio, desired_overlap, control_ind not in desired_cols, np.sum(errors_by_rows[next_rows])]
#
#             if solutions[-1].get('metric') is None or solutions[-1].get('metric') < metric:
#                 solutions[-1]['metric'] = metric
#                 solutions[-1]['target_ind'] = target_ind
#                 solutions[-1]['control_ind'] = control_ind
#                 solutions[-1]['control_val'] = control_val
#                 solutions[-1]['next_rows'] = next_rows
#                 solutions[-1]['extra_targets'] = []
#
#         desired_cols = np.where(desired_cols)[0]
#         final_target_ind = desired_cols[0] if len(desired_cols) == 1 else None
#         errors_by_rows = np.sum(differences[:, desired_cols], 1)
#         error_solutions = []
#         alignment_solutions = []
#         selected_control_inds = []
#         current_rows = np.ones(bases.shape[0], dtype=bool)
#         for control_num in range(1, bases.shape[1]):
#             error_solutions.append({})
#             alignment_solutions.append({})
#             for control_ind in range(bases.shape[1]):
#                 if control_ind == final_target_ind or control_ind in selected_control_inds:
#                     continue
#                 for control_val in range(2):
#                     if control_ind in desired_cols:
#                         target_options = set(desired_cols) - {control_ind} if final_target_ind is None else [final_target_ind]
#                         for target_ind in target_options:
#                             if control_val == 0:
#                                 next_rows = current_rows & (np.all(bases[:, [target_ind, control_ind]] == [0, 0], 1) | np.all(bases[:, [target_ind, control_ind]] == [1, 1], 1))
#                             else:
#                                 next_rows = current_rows & (np.all(bases[:, [target_ind, control_ind]] == [0, 1], 1) | np.all(bases[:, [target_ind, control_ind]] == [1, 0], 1))
#                             _update_solutions('error', next_rows, target_ind, error_solutions)
#                             _update_solutions('alignment', next_rows, target_ind, alignment_solutions)
#                     else:
#                         next_rows = current_rows & (bases[:, control_ind] == control_val)
#                         target_ind = final_target_ind if final_target_ind is not None else next(iter(set(desired_cols) - {control_ind}))
#                         _update_solutions('error', next_rows, target_ind, error_solutions)
#                         _update_solutions('alignment', next_rows, target_ind, alignment_solutions)
#
#             selected_rows_error = np.sum(differences[error_solutions[-1]['next_rows'], :], 0)
#             for col in range(len(selected_rows_error)):
#                 if col in desired_cols or col in selected_control_inds or col == error_solutions[-1]['control_ind'] or selected_rows_error[col] <= 0:
#                     continue
#                 error_solutions[-1]['metric'] += selected_rows_error[col]
#                 error_solutions[-1]['extra_targets'].append(col)
#
#             if alignment_solutions[-1]['metric'][0] == float('inf'):
#                 break
#             current_rows = alignment_solutions[-1]['next_rows']
#             selected_control_inds.append(alignment_solutions[-1]['control_ind'])
#             if alignment_solutions[-1]['control_ind'] in desired_cols:
#                 final_target_ind = alignment_solutions[-1]['target_ind']
#
#         errors_by_controls = np.array([solution['metric'] for solution in error_solutions])
#         error_control_ratio = errors_by_controls / np.arange(1, len(error_solutions) + 1)
#         best_solution_ind = np.argmax(error_control_ratio)
#         primary_target_ind = error_solutions[best_solution_ind]['target_ind']
#         conjugated_target_inds = desired_cols.tolist() + error_solutions[best_solution_ind]['extra_targets']
#         control_inds = [solution['control_ind'] for solution in alignment_solutions[:best_solution_ind]] + [error_solutions[best_solution_ind]['control_ind']]
#         control_vals = [solution['control_val'] for solution in alignment_solutions[:best_solution_ind]] + [error_solutions[best_solution_ind]['control_val']]
#         affected_states = error_solutions[best_solution_ind]['next_rows']
#         return make_dict(primary_target_ind, conjugated_target_inds, control_inds, control_vals, affected_states)
#
#     def apply_mcx_gate(self, gate_description: dict, bases: ndarray, difference: ndarray, circuit: QuantumCircuit):
#         """
#         Applies specified multi-controlled X gate and updates current state, differences and circuit.
#         :param gate_description: Dict with the parameters describing the mcx gate.
#         :param bases: Current set of non-zero bases (output, to be updated).
#         :param difference: Current differences with the target set of bases (output, to be updated).
#         :param circuit: Current quantum circuit (output, to be updated with the new gates).
#         """
#         if 'control_inds' not in gate_description:
#             affected_states = list(range(bases.shape[0]))
#             affected_targets = gate_description['primary_target_ind']
#             circuit.x(gate_description['primary_target_ind'])
#         else:
#             affected_states = gate_description['affected_states']
#             affected_targets = gate_description['conjugated_target_inds']
#
#             cx_conjugation_circuit = QuantumCircuit(circuit.num_qubits)
#             for target_ind in gate_description['conjugated_target_inds']:
#                 if gate_description['primary_target_ind'] == target_ind:
#                     continue
#                 cx_conjugation_circuit.cx(gate_description['primary_target_ind'], target_ind)
#
#             if np.any(bases[:, gate_description['primary_target_ind']] == 1):
#                 circuit.compose(cx_conjugation_circuit, inplace=True)
#             ctrl_state = ''.join([str(num) for num in gate_description['control_vals']])
#             circuit.mcx(gate_description['control_inds'], gate_description['primary_target_ind'], ctrl_state=ctrl_state[::-1])
#             if np.any(bases[:, gate_description['primary_target_ind']] == 0):
#                 circuit.compose(cx_conjugation_circuit.reverse_ops(), inplace=True)
#         bases[np.ix_(affected_states, affected_targets)] ^= 1
#         difference[np.ix_(affected_states, affected_targets)] *= -1
#
#     def get_permutation_circuit(self, permutation: dict[str, str]) -> QuantumCircuit:
#         all_bases_old = np.array([[int(val) for val in basis] for basis in permutation])
#         all_bases_new = np.array([[int(val) for val in basis] for basis in permutation.values()])
#         difference = (all_bases_old ^ all_bases_new) * 2 - 1
#         qc = QuantumCircuit(all_bases_old.shape[1])
#         for i in range(all_bases_old.shape[1]):
#             if np.sum(difference[:, i]) > 0:
#                 self.apply_mcx_gate(dict(primary_target_ind=i), all_bases_old, difference, qc)
#         while np.any(difference > 0):
#             rows, cols = self.solve_bilinear_problem(difference)
#             gate_description = self.find_mcx_gate(all_bases_old, difference, rows, cols)
#             self.apply_mcx_gate(gate_description, all_bases_old, difference, qc)
#         return qc.reverse_bits()
