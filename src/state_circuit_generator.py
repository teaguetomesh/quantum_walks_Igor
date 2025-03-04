""" Module for functions related to circuit generation for arbitrary state preparation via quantum walks. """
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy import ndarray
from pysat.examples.hitman import Hitman
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate, RXGate
from qiskit.quantum_info import Statevector

from src.permutation_circuit_generator import PermutationCircuitGenerator
from src.permutation_generator import DensePermutationGenerator
from src.qiskit_utilities import remove_leading_cx_gates
from src.quantum_walks import PathSegment, PathFinder
from src.utilities import greedy_decision_tree
from src.validation import get_state_vector


class StateCircuitGenerator(ABC):
    """ Base class for generating circuits that prepare a given state. """

    @abstractmethod
    def generate_circuit(self, target_state: dict[str, complex]) -> QuantumCircuit:
        """ Generates a quantum circuit that prepares target_state, described as dictionary of bitstrings and corresponding probability amplitudes. """
        pass


class QiskitDefaultGenerator(StateCircuitGenerator):
    """ Generates state preparation circuit via qiskit's default built-in method. """

    def generate_circuit(self, target_state: dict[str, complex]) -> QuantumCircuit:
        target_state_vector = get_state_vector(target_state)
        num_qubits = len(next(iter(target_state.keys())))
        circuit = QuantumCircuit(num_qubits)
        circuit.prepare_state(target_state_vector)
        return circuit


@dataclass(kw_only=True)
class SingleEdgeGenerator(StateCircuitGenerator):
    """
    Finds a particular path through the target basis states and generates the circuit based on single-edge segments of that path.
    :var path_finder: PathFinder class implementing a particular heuristic for finding path through the basis states.
    :var reduce_controls: True to search for minimally necessary state of controls. False to use all n-1 controls (to save classical time).
    :var remove_leading_cx: True to remove leading CX gates whose controls are never satisfied.
    :var add_barriers: True to insert barriers between path segments.
    """
    path_finder: PathFinder
    reduce_controls: bool = True
    remove_leading_cx: bool = True
    add_barriers: bool = False

    @staticmethod
    def update_visited(visited: list[list[int]], control: int, target: int):
        """
        Updates visited nodes to reflect the action of specified CX gates.
        :param visited: List of basis labels of visited states.
        :param control: Index of the control qubit for the CX operation.
        :param target: Index of the target qubit for the CX operation.
        """
        for label in visited:
            if label[control] == 1:
                label[target] = 1 - label[target]

    @staticmethod
    def find_min_control_set(existing_states: list[list[int]], target_state_ind: int, interaction_ind: int) -> list[int]:
        """
        Finds minimum set of control necessary to select the target state.
        :param existing_states: List of states with non-zero amplitudes.
        :param target_state_ind: Index of the target state in the existing_states.
        :param interaction_ind: Index of target qubit for the controlled operation (to exclude from consideration for the control set).
        :return: Minimum set of control indices necessary to select the target state.
        """
        get_diff_inds = lambda state1, state2: [ind for ind in range(len(state1)) if ind != interaction_ind and state1[ind] != state2[ind]]
        difference_inds = [get_diff_inds(state, existing_states[target_state_ind]) for state_ind, state in enumerate(existing_states) if state_ind != target_state_ind]
        hitman = Hitman()
        for inds_set in difference_inds:
            hitman.hit(inds_set)
        return hitman.get()

    def convert_path_to_circuit(self, path: list[PathSegment]) -> QuantumCircuit:
        """
        Converts quantum walks to qiskit circuit.
        :param path: List of path segments, describing the state preparation path.
        :return: Implementing circuit.
        """
        starting_state = path[0].labels[0]
        qc = QuantumCircuit(len(starting_state))
        indices_1 = [ind for ind, elem in enumerate(starting_state) if elem == "1"]
        for ind in indices_1:
            qc.x(ind)
        if self.add_barriers:
            qc.barrier()

        visited = [[int(char) for char in starting_state]]
        for segment in path:
            origin = [int(char) for char in segment.labels[0]]
            destination = [int(char) for char in segment.labels[1]]
            diff_inds = np.where(np.array(origin) != np.array(destination))[0]
            interaction_ind = diff_inds[0]

            visited_transformed = copy.deepcopy(visited)
            for ind in diff_inds[1:]:
                qc.cx(interaction_ind, ind)
                SingleEdgeGenerator.update_visited(visited_transformed, interaction_ind, ind)

            origin_ind = visited.index(origin)
            if self.reduce_controls:
                control_indices = SingleEdgeGenerator.find_min_control_set(visited_transformed, origin_ind, interaction_ind)
            else:
                control_indices = [ind for ind in range(len(origin)) if ind != interaction_ind]

            for ind in control_indices:
                if visited_transformed[origin_ind][ind] == 0:
                    qc.x(ind)

            rz_angle = 2 * segment.phase_time
            if origin[interaction_ind] == 1:
                rz_angle *= -1
            if rz_angle != 0:
                rz_gate = RZGate(rz_angle)
                if len(control_indices) > 0:
                    rz_gate = rz_gate.control(len(control_indices))
                qc.append(rz_gate, control_indices + [interaction_ind])

            rx_angle = 2 * segment.amplitude_time
            if rx_angle != 0:
                rx_gate = RXGate(rx_angle)
                if len(control_indices) > 0:
                    rx_gate = rx_gate.control(len(control_indices))
                qc.append(rx_gate, control_indices + [interaction_ind])
                visited.append(destination)

            for ind in control_indices:
                if visited_transformed[origin_ind][ind] == 0:
                    qc.x(ind)

            for ind in reversed(diff_inds[1:]):
                qc.cx(interaction_ind, ind)
            if self.add_barriers:
                qc.barrier()

        if self.remove_leading_cx:
            qc = remove_leading_cx_gates(qc)

        return qc.reverse_bits()

    def generate_circuit(self, target_state: dict[str, complex]) -> QuantumCircuit:
        path = self.path_finder.get_path(target_state)
        circuit = self.convert_path_to_circuit(path)
        return circuit


@dataclass(kw_only=True)
class DensePermuteGenerator(StateCircuitGenerator):
    """ Generates a circuit for a dense state, then permutes it to the target state. """
    permutation_generator: DensePermutationGenerator
    permutation_circuit_generator: PermutationCircuitGenerator

    @staticmethod
    def map_to_dense_state(state: dict[str, complex], dense_permutation: dict[str, str], dense_qubits: list[int]) -> list[complex]:
        """ Permutes state according to given dense permutation and returns contiguous list of amplitudes where i-th element corresponds to basis i. """
        dense_state = [0] * 2 ** len(dense_qubits)
        for basis, amplitude in state.items():
            mapped_basis = dense_permutation[basis]
            dense_coords = ''.join([mapped_basis[i] for i in dense_qubits])
            ind = int(dense_coords, 2)
            dense_state[ind] = amplitude
        return dense_state

    @abstractmethod
    def get_dense_state_circuit(self, dense_state: list[complex]) -> QuantumCircuit:
        pass

    def generate_circuit(self, target_state: dict[str, complex]) -> QuantumCircuit:
        dense_permutation, dense_qubits = self.permutation_generator.get_permutation(target_state)
        dense_state = self.map_to_dense_state(target_state, dense_permutation, dense_qubits)
        dense_state_qc = self.get_dense_state_circuit(dense_state)
        inverse_permutation = {val: key for key, val in dense_permutation.items()}
        permutation_qc = self.permutation_circuit_generator.get_permutation_circuit(inverse_permutation)
        overall_qc = QuantumCircuit(permutation_qc.num_qubits)
        any_dense_basis = next(iter(inverse_permutation))
        sparse_qubits = list(set(range(len(any_dense_basis))) - set(dense_qubits))
        for qubit in sparse_qubits:
            if any_dense_basis[qubit] == '1':
                overall_qc.x(len(any_dense_basis) - qubit - 1)
        overall_qc.append(dense_state_qc, list(len(any_dense_basis) - 1 - np.array(dense_qubits)[::-1]))
        overall_qc.append(permutation_qc, range(permutation_qc.num_qubits))
        return overall_qc


@dataclass(kw_only=True)
class QiskitDenseGenerator(DensePermuteGenerator):
    """ Uses qiskit's built-in state preparation on dense state. """

    def get_dense_state_circuit(self, dense_state: list[complex]) -> QuantumCircuit:
        """ Returns a quantum circuit that prepares a dense state via qiskit's prepare_state method. """
        num_qubits = int(np.ceil(np.log2(len(dense_state))))
        qc = QuantumCircuit(num_qubits)
        qc.prepare_state(dense_state, range(num_qubits))
        return qc


@dataclass(kw_only=True)
class MultiEdgeDenseGenerator(DensePermuteGenerator):
    """ Uses multi-edge walk to prepare a dense state. """

    def apply_mcrx(self, time: float, target_ind: int, control_inds: ndarray, qc: QuantumCircuit, current_state: Statevector):
        """ Applies multi-controlled Rx gate with the given parameters to the given quantum circuit and updates current state vector. """
        if not np.isclose(time, 0):
            rx_gate = RXGate(2 * time)
            if len(control_inds) > 0:
                rx_gate = rx_gate.control(len(control_inds))
            target_ind = current_state.num_qubits - 2 - target_ind
            control_inds = current_state.num_qubits - 2 - control_inds
            gate_circuit = QuantumCircuit(current_state.num_qubits)
            gate_circuit.append(rx_gate, control_inds.tolist() + [target_ind])
            qc.compose(gate_circuit, range(current_state.num_qubits), inplace=True)
            return current_state.evolve(gate_circuit)
        return current_state

    def apply_mcrz(self, time: float, control_inds: ndarray, qc: QuantumCircuit, current_state: Statevector):
        """ Applies multi-controlled Rz gate with the given parameters to the given quantum circuit and updates current state vector. """
        if not np.isclose(time, 0):
            rz_gate = RZGate(2 * time)
            if len(control_inds) > 0:
                rz_gate = rz_gate.control(len(control_inds))
            control_inds = current_state.num_qubits - 2 - control_inds
            gate_circuit = QuantumCircuit(current_state.num_qubits)
            gate_circuit.append(rz_gate, control_inds.tolist() + [current_state.num_qubits - 1])
            qc.compose(gate_circuit, range(gate_circuit.num_qubits), inplace=True)
            return current_state.evolve(gate_circuit)
        return current_state

    def get_dense_state_circuit(self, dense_state: list[complex], num_qubits: int) -> QuantumCircuit:
        num_qubits_dense = int(np.log2(len(dense_state)))
        assert num_qubits_dense < num_qubits, 'State is too dense, need at least 1 free qubit for multi-edge phase shift to work'
        dense_state = np.array(dense_state)
        qc = QuantumCircuit(num_qubits)
        current_state = Statevector.from_int(0, 2 ** (num_qubits_dense + 1))
        for dim in range(num_qubits_dense):
            target_ind = num_qubits_dense - dim - 1
            for basis_ind in range(2 ** dim):
                amplitude_1 = current_state[basis_ind]
                amplitude_2 = current_state[basis_ind ^ 1 << dim]
                x = abs(amplitude_1)
                y = abs(amplitude_2)
                a = np.angle(amplitude_1) - np.angle(amplitude_2)
                p = np.sum(abs(dense_state[basis_ind :: 2 ** (dim + 1)]) ** 2)
                time_num = (x ** 2 * y ** 2 * (1 + np.exp(2j * a)) ** 2 - 4 * p * np.exp(2j * a) * (x ** 2 + y ** 2 - p)) ** 0.5 + np.exp(1j * a) * (x ** 2 + y ** 2 - 2 * p)
                time_denom = np.exp(1j * a) * (y ** 2 - x ** 2) + x * y * (1 - np.exp(2j * a))
                time = -0.5j * np.log(time_num / time_denom)
                assert time.imag < 1e-5, f'Failed to solve for time. Time: {time}'
                time = time.real
                basis_ind_bin = format(basis_ind, f'0{num_qubits_dense}b')
                controls = np.where(np.array(list(basis_ind_bin)) == '1')[0]
                current_state = self.apply_mcrx(time, target_ind, controls, qc, current_state)

        for basis_ind in range(len(current_state) // 2):
            time = np.angle(current_state[basis_ind]) - np.angle(dense_state[basis_ind])
            basis_ind_bin = format(basis_ind, f'0{num_qubits_dense}b')
            controls = np.where(np.array(list(basis_ind_bin)) == '1')[0]
            current_state = self.apply_mcrz(time, controls, qc, current_state)
        return qc


@dataclass
class MultiEdgeSparseGenerator(StateCircuitGenerator):
    """ Uses sparse state preparation with multi-edge approach. """
    permutation_circuit_generator: PermutationCircuitGenerator

    @dataclass
    class TreeEdge:
        bases: (str, str)
        transformed_bases: (str, str) = None
        probabilities: (float, float) = None

    @dataclass
    class TreeLevel:
        rx_dims: list[int]
        target_ind: int
        edges: list[(TreeEdge, list[int], str)]

    def get_basis_distance(self, basis1: str, basis2: str, rx_dims: list[int] | None) -> int:
        """ Calculates modified Hamming distance between two bitstrings of equal length. """
        diff_inds = np.array([basis1[i] != basis2[i] for i in range(len(basis1))])
        if rx_dims is not None:
            diff_inds[rx_dims] ^= True
        return sum(diff_inds)

    def find_pairs_greedy(self, distances: list[(int, int, int)]) -> list[(int, int)]:
        """ Greedily picks specified number of pairs with minimal distance. """
        distances = sorted(distances, key=lambda x: x[2])
        pairs = []
        used_inds = set()
        for next_pair in distances:
            if next_pair[0] in used_inds or next_pair[1] in used_inds:
                continue
            pairs.append(next_pair)
            used_inds.update(next_pair[:2])
        return pairs

    def find_edges_dim(self, bases: list[str], rx_dims: list[int]) -> (int, list[TreeEdge]):
        """ Finds given number of best pairs among given bases with given Rx dimensions. Returns total score of pairs and the corresponding edges. """
        if len(rx_dims) == 0:
            return np.inf, None

        pairwise_distances = []
        for i in range(len(bases)):
            for j in range(i + 1, len(bases)):
                distance = self.get_basis_distance(bases[i], bases[j], rx_dims)
                pairwise_distances.append((i, j, distance))

        pairs = self.find_pairs_greedy(pairwise_distances)
        total_weight = sum(pair[2] for pair in pairs)
        edges = [self.TreeEdge((bases[pair[0]], bases[pair[1]])) for pair in pairs]
        return total_weight, edges

    def find_edges(self, bases: list[str]) -> (list[int], list[TreeEdge]):
        """ Finds given number of best pairs among given bases. Greedily chooses the best interaction dimensions for the Rx gate. """
        target_func = lambda rx_dims, _: -self.find_edges_dim(bases, rx_dims)
        node = greedy_decision_tree([1] * len(bases[0]), target_func, False, 1, True)[0]
        rx_dims = node.groups
        edges = node.extra_output[0]
        return rx_dims, edges

    def form_remaining_edges(self, edges: list[TreeEdge], bases: list[str]):
        """ Forms remaining edges to complete a full level and appends them to given list of edges. """
        used_bases = {basis for edge in edges for basis in edge.bases}
        for basis in bases:
            if basis in used_bases:
                continue
            edges.append(self.TreeEdge((basis, None)))

    def orient_edges(self, edges: list[TreeEdge], target_ind: int) -> list[TreeEdge]:
        """ Chooses origin. Returns updated edges. """
        oriented_edges = copy.deepcopy(edges)
        for edge in oriented_edges:
            if edge.bases[1] is not None and edge.bases[1][target_ind] == '0':
                edge.bases = edge.bases[::-1]
        return oriented_edges

    def change_basis(self, basis: str, change_inds: list[int]) -> str:
        """ Flips bitstring in the specified indices. """
        changed_basis = np.array([int(val) for val in basis])
        changed_basis[change_inds] ^= 1
        changed_basis = ''.join([str(val) for val in changed_basis])
        return changed_basis

    def change_basis_if(self, basis: str, change_inds: list[int], control_ind: int) -> str:
        """ Flips bitstring in the specified indices if the control index is 1. """
        if basis[control_ind] == '0':
            return basis
        return self.change_basis(basis, change_inds)

    def apply_transform(self, edges: list[TreeEdge], target_ind: int, rx_dims: list[int]):
        """ Adds transformed origin. """
        change_inds = [ind for ind in rx_dims if ind != target_ind]
        for edge in edges:
            edge.transformed_bases = tuple(self.change_basis_if(basis, change_inds, target_ind) if basis is not None else None for basis in edge.bases)

    def get_different_inds(self, basis_1: str, basis_2: str, ignore_ind: int) -> list[int]:
        """ Returns different indices between two bases, ignoring ignore_ind. """
        return [ind for ind in range(len(basis_1)) if ind != ignore_ind and basis_1[ind] != basis_2[ind]]

    def calculate_different_ind_matrix(self, bases: list[str], ignore_ind: int) -> ndarray[list[int]]:
        """ Calculates a matrix where element [i, j] is a list of different indices between i-th and j-th bases, except ignore_ind. """
        diff_ind_matrix = np.empty((len(bases), len(bases)), dtype=object)
        for i in range(diff_ind_matrix.shape[0]):
            for j in range(i, diff_ind_matrix.shape[1]):
                diff_ind_matrix[i, j] = diff_ind_matrix[j, i] = self.get_different_inds(bases[i], bases[j], ignore_ind)
        return diff_ind_matrix

    def solve_minimum_hitting_set(self, sets: list[list[int]]) -> list[int]:
        """ Finds the smallest set of integers that overlaps with all given sets. """
        hitman = Hitman()
        for set in sets:
            hitman.hit(set)
        solution = hitman.get()
        return solution

    def get_cx_cost(self, num_controls: int) -> int:
        """ Returns the number of CX gates in the decomposition of multi-controlled Rx gate with the specified number of controls. """
        cx_by_num_controls = [0, 2, 8, 20, 24, 40, 56, 80, 104]
        if num_controls < len(cx_by_num_controls):
            return cx_by_num_controls[num_controls]
        else:
            return cx_by_num_controls[-1] + (num_controls - len(cx_by_num_controls) - 1) * 16

    def find_smallest_control_set(self, diff_ind_matrix: ndarray[list[int]], ignore_inds: list[int]) -> (int, list[int]):
        """ Returns the smallest control set necessary to distinguish a state given by the last index in ignore_inds from other states. """
        sets = []
        for col_ind, diff_inds in enumerate(diff_ind_matrix[ignore_inds[-1], :]):
            if col_ind in ignore_inds:
                continue
            sets.append(diff_inds)
        control_inds = self.solve_minimum_hitting_set(sets)
        cost = self.get_cx_cost(len(control_inds))
        return cost, control_inds

    def find_implementation_order(self, bases: list[str], target_ind: int, order_inds: list[int]) -> (int, list[(int, list[int])]):
        """ Finds the best order to implement a given set of bases (based on control reduction) with given target ind and rx_dims. """
        diff_ind_matrix = self.calculate_different_ind_matrix(bases, target_ind)
        target_func = lambda group_inds, _: -self.find_smallest_control_set(diff_ind_matrix, np.array(order_inds)[group_inds])
        last_node = greedy_decision_tree([1] * len(order_inds), target_func, True, 1, False)[0]
        all_nodes = [last_node]
        while all_nodes[-1].parent is not None:
            all_nodes.append(all_nodes[-1].parent)
        total_cost = sum(node.score for node in all_nodes)
        order = [(node.groups[-1], node.extra_output[0]) for node in all_nodes]
        return total_cost, order

    def get_substring(self, string: str, inds: list[int]) -> str:
        """ Returns substring at specified indices. """
        return ''.join([string[i] for i in inds])

    def get_ordered_edges(self, oriented_edges: list[TreeEdge], order: list[(int, list[int])]) -> list[(TreeEdge, list[int], str)]:
        """ Returns edges and corresponding control indices and values according to given order. """
        ordered_edges = []
        for edge_ind, control_inds in order:
            control_vals = self.get_substring(oriented_edges[edge_ind].transformed_bases[0], control_inds)
            ordered_edges.append((oriented_edges[edge_ind], control_inds, control_vals))
        return ordered_edges

    def find_implementing_gates(self, rx_dims: list[int], edges: list[TreeEdge]) -> TreeLevel:
        """ Finds controls, target and order for the gates implementing given rx_dims and pairs. Returns description of tree level. """
        num_none_edges = 1 if edges[-1].bases[1] is None else 0
        non_none_inds = list(range(len(edges) - num_none_edges))
        best_sequence = (np.inf, None, None)
        for target_dim in rx_dims:
            oriented_edges = self.orient_edges(edges, target_dim)
            self.apply_transform(oriented_edges, target_dim, rx_dims)
            parent_bases = [edge.transformed_bases[0] for edge in oriented_edges]
            cost, order = self.find_implementation_order(parent_bases, target_dim, non_none_inds)
            if num_none_edges == 1:
                order = [(len(edges) - 1, [])] + order
            if cost < best_sequence[0]:
                ordered_edges = self.get_ordered_edges(oriented_edges, order)
                best_sequence = (cost, target_dim, ordered_edges)
        return self.TreeLevel(rx_dims, *best_sequence[1:])

    def calculate_probabilities(self, tree: list[TreeLevel], target_state: dict[str, complex]):
        """ Calculates probabilities for all tree edges. """
        for edge, _, _ in tree[-1].edges:
            edge.probabilities = (abs(target_state[edge.bases[0]]) ** 2, abs(target_state.get(edge.bases[1], 0)) ** 2)
        for level, next_level in reversed(list(zip(tree[:-1], tree[1:]))):
            for edge, _, _ in level.edges:
                prob_origin = next(sum(next_edge.probabilities) for next_edge, _, _ in next_level.edges if next_edge.bases[0] == edge.bases[0])
                prob_dest = next(sum(next_edge.probabilities) for next_edge, _, _ in next_level.edges if next_edge.bases[0] == edge.bases[1]) if edge.bases[1] is not None else 0
                edge.probabilities = (prob_origin, prob_dest)

    def build_multiedge_tree(self, target_state: dict[str, complex]) -> list[TreeLevel]:
        """ Builds a hierarchy of basis clusters that minimizes total hamming distance between clusters (with restrictions).
        Returns a list of tree levels. Each level is a list of gates corresponding to each edge of the tree on that level. """
        target_bases = list(target_state)
        current_bases = target_bases[:]
        tree = []
        while len(current_bases) > 1:
            rx_dims, edges = self.find_edges(current_bases)
            if len(current_bases) > 2 * len(edges):
                self.form_remaining_edges(edges, current_bases)
            tree_level = self.find_implementing_gates(rx_dims, edges)
            tree.append(tree_level)
            current_bases = [edge.bases[0] for edge, _, _ in tree_level.edges]
        tree.reverse()
        self.calculate_probabilities(tree, target_state)
        return tree

    def update_state_phase(self, current_state: dict[str, complex], control_inds: list[int], control_vals: str, target_ind: int, rotation_time: float):
        """ Updates current state with the result of a phase rotation. """
        for basis in current_state:
            basis_control_vals = self.get_substring(basis, control_inds)
            if basis_control_vals != control_vals:
                continue
            sign = (-1) ** (basis[target_ind] == '0')
            current_state[basis] *= np.exp(1j * sign * rotation_time)

    def adjust_phase(self, edge: TreeEdge, control_inds: list[int], control_vals: str, target_ind: int, current_state: dict[str, complex], target_state: dict[str, complex],
                     circuit: QuantumCircuit, mode: str = 'self'):
        """ Adjusts phase for given edge to match target_state. If 'self' mode adjusts phase based on origin phase only.
        In 'average' mode adjusts phase to phase average in order to simultaneously fix origin and destination phases with an additional rotation later. """
        current_phase = np.angle(current_state[edge.transformed_bases[0]])
        target_phase = np.angle(target_state[edge.bases[0]])
        if mode == 'self':
            rotation_time = target_phase - current_phase
        elif mode == 'average':
            target_phase_2 = np.angle(target_state[edge.bases[1]])
            rotation_time = (target_phase + target_phase_2) / 2 + np.pi / 4 - current_phase
        else:
            raise Exception('Unknown mode')

        if np.isclose(rotation_time, 0, atol=1e-6):
            return
        if abs(rotation_time) > np.pi:
            rotation_time -= 2 * np.pi * np.sign(rotation_time)
        if edge.transformed_bases[0][target_ind] == '0':
            rotation_time *= -1
        rz_gate = RZGate(2 * rotation_time)
        if len(control_inds) > 0:
            rz_gate = rz_gate.control(len(control_inds), ctrl_state=control_vals[::-1])
        circuit.append(rz_gate, control_inds + [target_ind])
        self.update_state_phase(current_state, control_inds, control_vals, target_ind, rotation_time)

    def solve_rx_time(self, origin_amplitude: complex, destination_amplitude: complex, target_origin_probability: float) -> float:
        """ Finds rotation time necessary to achieve target probability on origin. """
        x = abs(origin_amplitude)
        y = abs(destination_amplitude)
        a = np.angle(origin_amplitude) - np.angle(destination_amplitude)
        p = target_origin_probability
        time_num = (x ** 2 * y ** 2 * (1 + np.exp(2j * a)) ** 2 - 4 * p * np.exp(2j * a) * (x ** 2 + y ** 2 - p)) ** 0.5 + np.exp(1j * a) * (x ** 2 + y ** 2 - 2 * p)
        time_denom = np.exp(1j * a) * (y ** 2 - x ** 2) + x * y * (1 - np.exp(2j * a))
        rotation_time = -0.5j * np.log(time_num / time_denom)
        assert abs(rotation_time.imag) < 1e-5, f'Failed to solve for time. Time: {rotation_time}'
        rotation_time = rotation_time.real
        if np.isclose(y, 0, atol=1e-6):
            rotation_time = abs(rotation_time)
        return rotation_time

    def update_state_amplitudes(self, current_state: dict[str, complex], control_inds: list[int], control_vals: str, target_ind: int, rotation_time: float):
        """ Updates amplitudes in the current state to match the state after Rx rotation and permutation. """
        for basis in list(current_state):
            neighbor = self.change_basis(basis, [target_ind])
            if basis[target_ind] == '1' and neighbor in current_state:
                continue
            basis_control_vals = self.get_substring(basis, control_inds)
            if basis_control_vals != control_vals:
                continue
            origin_amplitude = current_state[basis]
            destination_amplitude = current_state.get(neighbor, 0)
            current_state[basis] = origin_amplitude * np.cos(rotation_time) - 1j * destination_amplitude * np.sin(rotation_time)
            current_state[neighbor] = -1j * origin_amplitude * np.sin(rotation_time) + destination_amplitude * np.cos(rotation_time)

    def transfer_amplitude(self, edge: TreeEdge, control_inds: list[int], control_vals: str, target_ind: int, current_state: dict[str, complex], circuit: QuantumCircuit):
        """ Transfers amplitude for given edge to match the target state. """
        origin_amplitude = current_state[edge.transformed_bases[0]]
        destination_amplitude = current_state.get(self.change_basis(edge.transformed_bases[0], [target_ind]), 0)
        total_prob = abs(origin_amplitude) ** 2 + abs(destination_amplitude) ** 2
        rotation_time = self.solve_rx_time(origin_amplitude, destination_amplitude, total_prob)
        rotation_time += self.solve_rx_time(total_prob ** 0.5, 0, edge.probabilities[0])
        if np.isclose(rotation_time, 0, atol=1e-6):
            return
        rx_gate = RXGate(2 * rotation_time)
        if len(control_inds) > 0:
            rx_gate = rx_gate.control(len(control_inds), ctrl_state=control_vals[::-1])
        circuit.append(rx_gate, control_inds + [target_ind])
        self.update_state_amplitudes(current_state, control_inds, control_vals, target_ind, rotation_time)

    def prepare_level_circuit(self, level: TreeLevel, current_state: dict[str, complex], target_state: dict[str, complex]) -> (QuantumCircuit, dict[str, complex]):
        """ Prepares a circuit for a given tree level. """
        conjugating_circuit = QuantumCircuit(len(next(iter(target_state))))
        for dim in level.rx_dims:
            if dim == level.target_ind:
                continue
            conjugating_circuit.cx(level.target_ind, dim)

        last_level = len(level.edges) >= len(target_state) / 2
        level_circuit = QuantumCircuit(conjugating_circuit.num_qubits)
        level_circuit.compose(conjugating_circuit, inplace=True)
        change_inds = [ind for ind in level.rx_dims if ind != level.target_ind]
        current_state = {self.change_basis_if(basis, change_inds, level.target_ind): amplitude for basis, amplitude in current_state.items()}

        if last_level:
            for edge, control_inds, control_vals in level.edges:
                if edge.bases[1] is None:
                    continue
                self.adjust_phase(edge, control_inds, control_vals, level.target_ind, current_state, target_state, level_circuit, 'average')
        for edge, control_inds, control_vals in level.edges:
            if edge.bases[1] is None:
                continue
            self.transfer_amplitude(edge, control_inds, control_vals, level.target_ind, current_state, level_circuit)
        if last_level:
            for edge, control_inds, control_vals in level.edges:
                self.adjust_phase(edge, control_inds, control_vals, level.target_ind, current_state, target_state, level_circuit)

        level_circuit.compose(conjugating_circuit.reverse_ops(), inplace=True)
        current_state = {self.change_basis_if(basis, change_inds, level.target_ind): amplitude for basis, amplitude in current_state.items()}
        return level_circuit.reverse_bits(), current_state

    def prepare_permutation_circuit(self, level: TreeLevel, current_state: dict[str, complex]) -> QuantumCircuit:
        """ Prepares a circuit that moves origin images to destinations and updates the current state. """
        permutation = dict()
        for edge, _, _ in level.edges:
            permutation[edge.bases[0]] = edge.bases[0]
            if edge.bases[1] is not None:
                image = self.change_basis(edge.bases[0], level.rx_dims)
                permutation[image] = edge.bases[1]
                if image != edge.bases[1]:
                    current_state[edge.bases[1]] = current_state.pop(image)
        permutation_circuit = self.permutation_circuit_generator.get_permutation_circuit(permutation)
        return permutation_circuit

    def implement_multiedge_tree(self, tree: list[TreeLevel], target_state: dict[str, complex]) -> QuantumCircuit:
        """ Converts multi-edge tree to its circuit implementation. """
        overall_circuit = QuantumCircuit(len(next(iter(target_state))))
        for ind, val in enumerate(tree[0].edges[0][0].bases[0]):
            if val == '1':
                overall_circuit.x(overall_circuit.num_qubits - 1 - ind)

        current_state = {tree[0].edges[0][0].bases[0]: 1}
        for level in tree:
            level_circuit, current_state = self.prepare_level_circuit(level, current_state, target_state)
            overall_circuit.compose(level_circuit, inplace=True)
            permutation_circuit = self.prepare_permutation_circuit(level, current_state)
            if permutation_circuit.num_qubits > overall_circuit.num_qubits:
                overall_circuit = QuantumCircuit(permutation_circuit.num_qubits).compose(overall_circuit)
            overall_circuit.compose(permutation_circuit, inplace=True)
        return overall_circuit

    def generate_circuit(self, target_state: dict[str, complex]) -> QuantumCircuit:
        tree = self.build_multiedge_tree(target_state)
        circuit = self.implement_multiedge_tree(tree, target_state)
        return circuit
