""" Module for functions related to circuit generation for arbitrary state preparation via quantum walks. """
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass

import networkx as nx
import numpy as np
from networkx.algorithms import matching
from numpy import ndarray
from pysat.examples.hitman import Hitman
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate, RXGate
from qiskit.quantum_info import Statevector

from src.permutation_circuit_generator import PermutationCircuitGenerator
from src.permutation_generator import PermutationGenerator
from src.quantum_walks import PathSegment, PathFinder
from src.validation import get_state_vector


class StateCircuitGenerator(ABC):
    """ Base class for generating circuits that prepare a given state. """

    @abstractmethod
    def generate_circuit(self, target_state: dict[str, complex]) -> QuantumCircuit:
        """ Generates a quantum circuit that prepares target_state, described as dictionary of bitstrings and corresponding probability amplitudes. """
        pass


class StateCircuitGeneratorQiskitDefault(StateCircuitGenerator):
    """ Generates state preparation circuit via qiskit's default built-in method. """

    def generate_circuit(self, target_state: dict[str, complex]) -> QuantumCircuit:
        target_state_vector = get_state_vector(target_state)
        num_qubits = len(next(iter(target_state.keys())))
        circuit = QuantumCircuit(num_qubits)
        circuit.prepare_state(target_state_vector)
        return circuit


@dataclass(kw_only=True)
class StateCircuitGeneratorSingleEdge(StateCircuitGenerator):
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
                StateCircuitGeneratorSingleEdge.update_visited(visited_transformed, interaction_ind, ind)

            origin_ind = visited.index(origin)
            if self.reduce_controls:
                control_indices = StateCircuitGeneratorSingleEdge.find_min_control_set(visited_transformed, origin_ind, interaction_ind)
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
            qc = StateCircuitGeneratorSingleEdge.remove_leading_cx_gates(qc)

        return qc.reverse_bits()

    def generate_circuit(self, target_state: dict[str, complex]) -> QuantumCircuit:
        path = self.path_finder.get_path(target_state)
        circuit = self.convert_path_to_circuit(path)
        return circuit


@dataclass(kw_only=True)
class StateCircuitGeneratorDensePermute(StateCircuitGenerator):
    """ Generates a circuit for a dense state, then permutes it to the target state.
    dense_permutation_generator has to generate a permutation into the smallest number of qubits able to fit the target state. """
    dense_permutation_generator: PermutationGenerator
    permutation_circuit_generator: PermutationCircuitGenerator

    @staticmethod
    def map_to_dense_state(state: dict[str, complex], dense_permutation: dict[str, str]) -> list[complex]:
        """ Permutes state according to given dense permutation and returns contiguous list of amplitudes where i-th element corresponds to basis i. """
        num_qubits_dense = int(np.ceil(np.log2(len(state))))
        dense_state = [0] * 2 ** num_qubits_dense
        for i, (basis, amplitude) in enumerate(state.items()):
            ind = int(dense_permutation[basis], 2)
            dense_state[ind] = amplitude
        return dense_state

    @abstractmethod
    def get_dense_state_circuit(self, dense_state: list[complex]) -> QuantumCircuit:
        pass

    def generate_circuit(self, target_state: dict[str, complex]) -> QuantumCircuit:
        dense_permutation = self.dense_permutation_generator.get_permutation(target_state)
        dense_state = self.map_to_dense_state(target_state, dense_permutation)
        dense_state_qc = self.get_dense_state_circuit(dense_state)
        inverse_permutation = {val: key for key, val in dense_permutation.items()}
        permutation_qc = self.permutation_circuit_generator.get_permutation_circuit(inverse_permutation)
        overall_qc = QuantumCircuit(permutation_qc.num_qubits)
        overall_qc.append(dense_state_qc, range(dense_state_qc.num_qubits))
        overall_qc.append(permutation_qc, range(permutation_qc.num_qubits))
        return overall_qc


@dataclass(kw_only=True)
class StateCircuitGeneratorQiskitDense(StateCircuitGeneratorDensePermute):
    """ Uses qiskit's built-in state preparation on dense state. """

    def get_dense_state_circuit(self, dense_state: list[complex]) -> QuantumCircuit:
        """ Returns a quantum circuit that prepares a dense state via qiskit's prepare_state method. """
        num_qubits = int(np.ceil(np.log2(len(dense_state))))
        qc = QuantumCircuit(num_qubits)
        qc.prepare_state(dense_state, range(num_qubits))
        return qc


@dataclass(kw_only=True)
class StateCircuitGeneratorMultiEdgeDense(StateCircuitGeneratorDensePermute):
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


class MultiEdgeGeneratorSparse(StateCircuitGenerator):
    """ Uses sparse state preparation with multi-edge approach. """

    # def get_edge_distance(self, edge1: tuple[str, str | None], edge2: tuple[str, str | None]) -> tuple[int, tuple[str, str]]:
    #     """ Calculates distance between two edges as minimum Hamming distances between their nodes. Returns distance and node pair achieving it. """
    #     min_distance = None
    #     node1 = node2 = None
    #     for n1, n2 in product(edge1, edge2):
    #         if n1 is None or n2 is None:
    #             continue
    #         distance = sum(n1[k] != n2[k] for k in range(len(n1)))
    #         if min_distance is None or distance < min_distance:
    #             min_distance = distance
    #             node1 = n1
    #             node2 = n2
    #     return min_distance, (node1, node2)

    def get_basis_distance(self, basis1: str, basis2: str, rx_dims: list[int] | None) -> int:
        """ Calculates modified Hamming distance between two bitstrings of equal length. """
        diff_inds = np.array([basis1[i] != basis2[i] for i in range(len(basis1))])
        if rx_dims is not None:
            diff_inds[rx_dims] ^= 1
        return sum(diff_inds)

    def find_pairs(self, bases: list[str], rx_dims: list[int] | None) -> list[tuple[str, str]]:
        """ Groups bases in pairs based on their modified Hamming distance. See get_basis_distance for details. """
        pairwise_distances = np.zeros((len(bases), len(bases)), dtype=int)
        for i in range(pairwise_distances.shape[0]):
            for j in range(i + 1, pairwise_distances.shape[1]):
                pairwise_distances[i, j] = pairwise_distances[j, i] = self.get_basis_distance(bases[i], bases[j], rx_dims)
        G = nx.from_numpy_array(pairwise_distances)
        return list(matching.min_weight_matching(G))

    def select_interaction_dims(self, pairs: list[tuple[str, str]]) -> list[int]:
        """ Selects difference indices that are present in more than half pairs, or the most common difference if no difference is sufficiently common. """
        diff_inds = np.array([np.array(list(node1)) != np.array(list(node2)) for node1, node2 in pairs])
        diff_counts = np.sum(diff_inds, 0)
        if any(diff_counts > len(pairs) // 2):
            return np.where(diff_counts > len(pairs) // 2)[0].tolist()
        return [np.argmax(diff_counts)]

    def build_clustering_tree(self, target_bases: list[str]) -> list[list[tuple[str, str]]]:
        """ Builds a hierarchy of basis clustering that minimizes total hamming distance between clusters.
         Returns a list where the 0th index corresponds to clustering level, 1st index to edges on that level, and 2nd index to bases in that edge. """
        num_levels = int(np.ceil(np.log2(len(target_bases))))
        clustering_tree = []
        for level in range(num_levels):
            num_edges = len(target_bases) - 2 ** (num_levels - 1) if level == 0 else 2 ** (num_levels - 1 - level)
            pairs = self.find_pairs(target_bases, None)
            interaction_dims = self.select_interaction_dims(pairs)
            pairs = self.find_pairs(target_bases, interaction_dims)
            selected_edge_inds = np.array(list(cluster)[:num_edges_last_layer])
            unselected_edge_inds = [ind for ind in range(len(prev_edges)) if ind not in selected_edge_inds]
            selected_edges = [distance_edges[*edge] for edge in selected_edge_inds]
            clustering_tree.append(selected_edges)
            new_edges = selected_edges + np.array(prev_edges, dtype=object)[unselected_edge_inds].tolist()
        return clustering_tree

    def generate_circuit(self, target_state: dict[str, complex]) -> QuantumCircuit:
        clustering_tree = self.build_clustering_tree(list(target_state))
