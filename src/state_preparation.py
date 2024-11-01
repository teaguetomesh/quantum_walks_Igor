""" Module for functions related to circuit generation for arbitrary state preparation via quantum walks. """
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy import ndindex
from pysat.examples.hitman import Hitman
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate, RXGate
from qiskit.converters import circuit_to_dag, dag_to_circuit

from src.permutation_circuit_generator import PermutationCircuitGenerator
from src.permutation_generator import PermutationGenerator
from src.quantum_walks import PathSegment, PathFinder
from src.validation import get_state_vector


class CircuitGenerator(ABC):
    """ Base class for generating circuits that prepare a given state. """

    @abstractmethod
    def generate_circuit(self, target_state: dict[str, complex]) -> QuantumCircuit:
        """ Generates a quantum circuit that prepares target_state, described as dictionary of bitstrings and corresponding probability amplitudes. """
        pass


class CircuitGeneratorQiskitDefault(CircuitGenerator):
    """ Generates state preparation circuit via qiskit's default built-in method. """

    def generate_circuit(self, target_state: dict[str, complex]) -> QuantumCircuit:
        target_state_vector = get_state_vector(target_state)
        num_qubits = len(next(iter(target_state.keys())))
        circuit = QuantumCircuit(num_qubits)
        circuit.prepare_state(target_state_vector)
        return circuit


@dataclass(kw_only=True)
class CircuitGeneratorPath(CircuitGenerator):
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

    @staticmethod
    def remove_leading_cx_gates(qc: QuantumCircuit) -> QuantumCircuit:
        """
        Removes leading CX gates whose controls are always false.
        :param qc: Input quantum circuit.
        :return: Optimized quantum circuit where the leading CX gates are removed where possible.
        """
        dag = circuit_to_dag(qc)
        wires = dag.wires
        # Go through the ops in each wire and remove cx ops until we run into a non cx operation.
        for w in wires:
            for node in list(dag.nodes_on_wire(w, only_ops=True)):
                if node.name == "barrier":
                    continue
                elif node.name != "cx":
                    break
                # The control is the current wire
                elif node.qargs[0] == w:
                    dag.remove_op_node(node)
                # CX but with target on qubit wire w.
                else:
                    break
        return dag_to_circuit(dag)

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
                CircuitGeneratorPath.update_visited(visited_transformed, interaction_ind, ind)

            origin_ind = visited.index(origin)
            if self.reduce_controls:
                control_indices = CircuitGeneratorPath.find_min_control_set(visited_transformed, origin_ind, interaction_ind)
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
            qc = CircuitGeneratorPath.remove_leading_cx_gates(qc)

        return qc.reverse_bits()

    def generate_circuit(self, target_state: dict[str, complex]) -> QuantumCircuit:
        path = self.path_finder.get_path(target_state)
        circuit = self.convert_path_to_circuit(path)
        return circuit


@dataclass(kw_only=True)
class CircuitGeneratorQiskitDense(CircuitGenerator):
    """ Uses qiskit's built-in state preparation on dense state.
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

    @staticmethod
    def get_state_preparation_circuit(dense_state: list[complex], num_qubits: int) -> QuantumCircuit:
        """ Returns a quantum circuit that prepares a dense state via qiskit's prepare_state method. """
        qc = QuantumCircuit(num_qubits)
        num_qubits_dense = int(np.ceil(np.log2(len(dense_state))))
        qc.prepare_state(dense_state, range(num_qubits_dense))
        return qc

    def generate_circuit(self, target_state: dict[str, complex]) -> QuantumCircuit:
        dense_permutation = self.dense_permutation_generator.get_permutation(target_state)
        dense_state = self.map_to_dense_state(target_state, dense_permutation)
        num_qubits = len(next(iter(target_state)))
        state_preparation_qc = self.get_state_preparation_circuit(dense_state, num_qubits)
        inverse_permutation = {val: key for key, val in dense_permutation.items()}
        permutation_qc = self.permutation_circuit_generator.get_permutation_circuit(inverse_permutation)
        overall_qc = state_preparation_qc.compose(permutation_qc)
        return overall_qc


class CircuitGeneratorMultiEdge:
    """ Uses multi-edge walk to generate a circuit for a dense state. """

    def get_state_preparation_circuit(self, dense_state: list[complex], num_qubits: int) -> QuantumCircuit:
        num_qubits_dense = int(np.ceil(np.log2(len(dense_state))))
        assert num_qubits_dense < num_qubits, 'State is too dense, need at least 1 free qubit for phase shift to work'
        dense_state_extended = dense_state + [0] * (2 ** num_qubits_dense - len(dense_state))
        target_cube = np.array(dense_state_extended).reshape([2] * num_qubits_dense)
        target_cube_prob = abs(target_cube) ** 2
        current_cube = np.zeros_like(target_cube_prob)
        current_cube[*[0] * num_qubits_dense] = 1
        qc = QuantumCircuit(num_qubits)
        for dim in range(num_qubits_dense):
            slice_indices = ndindex(*[2] * (dim + 1))
            target_slice_probs = np.array([np.sum(target_cube_prob[..., *index]) for index in slice_indices]).reshape(*[2] * (dim + 1))
            for index in slice_indices[:len(slice_indices) // 2]:
                target_prob = target_slice_probs[index]
                current_amplitude_1 = current_cube[*[0] * (num_qubits_dense - dim) + index]

        return qc.reverse_bits()
