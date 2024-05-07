""" Module for converting quantum walks to circuit representation. """
import copy

from pysat.examples.hitman import Hitman
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate, RXGate

from src.quantum_walks import PathSegment


class PathConverter:
    """ Converts state preparation paths to the equivalent circuits. """
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
    def convert_path_to_circuit(path: list[PathSegment], reduce_controls: bool = True) -> QuantumCircuit:
        """
        Converts quantum walks to qiskit circuit.
        :param path: List of path segments, describing the state preparation path.
        :param reduce_controls: True to search for minimally necessary state of controls. False to use all n-1 controls (for debug purposes).
        :return: Implementing circuit.
        """
        starting_state = path[0].labels[0]
        qc = QuantumCircuit(len(starting_state))
        indices_1 = [ind for ind, elem in enumerate(starting_state) if elem == "1"]
        for ind in indices_1:
            qc.x(ind)
        # qc.barrier()

        visited = [[int(char) for char in starting_state]]
        for segment in path:
            origin = [int(char) for char in segment.labels[0]]
            destination = [int(char) for char in segment.labels[1]]
            z1 = copy.deepcopy(origin)
            z2 = copy.deepcopy(destination)
            if z1.count(1) > z2.count(1):
                z1, z2 = z2, z1

            zero_inds_z1 = [ind for ind, elem in enumerate(z1) if elem == 0]
            zero_inds_z2 = [ind for ind, elem in enumerate(z2) if elem == 0]
            interaction_ind = list(set(zero_inds_z1) - set(zero_inds_z2))[0]  # Gets the first nonzero index for z2 that's a zero for z1.
            zero_inds_z1.remove(interaction_ind)  # we will use this as a control to turn the z2 to all ones without affecting z1.
            for zero_ind in zero_inds_z1:
                qc.x(zero_ind)
                z2[zero_ind] = 1 - z2[zero_ind]

            visited_transformed = copy.deepcopy(visited)
            zero_inds_z2 = [ind for ind, elem in enumerate(z2) if elem == 0]
            for zero_ind in zero_inds_z2:
                qc.cx(interaction_ind, zero_ind)
                PathConverter.update_visited(visited_transformed, interaction_ind, zero_ind)

            origin_ind = visited.index(origin)
            if reduce_controls:
                control_indices = PathConverter.find_min_control_set(visited_transformed, origin_ind, interaction_ind)
            else:
                control_indices = [ind for ind in range(len(z1)) if ind != interaction_ind]

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

            for zero_ind in reversed(zero_inds_z2):
                qc.cx(interaction_ind, zero_ind)
            for zero_ind in zero_inds_z1:
                qc.x(zero_ind)
            # qc.barrier()

        return qc
