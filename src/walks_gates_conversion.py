""" Module for converting quantum walks to circuit representation. """
import copy

import numpy as np
from pysat.examples.hitman import Hitman
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate, RXGate, PhaseGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Clifford
from copy import deepcopy
from qclib.gates.ldmcu import Ldmcu
from qiskit.circuit.library import UGate, UnitaryGate


from src.quantum_walks import PathSegment, LeafPathSegment


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
        # print(get_diff_inds)
        # print("interaction idx ", interaction_ind)
        # print("difference indices", difference_inds)
        hitman = Hitman()
        for inds_set in difference_inds:
            hitman.hit(inds_set)
        # print("hitman min set ", hitman.get())
        return hitman.get()

    @staticmethod
    def remove_leading_cx(qc: QuantumCircuit) -> QuantumCircuit:
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

    @staticmethod
    def _get_good_interaction_idx(diffs, visited_transformed, origin_ind, interaction_ind):
        visited_transformed=deepcopy(visited_transformed)
        diffs=deepcopy(diffs)
        diffs.remove(interaction_ind)
        for ind in diffs:
            PathConverter.update_visited(visited_transformed, interaction_ind, ind)
        return len(PathConverter.find_min_control_set(visited_transformed, origin_ind, interaction_ind))

    @staticmethod
    def get_good_interaction_idx(diffs, visited_transformed, origin_ind, destination):
        # print("z1 idx ", origin_ind)
        def counter_func(idx, diffs_origin_rest): 
            # print("idx ", idx)
            # print("diffs ", diffs_origin_rest)
            return sum([block.count(idx) for block in diffs_origin_rest])
        # def _create_remaining_basis(elem, basis):
        origin_node=visited_transformed[origin_ind]
        remaining_basis=deepcopy(visited_transformed)
        remaining_basis.pop(origin_ind)
        diffs_origin_rest =[[ind for ind in range(len(origin_node)) if origin_node[ind] != z1[ind]] for z1 in remaining_basis]
        # print("diffs ", diffs)
        # print("diffs origin rest ", diffs_origin_rest)
        # for idx in diffs:
            # print(counter_func(idx, diffs_origin_rest))
        sorted_diffs=sorted(diffs, key=
                            lambda temp_interaction_idx: (
                            PathConverter._get_good_interaction_idx(diffs, visited_transformed, origin_ind, temp_interaction_idx),
                            counter_func(temp_interaction_idx, diffs_origin_rest)))
        # print("sorted z1 z2 diffs, ", sorted_diffs)
        return sorted_diffs[0]

    @staticmethod
    def convert_path_to_circuit(path: list[PathSegment], reduce_controls: bool = True, remove_leading_cx: bool = True, add_barriers: bool = False) -> QuantumCircuit:
        """
        Converts quantum walks to qiskit circuit.
        :param path: List of path segments, describing the state preparation path.
        :param reduce_controls: True to search for minimally necessary state of controls. False to use all n-1 controls (for debug purposes).
        :param remove_leading_cx: True to remove leading CX gates whose controls are never satisfied.
        :param add_barriers: True to insert barriers between path segments.
        :return: Implementing circuit.
        """
        starting_state = path[0].labels[0]
        qc = QuantumCircuit(len(starting_state))
        indices_1 = [ind for ind, elem in enumerate(starting_state) if elem == "1"]
        for ind in indices_1:
            qc.x(ind)
        if add_barriers:
            qc.barrier()

        visited = [[int(char) for char in starting_state]]
        # print("#########################")
        # print(path)
        for segment in path:
            # print(path)
            origin = [int(char) for char in segment.labels[0]]
            destination = [int(char) for char in segment.labels[1]]
            diff_inds = np.where(np.array(origin) != np.array(destination))[0]
            interaction_ind = diff_inds[0]

            visited_transformed = copy.deepcopy(visited)
            for ind in diff_inds[1:]:
                qc.cx(interaction_ind, ind)
                PathConverter.update_visited(visited_transformed, interaction_ind, ind)

            origin_ind = visited.index(origin)
            if reduce_controls:
                control_indices = PathConverter.find_min_control_set(visited_transformed, origin_ind, interaction_ind)
            else:
                control_indices = [ind for ind in range(len(origin)) if ind != interaction_ind]


            # print("visited ", visited)
            # print("visited transformed ", visited_transformed)
            # print("origin index ", origin_ind)
            # print("interaction index ", interaction_ind)
            # print("origin ", origin)
            # print("destination ", destination)
            # print("controls ", control_indices)
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
            if add_barriers:
                qc.barrier()

        if remove_leading_cx:
            qc = PathConverter.remove_leading_cx(qc)

        return qc
    

    @staticmethod
    def convert_path_to_circuit_pframes(path: list[PathSegment], reduce_controls: bool = True, remove_leading_cx: bool = True, add_barriers: bool = False) -> QuantumCircuit:
        """
        Converts quantum walks to qiskit circuit.
        :param path: List of path segments, describing the state preparation path.
        :param reduce_controls: True to search for minimally necessary state of controls. False to use all n-1 controls (for debug purposes).
        :param remove_leading_cx: True to remove leading CX gates whose controls are never satisfied.
        :param add_barriers: True to insert barriers between path segments.
        :return: Implementing circuit.
        """
        starting_state = path[0].labels[0]
        qc = QuantumCircuit(len(starting_state))
        indices_1 = [ind for ind, elem in enumerate(starting_state) if elem == "1"]
        for ind in indices_1:
            qc.x(ind)
        if add_barriers:
            qc.barrier()

        # circuit for the ending rotations.
        circ_end=QuantumCircuit(len(starting_state))


        visited = [[int(char) for char in starting_state]]
        # print("#########################")
        # print(path)
        path_mutable=deepcopy(path)
        for idx, _ in enumerate(path):
            segment=path_mutable[idx]
            # print(path_mutable)
            origin = [int(char) for char in segment.labels[0]]
            destination = [int(char) for char in segment.labels[1]]
            diff_inds = np.where(np.array(origin) != np.array(destination))[0]
            diff_var_dummy=diff_inds.tolist()
            interaction_ind = diff_inds[0]
            # print("diffs: ", diff_var_dummy)

            visited_transformed = copy.deepcopy(visited)
            # print("visited: ", visited)
            # print("destination: ", destination)
            for ind in diff_inds[1:]:
                qc.cx(interaction_ind, ind)
                PathConverter.update_visited(visited_transformed, interaction_ind, ind)

            origin_ind = visited.index(origin)
            if reduce_controls:
                control_indices = PathConverter.find_min_control_set(visited_transformed, origin_ind, interaction_ind)
            else:
                control_indices = [ind for ind in range(len(origin)) if ind != interaction_ind]

            # if control_indices is None:
            # print("visited ", visited)
            # print("visited transformed ", visited_transformed)
            # print("origin index ", origin_ind)
            # print("interaction index ", interaction_ind)
            # print("origin ", origin)
            # print("destination ", destination)
            # print("controls ", control_indices)
            # print("diff indices ", diff_inds)
            for ind in control_indices:
                if visited_transformed[origin_ind][ind] == 0:
                    qc.x(ind)

            # rz_angle = 2 * segment.phase_time
            # rx_angle = 2 * segment.amplitude_time
            # if origin[interaction_ind] == 1:
            #     rz_angle *= -1

            # if not control_indices:
            #     if rz_angle != 0:
            #         rz_gate = RZGate(rz_angle)
            #         qc.append(rz_gate, control_indices + [interaction_ind])
            #     if rx_angle != 0:
            #         rx_gate = RXGate(rx_angle)
            #         qc.append(rx_gate, control_indices + [interaction_ind])
            # else:
            #     gate_definition=np.array([[np.exp(-1j*rz_angle/2)*np.cos(rx_angle/2),-1j*np.exp(1j*rz_angle/2)*np.sin(rx_angle/2)],
            #                                 [-1j*np.exp(-1j*rz_angle/2)*np.sin(rx_angle/2), np.exp(1j*rz_angle/2)*np.cos(rx_angle/2)]])
            #     # gate_definition = UGate(rx_angle, -(rz_angle/2+3*np.pi/2), -rz_angle/2+3*np.pi/2, label="U").to_matrix()
            #     Ldmcu.ldmcu(qc, gate_definition, control_indices, interaction_ind)

            rz_angle = 2 * segment.phase_time
            # print("rz angle ", rz_angle)
            if origin[interaction_ind] == 1:
                rz_angle *= -1
            if rz_angle != 0:
                rz_gate = RZGate(rz_angle)
                if len(control_indices) > 0:
                    rz_gate = rz_gate.control(len(control_indices))
                qc.append(rz_gate, control_indices + [interaction_ind])

            rx_angle = 2 * segment.amplitude_time
            # print("rx angle ", rx_angle)
            if rx_angle != 0:
                rx_gate = RXGate(rx_angle)
                if len(control_indices) > 0:
                    rx_gate = rx_gate.control(len(control_indices))
                qc.append(rx_gate, control_indices + [interaction_ind])
                # visited.append(destination)

            for ind in control_indices:
                if visited_transformed[origin_ind][ind] == 0:
                    qc.x(ind)

            # update the set of visited
            for ind in diff_inds[1:]:
                destination_transformed=[destination]
                PathConverter.update_visited(destination_transformed, interaction_ind, ind)
                destination=destination_transformed[0]
                # print("updating destination and path")
                # update the target state.
                for segment in path_mutable:
                    z1=[[int(char) for char in segment.labels[0]]]
                    z2=[[int(char) for char in segment.labels[1]]]
                    # print(f"initial {z1} {z2}")
                    PathConverter.update_visited(z1, interaction_ind, ind)
                    PathConverter.update_visited(z2, interaction_ind, ind)
                    # print(f"updated {z1} {z2}")
                    segment.labels[0]="".join(list(map(str,z1[0])))
                    segment.labels[1]="".join(list(map(str,z2[0])))
                    # print(f"attached {segment.labels[0]} {segment.labels[1]}")
            if rx_angle % 2*np.pi !=0:
                # print("appending destination")
                visited_transformed.append(destination)
                # print("visited transformed: ", visited_transformed)
            # print("updating visited.")
            visited=[list(x) for x in set(tuple(x) for x in visited_transformed)]
            # print("visited transformed: ", visited_transformed)
                # for ind in reversed(diff_inds[1:]):
                #     qc.cx(interaction_ind, ind)      
                # visited.append(destination)      

            
                # circ_end.barrier()

            for ind in diff_inds[1:]:
                circ_end.cx(interaction_ind, ind)

            # for ind in control_indices:
            #     if visited_transformed[origin_ind][ind] == 0:
            #         circ_end.x(ind)            

            if add_barriers:
                qc.barrier()

        # print(qc)
        # qc.barrier()
        # qc.barrier()
        qc=qc.compose(Clifford(circ_end.inverse()).to_circuit())
        # print(circ_end)
        # print(qc)

        if remove_leading_cx:
            qc = PathConverter.remove_leading_cx(qc)

        return qc
    

    @staticmethod
    def convert_path_to_circuit_pframes_backwards_leafsm(path: list[PathSegment], reduce_controls: bool = True, remove_leading_cx: bool = True, add_barriers: bool = False) -> QuantumCircuit:
        """
        Converts quantum walks to qiskit circuit.
        :param path: List of path segments, describing the state preparation path.
        :param reduce_controls: True to search for minimally necessary state of controls. False to use all n-1 controls (for debug purposes).
        :param remove_leading_cx: True to remove leading CX gates whose controls are never satisfied.
        :param add_barriers: True to insert barriers between path segments.
        :return: Implementing circuit.
        """

        qc = QuantumCircuit(len(path[0].labels[0]))
        
        all_basis_sts=[elem for seg in path for elem in seg.labels]
        # print("all basis sts ", all_basis_sts)
        visited = [[int(char) for char in st] for st in list(dict.fromkeys(all_basis_sts))]
        path=path[::-1]
        # print("reversed path ", path)
        # print("#########################")
        # print("unique basis sts ", visited)
        path_mutable=deepcopy(path)
        for idx, _ in enumerate(path):
            segment=path_mutable[idx]
            # print(path_mutable)
            origin = [int(char) for char in segment.labels[0]]
            destination = [int(char) for char in segment.labels[1]]
            diff_inds = list(np.where(np.array(origin) != np.array(destination))[0])
            diff_var_dummy=diff_inds
            # print("diffs: ", diff_var_dummy)

            # print("initial visited: ", visited)
            # print("destination: ", destination)
            # print("origin ", origin)
            visited=[elem for elem in visited if elem!=destination]
            visited_transformed = copy.deepcopy(visited)
            # visited_transformed = [elem for elem in visited_transformed if elem!=destination]
            # print("popped destination transformed: ", visited_transformed)
            # print("z1 z2 diffs ", diff_inds)
            # print("visited ", visited)
            if segment.interaction_index is not None:
                interaction_ind=segment.interaction_index
            elif len(diff_inds)==1:
                interaction_ind=diff_inds[0]
            else:
                interaction_ind=PathConverter.get_good_interaction_idx(diff_inds, visited, visited.index(origin), segment.labels[1])
            # interaction_ind = diff_inds[0]
            # print("diff inds ", diff_inds)
            # print("interaction idx ", interaction_ind)
            diff_inds.remove(interaction_ind)
            for ind in diff_inds:
            # for ind in diff_inds[1:]:
                qc.cx(interaction_ind, ind)
                PathConverter.update_visited(visited_transformed, interaction_ind, ind)

            origin_ind = visited.index(origin)
            # print("origin index ", origin_ind)
            if reduce_controls:
                control_indices = PathConverter.find_min_control_set(visited_transformed, origin_ind, interaction_ind)
            else:
                control_indices = [ind for ind in range(len(origin)) if ind != interaction_ind]

            # if control_indices is None:
            # print("visited ", visited)
            # print("visited transformed ", visited_transformed)
            # print("interaction index ", interaction_ind)
            # print("destination ", destination)
            # print("controls ", control_indices)
            # print("diff indices ", diff_inds)
            for ind in control_indices:
                if visited_transformed[origin_ind][ind] == 0:
                    qc.x(ind)

           
            if isinstance(segment, PathSegment):
                rz_angle = -2 * segment.phase_time
                rx_angle = -2 * segment.amplitude_time
                if origin[interaction_ind] == 1:
                    rz_angle *= -1

                if not control_indices:
                    if rx_angle != 0:
                        rx_gate = RXGate(rx_angle)
                        qc.append(rx_gate, control_indices + [interaction_ind])
                    if rz_angle != 0:
                        rz_gate = RZGate(rz_angle)
                        qc.append(rz_gate, control_indices + [interaction_ind])
                else:
                    # gate_definition=np.array([[np.exp(-1j*rz_angle/2)*np.cos(rx_angle/2),-1j*np.exp(1j*rz_angle/2)*np.sin(rx_angle/2)],
                    #                             [-1j*np.exp(-1j*rz_angle/2)*np.sin(rx_angle/2), np.exp(1j*rz_angle/2)*np.cos(rx_angle/2)]])
                    gate_definition=np.array([[np.exp(-1j*rz_angle/2)*np.cos(rx_angle/2),-1j*np.exp(-1j*rz_angle/2)*np.sin(rx_angle/2)],
                                            [-1j*np.exp(1j*rz_angle/2)*np.sin(rx_angle/2), np.exp(1j*rz_angle/2)*np.cos(rx_angle/2)]])
                    Ldmcu.ldmcu(qc, gate_definition, control_indices, interaction_ind)

                    # gate_definition=np.array([[np.exp(-1j*rz_angle/2)*np.cos(rx_angle/2),-1j*np.exp(-1j*rz_angle/2)*np.sin(rx_angle/2)],
                    #                         [-1j*np.exp(1j*rz_angle/2)*np.sin(rx_angle/2), np.exp(1j*rz_angle/2)*np.cos(rx_angle/2)]])
                    # gate_definition=UnitaryGate(gate_definition).control(len(control_indices))
                    # qc.append(gate_definition, control_indices+ [interaction_ind])

                # rx_angle = 2 * segment.amplitude_time
                # rx_angle = -rx_angle #do the opposite since we will be inverting
                # # print("rx angle ", rx_angle)
                # if rx_angle != 0:
                #     rx_gate = RXGate(rx_angle)
                #     if len(control_indices) > 0:
                #         rx_gate = rx_gate.control(len(control_indices))
                #     qc.append(rx_gate, control_indices + [interaction_ind])
                #     # visited.append(destination)

                # rz_angle = 2 * segment.phase_time
                # rz_angle = -rz_angle #do the opposite since we will be inverting
                # # print("rz angle ", rz_angle)
                # if origin[interaction_ind] == 1:
                #     rz_angle *= -1
                # if rz_angle != 0:
                #     rz_gate = RZGate(rz_angle)
                #     if len(control_indices) > 0:
                #         rz_gate = rz_gate.control(len(control_indices))
                #     qc.append(rz_gate, control_indices + [interaction_ind])
            else: #LeafPathSegment. Everything should be backwards.
                rz_angle1=-segment.phase_time1
                rz_angle2=-segment.phase_time2
                rx_angle=-segment.amplitude_time
                if not control_indices:
                    if origin[interaction_ind]==1:
                        rz_angle2=-1*rz_angle2
                    else:
                        rz_angle1=-1*rz_angle1
                    phase_gate1=PhaseGate(rz_angle1)
                    phase_gate2=PhaseGate(rz_angle2)
                    # phase_gate1=UnitaryGate([[1, 0], [0, np.exp(1j*rz_angle1)]])
                    # phase_gate2=UnitaryGate([[np.exp(1j*rz_angle2), 0], [0, 1]])
                    rx_gate = RXGate(2*rx_angle)
                    qc.append(phase_gate2, [interaction_ind])
                    qc.append(rx_gate, [interaction_ind])
                    qc.append(phase_gate1, [interaction_ind])
                else:
                    if origin[interaction_ind] == 0:
                        gate_definition=np.array([[np.exp(1j*rz_angle1)*np.cos(rx_angle), -1j*np.exp(1j*(rz_angle1+rz_angle2))*np.sin(rx_angle)],
                                            [-1j*np.sin(rx_angle), np.exp(1j*rz_angle2)*np.cos(rx_angle)]])
                    else:
                        gate_definition=np.array([[np.exp(1j*rz_angle2)*np.cos(rx_angle), -1j*np.sin(rx_angle)],
                                            [-1j*np.exp(1j*(rz_angle1+rz_angle2))*np.sin(rx_angle), np.exp(1j*rz_angle1)*np.cos(rx_angle)]])
                    # gate_definition = UGate(rx_angle, -(rz_angle/2+3*np.pi/2), -rz_angle/2+3*np.pi/2, label="U").to_matrix()
                    Ldmcu.ldmcu(qc, gate_definition, control_indices, interaction_ind)
            # print(qc)
            
            for ind in control_indices:
                if visited_transformed[origin_ind][ind] == 0:
                    qc.x(ind)

            # update the set of visited
            for ind in diff_inds:
            # for ind in diff_inds[1:]:
                destination_transformed=[destination]
                PathConverter.update_visited(destination_transformed, interaction_ind, ind)
                destination=destination_transformed[0]
                # print("updating destination and path")
                # update the target state.
                for segment in path_mutable:
                    z1=[[int(char) for char in segment.labels[0]]]
                    z2=[[int(char) for char in segment.labels[1]]]
                    # print(f"initial {z1} {z2}")
                    PathConverter.update_visited(z1, interaction_ind, ind)
                    PathConverter.update_visited(z2, interaction_ind, ind)
                    # print(f"updated {z1} {z2}")
                    segment.labels[0]="".join(list(map(str,z1[0])))
                    segment.labels[1]="".join(list(map(str,z2[0])))
                    # print(f"attached {segment.labels[0]} {segment.labels[1]}")

            # print("updating visited.")
            visited=[list(x) for x in set(tuple(x) for x in visited_transformed)]
            # print("visited transformed: ", visited_transformed)
                # for ind in reversed(diff_inds[1:]):
                #     qc.cx(interaction_ind, ind)      
                # visited.append(destination)      

            
                # circ_end.barrier()

            # for ind in diff_inds[1:]:
            #     circ_end.cx(interaction_ind, ind)

            # for ind in control_indices:
            #     if visited_transformed[origin_ind][ind] == 0:
            #         circ_end.x(ind)            

            if add_barriers:
                qc.barrier()
        # print("visited ", visited)
        starting_state = "".join([str(char) for char in visited[0]])
        # circ_end=QuantumCircuit(len(starting_state))
        indices_1 = [ind for ind, elem in enumerate(starting_state) if elem == "1"]
        for ind in indices_1:
            qc.x(ind)
        if add_barriers:
            qc.barrier()

        # print("circ end ", circ_end)
        # qc=qc.compose(Clifford(circ_end.inverse()).to_circuit())
        qc=qc.inverse()
        # print(circ_end)
        # print(qc)

        if remove_leading_cx:
            qc = PathConverter.remove_leading_cx(qc)

        return qc
    
    # @staticmethod
    # def convert_path_to_circuit_pframes_backwards(path: list[PathSegment], reduce_controls: bool = True, remove_leading_cx: bool = True, add_barriers: bool = False) -> QuantumCircuit:
    #     """
    #     Converts quantum walks to qiskit circuit.
    #     :param path: List of path segments, describing the state preparation path.
    #     :param reduce_controls: True to search for minimally necessary state of controls. False to use all n-1 controls (for debug purposes).
    #     :param remove_leading_cx: True to remove leading CX gates whose controls are never satisfied.
    #     :param add_barriers: True to insert barriers between path segments.
    #     :return: Implementing circuit.
    #     """
    #     # circuit for the ending rotations.
    #     # print("original ", path)

    #     # visited=[]
    #     # for idx, segment in enumerate(path):
    #     #     if idx!=0:
    #     #         visited.append(segment.labels[1])
    #     #     else:
    #     #         visited.append(segment.labels[0])
    #     #         visited.append(segment.labels[1])
    #     # visited=list(dict.fromkeys(visited)) 
    #     qc = QuantumCircuit(len(path[0].labels[0]))
        
    #     all_basis_sts=[elem for seg in path for elem in seg.labels]
    #     # print("all basis sts ", all_basis_sts)
    #     visited = [[int(char) for char in st] for st in list(dict.fromkeys(all_basis_sts))]
    #     path=path[::-1]
    #     # print("reversed path ", path)
    #     # print("#########################")
    #     # print("unique basis sts ", visited)
    #     path_mutable=deepcopy(path)
    #     for idx, _ in enumerate(path):
    #         segment=path_mutable[idx]
    #         # print(path_mutable)
    #         origin = [int(char) for char in segment.labels[0]]
    #         destination = [int(char) for char in segment.labels[1]]
    #         diff_inds = np.where(np.array(origin) != np.array(destination))[0]
    #         diff_var_dummy=diff_inds.tolist()
    #         interaction_ind = diff_inds[0]
    #         # print("diffs: ", diff_var_dummy)

    #         # print("initial visited: ", visited)
    #         # print("destination: ", destination)
    #         # print("origin ", origin)
    #         visited=[elem for elem in visited if elem!=destination]
    #         visited_transformed = copy.deepcopy(visited)
    #         # visited_transformed = [elem for elem in visited_transformed if elem!=destination]
    #         # print("popped destination transformed: ", visited_transformed)
    #         # print()
            

    #         for ind in diff_inds[1:]:
    #             qc.cx(interaction_ind, ind)
    #             PathConverter.update_visited(visited_transformed, interaction_ind, ind)

    #         origin_ind = visited.index(origin)
    #         # print("origin index ", origin_ind)
    #         if reduce_controls:
    #             control_indices = PathConverter.find_min_control_set(visited_transformed, origin_ind, interaction_ind)
    #         else:
    #             control_indices = [ind for ind in range(len(origin)) if ind != interaction_ind]

    #         # if control_indices is None:
    #         # print("visited ", visited)
    #         # print("visited transformed ", visited_transformed)
    #         # print("interaction index ", interaction_ind)
    #         # print("destination ", destination)
    #         # print("controls ", control_indices)
    #         # print("diff indices ", diff_inds)
    #         for ind in control_indices:
    #             if visited_transformed[origin_ind][ind] == 0:
    #                 qc.x(ind)

    #         # rz_angle = 2 * segment.phase_time
    #         # rx_angle = 2 * segment.amplitude_time
    #         # if origin[interaction_ind] == 1:
    #         #     rz_angle *= -1

    #         # if not control_indices:
    #         #     if rz_angle != 0:
    #         #         rz_gate = RZGate(rz_angle)
    #         #         qc.append(rz_gate, control_indices + [interaction_ind])
    #         #     if rx_angle != 0:
    #         #         rx_gate = RXGate(rx_angle)
    #         #         qc.append(rx_gate, control_indices + [interaction_ind])
    #         # else:
    #         #     gate_definition=np.array([[np.exp(-1j*rz_angle/2)*np.cos(rx_angle/2),-1j*np.exp(1j*rz_angle/2)*np.sin(rx_angle/2)],
    #         #                                 [-1j*np.exp(-1j*rz_angle/2)*np.sin(rx_angle/2), np.exp(1j*rz_angle/2)*np.cos(rx_angle/2)]])
    #         #     # gate_definition = UGate(rx_angle, -(rz_angle/2+3*np.pi/2), -rz_angle/2+3*np.pi/2, label="U").to_matrix()
    #         #     Ldmcu.ldmcu(qc, gate_definition, control_indices, interaction_ind)
    #         rx_angle = 2 * segment.amplitude_time
    #         rx_angle = -rx_angle #do the opposite since we will be inverting
    #         # print("rx angle ", rx_angle)
    #         if rx_angle != 0:
    #             rx_gate = RXGate(rx_angle)
    #             if len(control_indices) > 0:
    #                 rx_gate = rx_gate.control(len(control_indices))
    #             qc.append(rx_gate, control_indices + [interaction_ind])
    #             # visited.append(destination)


    #         rz_angle = 2 * segment.phase_time
    #         rz_angle = -rz_angle #do the opposite since we will be inverting
    #         # print("rz angle ", rz_angle)
    #         if origin[interaction_ind] == 1:
    #             rz_angle *= -1
    #         if rz_angle != 0:
    #             rz_gate = RZGate(rz_angle)
    #             if len(control_indices) > 0:
    #                 rz_gate = rz_gate.control(len(control_indices))
    #             qc.append(rz_gate, control_indices + [interaction_ind])
    #         # print(qc)
            
    #         for ind in control_indices:
    #             if visited_transformed[origin_ind][ind] == 0:
    #                 qc.x(ind)

    #         # update the set of visited
    #         for ind in diff_inds[1:]:
    #             destination_transformed=[destination]
    #             PathConverter.update_visited(destination_transformed, interaction_ind, ind)
    #             destination=destination_transformed[0]
    #             # print("updating destination and path")
    #             # update the target state.
    #             for segment in path_mutable:
    #                 z1=[[int(char) for char in segment.labels[0]]]
    #                 z2=[[int(char) for char in segment.labels[1]]]
    #                 # print(f"initial {z1} {z2}")
    #                 PathConverter.update_visited(z1, interaction_ind, ind)
    #                 PathConverter.update_visited(z2, interaction_ind, ind)
    #                 # print(f"updated {z1} {z2}")
    #                 segment.labels[0]="".join(list(map(str,z1[0])))
    #                 segment.labels[1]="".join(list(map(str,z2[0])))
    #                 # print(f"attached {segment.labels[0]} {segment.labels[1]}")

    #         # print("updating visited.")
    #         visited=[list(x) for x in set(tuple(x) for x in visited_transformed)]
    #         # print("visited transformed: ", visited_transformed)
    #             # for ind in reversed(diff_inds[1:]):
    #             #     qc.cx(interaction_ind, ind)      
    #             # visited.append(destination)      

            
    #             # circ_end.barrier()

    #         # for ind in diff_inds[1:]:
    #         #     circ_end.cx(interaction_ind, ind)

    #         # for ind in control_indices:
    #         #     if visited_transformed[origin_ind][ind] == 0:
    #         #         circ_end.x(ind)            

    #         if add_barriers:
    #             qc.barrier()
    #     # print("visited ", visited)
    #     starting_state = "".join([str(char) for char in visited[0]])
    #     # circ_end=QuantumCircuit(len(starting_state))
    #     indices_1 = [ind for ind, elem in enumerate(starting_state) if elem == "1"]
    #     for ind in indices_1:
    #         qc.x(ind)
    #     if add_barriers:
    #         qc.barrier()

    #     # print(qc)
    #     # qc.barrier()
    #     # qc.barrier()
    #     # print("circ end ", circ_end)
    #     # qc=qc.compose(Clifford(circ_end.inverse()).to_circuit())
    #     qc=qc.inverse()
    #     # print(circ_end)
    #     # print(qc)

    #     if remove_leading_cx:
    #         qc = PathConverter.remove_leading_cx(qc)

    #     return qc
