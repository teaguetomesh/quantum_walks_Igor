""" Qiskit-specific utilities. """
import qiskit.converters as conv
from qiskit import QuantumCircuit


def remove_leading_cx_gates(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Removes leading CX gates whose controls are always false.
    :param qc: Input quantum circuit.
    :return: Optimized quantum circuit where the leading CX gates are removed where possible.
    """
    dag = conv.circuit_to_dag(qc)
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
    return conv.dag_to_circuit(dag)
