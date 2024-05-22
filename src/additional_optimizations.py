# Additional optimizations

from qiskit.converters import (
    circuit_to_dag,
    dag_to_circuit
)
from qiskit import QuantumCircuit

def remove_leading_cx(qc: QuantumCircuit) -> QuantumCircuit:
    '''removes leading cx gates that are controlled on a ground state.'''
    dag=circuit_to_dag(qc)
    wires = dag.wires
    # go through the ops in each wire and remove cx ops until
    # we run into a non cx operation or a cx with target on the wire.
    for w in wires:
        for node in list(dag.nodes_on_wire(w, only_ops=True)):
            # Check if the node is cnot
            if node.name!="cx":
                break
            elif node.qargs[0]==w: # the control is the current wire so remove
                dag.remove_op_node(node)
            else: #cnot but with target on qubit wire w.
                break
    return dag_to_circuit(dag)