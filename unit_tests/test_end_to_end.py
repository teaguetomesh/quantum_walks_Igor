import numpy as np
import sys
sys.path.append('../')
from src.quantum_walks import PathFinderLinear, PathFinderSHP, PathFinderMST
from src.state_preparation import CircuitGeneratorPath
from qiskit import transpile
from qiskit_aer import AerSimulator

basis_gates = ["rx", "ry", "rz", "h", "cx"]

def execute_circuit(circuit):
    """
    Executes transpiled circuit and returns the statevector output.
    :param circuit: Transpiled qiskit circuit to execute.
    :return: State vector at the end of the circuit.
    """
    sim = AerSimulator()
    circuit.save_statevector(label="end")
    result = sim.run(circuit).result()
    state = result.data(0)["end"].data
    return state


def get_state_vector(target_state: dict[str, complex]):
    """
    Converts a state described by a dictionary into the state vector representation.
    :param target_state: The state to convert.
    :return: 2^n 1D array corresponding to the target state vector in big endian notation.
    """
    num_qubits = len(list(target_state.keys())[0])
    state_vector = np.zeros(2 ** num_qubits, dtype=complex)
    for basis, _ in target_state.items():
        basis_ind = int(basis[::-1], 2)
        state_vector[basis_ind] = target_state[basis]
    return state_vector


def get_fidelity(state1, state2) -> float:
    """
    Get the fidelity between the two states.
    :param state1: First state.
    :param state2: Second state.
    :return: Fidelity of the two states.
    """
    return abs(state1.conjugate().T @ state2) ** 2

def exec_state(target_state, path_finder, reduce_controls, optimization_level=3):
    path = path_finder.get_path(target_state)
    circuit = CircuitGeneratorPath.convert_path_to_circuit(path, reduce_controls)
    circuit = transpile(circuit, basis_gates=basis_gates, optimization_level=optimization_level)
    output_state_vector = execute_circuit(circuit)
    target_state_vector = get_state_vector(target_state)
    fidelity = get_fidelity(output_state_vector, target_state_vector)
    return fidelity

def test_general():
    # end to end test.
    random_state1={"00": 1/np.sqrt(2), "11": 1/np.sqrt(2)}
    random_state2={"00": 1/np.sqrt(2)*1j, "11": (1/2+1j/2)}
    random_state3={'01': (0.3225040489858759-0.2762653986846726j), '10': (0.7750180320299219+0.4679910446854529j)}
    random_state4={'0010': (-0.12332991320557402-0.25602126418276294j), '0100': (0.352282620478757-0.6407635099031103j), '0101': (0.3033725321232708-0.41973798774884435j), '1011': (0.2155784315914548+0.26433500091071327j)}
    random_state5={'001': (-0.3119819798683826+0.37401620629642257j), '010': (-0.685265269401355-0.031623713274994567j), '011': (0.5271799015355274+0.11946515974549463j)}
    random_state6={'01001': (0.6373067864536242+0.2102852452102945j), '10001': (-0.13435970242231168-0.02106819224266982j), '10011': (0.47718896650513604-0.273517908021147j), '10100': (0.18388222918848504+0.40521780224826276j), '11011': (-0.10523458379521804+0.13969238479000462j)}
    random_state7={'00000': (0.6373067864536242+0.2102852452102945j), '10001': (-0.13435970242231168-0.02106819224266982j), '10011': (0.47718896650513604-0.273517908021147j), '10100': (0.18388222918848504+0.40521780224826276j), '11011': (-0.10523458379521804+0.13969238479000462j)}
    random_state8={'00000': (-0.2333688706503088-0.2611158273618947j), '00010': (0.38219031250053825-0.3958211401430709j), '00111': (-0.2954722046458634+0.046088714525425005j), '01001': (-0.2261380434952844-0.059944014605614854j), '11010': (-0.6166849897887388+0.22395002880229722j)}
    random_state9={'00010': (0.5351456988039678+0.21425719569475046j), '01111': (-0.29331472616356125-0.3415601360670022j), '10001': (-0.330640821053152+0.12406108459450757j), '10101': (-0.2200228786923234-0.3520610711026033j), '11011': (0.3383040168098163+0.2312896496366119j)}
    state_list=[random_state1,random_state2,random_state3,random_state4,
            random_state5,random_state6,random_state7,random_state8, random_state9]
    fidelity_tol = 1e-8

    for target_state in state_list:
        #SHP
        reduce_controls = True
        fidelity = exec_state(target_state, PathFinderSHP(), reduce_controls)
        assert abs(1 - fidelity) < fidelity_tol, f"SHP + cntrl reduction failed to prepare the state: {target_state}. Fidelity: {fidelity}"
        #Linear
        fidelity = exec_state(target_state, PathFinderLinear(), reduce_controls)
        assert abs(1 - fidelity) < fidelity_tol, f"Linear + cntrl reduction failed to prepare the state: {target_state}. Fidelity: {fidelity}"
        #MST
        fidelity = exec_state(target_state, PathFinderMST(), reduce_controls)
        assert abs(1 - fidelity) < fidelity_tol, f"MST + cntrl reduction failed to prepare the state: {target_state}. Fidelity: {fidelity}"
        #SHP
        reduce_controls = False
        fidelity = exec_state(target_state, PathFinderSHP(), reduce_controls)
        assert abs(1 - fidelity) < fidelity_tol, f"SHP failed to prepare the state: {target_state}. Fidelity: {fidelity}"
        #Linear
        fidelity = exec_state(target_state, PathFinderLinear(), reduce_controls)
        assert abs(1 - fidelity) < fidelity_tol, f"Linear failed to prepare the state: {target_state}. Fidelity: {fidelity}"
        #MST
        fidelity = exec_state(target_state, PathFinderMST(), reduce_controls)
        assert abs(1 - fidelity) < fidelity_tol, f"MST failed to prepare the state: {target_state}. Fidelity: {fidelity}"