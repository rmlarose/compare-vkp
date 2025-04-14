from typing import Tuple, List
from dataclasses import dataclass
import numpy as np
import scipy.linalg as la
import cirq
import openfermion as of
import qiskit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.qasm2 import dumps
import convert

def load_water_hamiltonian() -> of.QubitOperator:
    """Load the water molecule Hamiltonian from its pubchem description, then convert to QubitOperator."""

    geom = of.chem.geometry_from_pubchem("water")
    basis = "sto-3g"
    multiplicity = 1
    charge = 0
    molecule = of.chem.MolecularData(geom, basis, multiplicity, charge)
    molecule.load()
    molecular_hamiltonian = molecule.get_molecular_hamiltonian()
    fermi_hamiltonian = of.transforms.get_fermion_operator(molecular_hamiltonian)
    return of.transforms.jordan_wigner(fermi_hamiltonian)


def load_hubbard_hamiltonian() -> of.QubitOperator:
    """Load a 2D Fermi-Hubbard Hamiltonian."""

    ham_fermi = of.hamiltonians.fermi_hubbard(3, 3, 1.0, 2.0, spinless=True)
    ham: of.QubitOperator = of.transforms.jordan_wigner(ham_fermi)
    return ham


def hf_ref_circuit(nqubits: int, noccupied: int) -> cirq.Circuit:
    """Get a circuit that prepares the state |11..10...0>.

    Arguments:
    nqubits: The number of qubits in the circuit.
    noccupied: The number of occupied spin-orbitals (qubits)."""

    ckt = cirq.Circuit()
    qs = cirq.LineQubit.range(nqubits)
    for i, q in enumerate(qs):
        if i < noccupied:
            ckt.append(cirq.X(q))
        else:
            ckt.append(cirq.I(q))
    return ckt


def neel_state_circuit(nqubits: int) -> cirq.Circuit:
    """Circuit to build a Neel state |101010...> on a number of qubits."""

    qs = cirq.LineQubit.range(nqubits)
    ckt = cirq.Circuit()
    for i, q in enumerate(qs):
        if i % 2 == 0:
            ckt.append(cirq.X(q))
        else:
            ckt.append(cirq.I(q))
    return ckt


def neel_state_circuit_qiskit(nqubits: int) -> cirq.Circuit:
    """Circuit to build a Neel state |101010...> on a number of qubits."""

    ckt = qiskit.QuantumCircuit(nqubits)
    for i in range(nqubits):
        if i % 2 == 0:
            ckt.x(i)
        else:
            ckt.id(i)
    return ckt


def hf_ref_circuit_qiskit(nqubits: int, noccupied: int) -> qiskit.QuantumCircuit:
    """Get a circuit that prepares the state |11..10...0>."""

    ckt = qiskit.QuantumCircuit(nqubits)
    for i in range(nqubits):
        if i < noccupied:
            ckt.x(i)
        else:
            ckt.id(i)
    return ckt


def paulihedral_trotter_circuit(ham: of.QubitOperator, dt: float) -> qiskit.QuantumCircuit:
    """Get a circuit for time evolution with the Hamiltonian ham
    using a time step of length dt."""

    assert dt > 0.0

    psum_ham = of.transforms.qubit_operator_to_pauli_sum(ham)
    qiskit_ham = convert.cirq_pauli_sum_to_qiskit_pauli_op(psum_ham)

    circuit = qiskit.QuantumCircuit(qiskit_ham.num_qubits)
    gate = PauliEvolutionGate(qiskit_ham)
    circuit.append(gate, range(qiskit_ham.num_qubits))
    return circuit


def _get_state_cirq(
    state_prep_circuit: cirq.Circuit, evolution_circuit: cirq.Circuit, d: int
) -> np.ndarray:
    """Get the state vector corresponding to (U_evolution)^d U_prep |0>."""

    total_circuit = state_prep_circuit.copy()
    for _ in range(d):
        total_circuit += evolution_circuit
    sim = cirq.Simulator()
    sim_result = sim.simulate(total_circuit)
    return sim_result.final_state_vector


def _get_state_qiskit(
    state_prep_circuit: qiskit.QuantumCircuit, evolution_circuit: qiskit.QuantumCircuit, d: int
) -> np.ndarray:
    """Get the state vector corresponding to (U_evolution)^d U_prep |0>."""

    total_circuit = state_prep_circuit.copy()
    for _ in range(d):
        total_circuit = total_circuit.compose(evolution_circuit)
    return qiskit.quantum_info.Statevector(total_circuit).data


def _evolve_state_cirq(reference_state: np.ndarray, evolution_circuit: cirq.Circuit, d: int) -> np.ndarray:
    """Evolve a give reference state by d applications of an evolution circuit."""

    if d == 0:
        return reference_state
    else:
        total_circuit = cirq.Circuit()
        for _ in range(d):
            total_circuit += evolution_circuit
        sim = cirq.Simulator()
        sim_result = sim.simulate(total_circuit, initial_state=reference_state)
        return sim_result.final_state_vector


def _evolve_state_qiskit(
        reference_state: np.ndarray, evolution_circuit: qiskit.QuantumCircuit, d: int
) -> np.ndarray:
    """Get the state vector corresponding to (U_evolution)^d U_prep |0>."""

    # TODO This function needs to set the initial state of the qubits.
    total_circuit = qiskit.QuantumCircuit()
    for _ in range(d):
        total_circuit = total_circuit.compose(evolution_circuit)
    return qiskit.quantum_info.Statevector(total_circuit).data


def subspace_matrices_from_ref_state(
        ham: of.QubitOperator,
        reference_state: np.ndarray,
        evolution_circuit: qiskit.QuantumCircuit | cirq.Circuit,
        d: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the subspace matrices from a given reference state."""

    ham_cirq = of.transforms.qubit_operator_to_pauli_sum(ham)
    ham_matrix = ham_cirq.matrix()

    h = np.zeros((d, d), dtype=complex)
    s = np.zeros((d, d), dtype=complex)
    print("Computing subspace matrices.")
    # Compute matrix element like <phi| H U^k |phi> etc.
    overlaps = [] # <phi| U^k |phi>
    mat_elems = []
    for k in range(d):
        print(f"On {k} / {d}")
        # Compute the state vectors for U^k |ref> and (U^dag)^k |ref.
        if isinstance(evolution_circuit, cirq.Circuit):
            evolved_state = _evolve_state_cirq(reference_state, evolution_circuit, k)
        else:
            evolved_state = _evolve_state_qiskit(reference_state, evolution_circuit, k)
        overlaps.append(np.vdot(reference_state, evolved_state))
        mat_elems.append(np.vdot(reference_state, ham_matrix @ evolved_state))
    # Fill the matrix.
    for i in range(d): # Loop over rows.
        for j in range(d):
            if i == j:
                h[i, i] = mat_elems[0]
                s[i, i] = overlaps[0]
            elif i > j:
                h[i, j] = mat_elems[i - j]
                s[i, j] = overlaps[i - j]
            else: # i < j
                h[i, j] = mat_elems[j - i].conjugate()
                s[i, j] = overlaps[j - i].conjugate()
    # breakpoint()
    # assert la.ishermitian(h, rtol=1e-12)
    # assert la.ishermitian(s, rtol=1e-12)
    return h, s


def subspace_matrices(
    ham: of.QubitOperator,
    state_prep_circuit: qiskit.QuantumCircuit | cirq.Circuit,
    evolution_circuit: qiskit.QuantumCircuit | cirq.Circuit,
    d: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the S and H matrices for the subsapce generated by
    realt time evolution for time tau, up to subspace dimension d.
    The subspace is generated by the unitary U = exp(-i H tau),
    wich we approximate by N steps of a first-order Trotter
    approximation.
    
    Arguments:
    ham - Hamiltonian of the system.
    state_prep_circuit - Circuit to prepare the reference state.
    evolution_circuit - time evolution circuit U to generate subsapce.
    tau - time for the evolution operator (not the trotter step, the time for U.)
    steps - Number of steps to make U.
    d - Subpsace dimension
    
    Returns:
    H - Hamiltonian projected into subspace.
    S - overlap matrix of subspace."""

    # Store the reference state to compute overlaps.
    if isinstance(state_prep_circuit, qiskit.QuantumCircuit):
        reference_state = qiskit.quantum_info.Statevector(state_prep_circuit).data
    else:
        sim = cirq.Simulator()
        sim_result = sim.simulate(state_prep_circuit)
        reference_state = sim_result.final_state_vector
    return subspace_matrices_from_ref_state(ham, reference_state, evolution_circuit, d)



def energy_vs_d(h, s, step:int = 1) -> Tuple[List[int], List[float]]:
    """Get energy vs. subspace dimension.
    If H and S are of dimension D, we can get the energy estimate
    for d < D by taking the upper left d x d blocks of H and S."""

    assert h.shape == s.shape
    assert h.shape[0] == h.shape[1]
    ds: List[int] = []
    energies: List[float] = []
    for d in range(1, h.shape[0], step):
        ds.append(d)
        lam, v = la.eig(h[:d, :d], s[:d, :d])
        energies.append(np.min(lam))
    return ds, energies


@dataclass 
class RTESubspaceResult:
    """Store the results of getting the H and S matrices."""
    tau: float
    nsteps: int
    h: np.ndarray
    s: np.ndarray
    ds: List[int]
    energies: List[float]
