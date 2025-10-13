from typing import Tuple, List, Dict
from dataclasses import dataclass
from mpi4py import MPI
import numpy as np
import scipy.linalg as la
import torch
import cirq
import qsimcirq
import openfermion as of
import qiskit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.qasm2 import dumps
from qiskit_aer import AerSimulator
import quimb
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductState, MatrixProductOperator
import convert
from tensor_network_common import pauli_sum_to_mpo

def xyz_hamiltonian(l: int, h: float, j: np.ndarray) -> of.QubitOperator:
    """Heisenberg XYZ Hamiltonian
    H = -h sum_i sigma^z_i + j_x sigma^x_i sigma^x_{i+1} + ..."""

    assert l > 0
    assert j.size == 3

    ham = of.QubitOperator()
    for i in range(l):
        ham += -h * of.QubitOperator(f"Z{i}")
    for i in range(l):
        if i != l-1:
            ham += j[0] * of.QubitOperator(f"X{i} X{i+1}")
            ham += j[1] * of.QubitOperator(f"Y{i} Y{i+1}")
            ham += j[2] * of.QubitOperator(f"Z{i} Z{i+1}")
    return ham


def load_xyz_hamiltonian() -> of.QubitOperator:
    """Load the Heisenberg Hamiltonian with set parameters."""

    l = 3
    hh = 1.0
    j = np.array([0.1, 0.3, 0.1])
    ham = xyz_hamiltonian(l, hh, j)
    return ham


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


def load_hubbard_hamiltonian(n: int = 4, t: float = 1.0, u: float = 2.0) -> of.QubitOperator:
    """Load a 2D Fermi-Hubbard Hamiltonian."""

    ham_fermi = of.hamiltonians.fermi_hubbard(n, n, t, u, spinless=True)
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


def neel_state_circuit_qiskit(nqubits: int) -> qiskit.QuantumCircuit:
    """Circuit to build a Neel state |101010...> on a number of qubits."""

    ckt = qiskit.QuantumCircuit(nqubits)
    for i in range(nqubits):
        if i % 2 == 0:
            ckt.x(i)
        else:
            ckt.id(i)
    return ckt


def cb_state_circuit_qiskit(nqubits: int, bits: List[bool]) -> qiskit.QuantumCircuit:
    """Circuit that prepares the chosen computational basis state."""

    assert len(bits) == nqubits
    ckt = qiskit.QuantumCircuit(nqubits)
    for i, b in enumerate(bits):
        if b:
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
        # sim = cirq.Simulator()
        sim = qsimcirq.QSimSimulator()
        sim_result = sim.simulate(total_circuit, initial_state=reference_state.astype(np.complex64))
        return sim_result.final_state_vector


def _evolve_state_qiskit(
        reference_state: np.ndarray, evolution_circuit: qiskit.QuantumCircuit, d: int, nq: int
) -> np.ndarray:
    """Get the state vector corresponding to (U_evolution)^d U_prep |0>."""

    total_circuit = qiskit.QuantumCircuit(nq)
    total_circuit.initialize(reference_state)
    for _ in range(d):
        total_circuit = total_circuit.compose(evolution_circuit)
    sim = AerSimulator(method="statevector")
    transpiled_circuit = qiskit.transpile(total_circuit, sim)
    transpiled_circuit.save_state()
    result = sim.run(transpiled_circuit).result()
    sv = result.get_statevector()
    return sv.data


def subspace_matrices_from_ref_state(
    ham: of.QubitOperator,
    reference_state: np.ndarray,
    evolution_circuit: qiskit.QuantumCircuit | cirq.Circuit,
    d: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the subspace matrices from a given reference state."""

    nq = of.utils.count_qubits(ham)
    # ham_cirq = of.transforms.qubit_operator_to_pauli_sum(ham)
    # ham_matrix = ham_cirq.matrix()
    ham_matrix = of.linalg.get_sparse_operator(ham)

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
            evolved_state = _evolve_state_qiskit(reference_state, evolution_circuit, k, nq)
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


def fill_subspace_matrices(
    mat_elems: List[complex], overlaps: List[complex]
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill subspace matrices from the computed matrix elements and overlaps."""

    assert len(mat_elems) == len(overlaps)
    d = len(mat_elems)
    h = np.zeros((d, d), dtype=complex)
    s = np.zeros((d, d), dtype=complex)
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


def tebd_matrix_element_and_overlap(
    ham_mpo: MatrixProductOperator,
    evolution_circuit: qiskit.QuantumCircuit,
    reference_mps: MatrixProductState,
    d: int,
    max_circuit_bond: int,
    backend_callback
) -> Tuple[complex, complex]:
    """Compute <psi|HU^d|psi> and <psi|U^d|psi> using TEBD"""

    if d == 0:
        evolved_mps = reference_mps.copy()
    else:
        # Make a circuit with d repetitions of the evolution circuit.
        nq = evolution_circuit.num_qubits
        total_circuit = qiskit.QuantumCircuit(nq)
        for _ in range(d):
            total_circuit = total_circuit.compose(evolution_circuit)
        # Convert the circuit to quimb format.
        qasm_str = dumps(total_circuit)
        # circuit_quimb = qtn.circuit.Circuit(psi0=reference_mps).from_qasm(qasm_str)
        # circuit_mps = qtn.circuit.CircuitMPS(psi0=reference_mps, max_bond=max_circuit_bond)
        # circuit_mps.apply_gates(circuit_quimb.gates)
        if backend_callback is not None:
            circuit_mps = qtn.circuit.CircuitMPS.from_openqasm2_str(
                qasm_str, psi0=reference_mps, max_bond=max_circuit_bond, progbar=False,
                to_backend=backend_callback
            )
        else:
            circuit_mps = qtn.circuit.CircuitMPS.from_openqasm2_str(
                qasm_str, psi0=reference_mps, max_bond=max_circuit_bond, progbar=False
            )
        evolved_mps = circuit_mps.psi
    # Build tensor networks for <psi| U^d |psi> and <psi| H U^d |psi>
    overlap = reference_mps.H @ evolved_mps
    mat_elem = reference_mps.H @ ham_mpo.apply(evolved_mps)
    return (mat_elem, overlap)


def tebd_subspace_matrices(
    hamiltonian: cirq.PauliSum, evolution_circuit: qiskit.QuantumCircuit,
    ref_state: MatrixProductState,
    d_max: int, max_circuit_bond: int, max_mpo_bond: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Comopute subspace matrices with TEBD."""

    matrix_elements: List[complex] = []
    overlaps: List[complex] = []
    for d in range(d_max):
        print(f"d={d}")
        mat_elem, overlap = tebd_matrix_element_and_overlap(
            hamiltonian, evolution_circuit, ref_state, d,
            max_circuit_bond, max_mpo_bond
        )
        matrix_elements.append(mat_elem)
        overlaps.append(overlap)
    h, s = fill_subspace_matrices(matrix_elements, overlaps)
    return (h, s)


def tebd_subspace_matrices_parallel(
    ham_mpo: MatrixProductOperator, evolution_circuit: qiskit.QuantumCircuit,
    ref_state: MatrixProductState,
    d_max: int, max_circuit_bond: int,
    mpi_comm_rank: int, mpi_comm_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Comopute subspace matrices with TEBD."""

    assert mpi_comm_size == d_max, \
        f"Calculating {d_max} elements, but using {mpi_comm_size} processes."

    # Each process compuate a specific matrix element.
    # Give each process a value of d, the power of the unitary U.
    all_d_vals = np.array(range(d_max))
    comm = MPI.COMM_WORLD
    my_d = comm.scatter(all_d_vals, root=0)
    mat_elem, overlap = tebd_matrix_element_and_overlap(
        ham_mpo, evolution_circuit, ref_state, my_d,
        max_circuit_bond
    )
    print(mpi_comm_rank, mat_elem, overlap)
    # Build a list of all the calculated matrix elements.
    # matrix_elements = np.zeros(d_max, dtype=complex)
    # overlaps = np.zeros(d_max, dtype=complex)
    # matrix_elements = np.array(comm.gather(mat_elem, root=0))
    # overlaps = np.array(comm.gather(overlap, root=0))
    # comm.Bcast([matrix_elements, MPI.COMPLEX], root=0)
    # comm.Bcast([overlaps, MPI.COMPLEX], root=0)
    matrix_elements = comm.allgather(mat_elem)
    overlaps = comm.allgather(overlap)
    print("All matrix elements sent to nodes.")
    h, s = fill_subspace_matrices(matrix_elements, overlaps)
    return (h, s)


def tebd_states_to_scratch(
    ev_circuit: qiskit.QuantumCircuit,
    ref_state: MatrixProductState, max_bond: int, d: int,
    scratch_dir: str, backend_callback
) -> Dict[int, str]:
    """Do successive steps of TEBD with the same circuit, storing the intermediate MPS's
    in a scratch directory."""

    qasm_str = dumps(ev_circuit)
    d_path_dict: Dict[int, str] = {}
    evolved_mps = ref_state.copy()
    for i in range(d):
        fname = f"{scratch_dir}/state_{i}.dump"
        quimb.save_to_disk(evolved_mps, fname)
        d_path_dict[i] = fname
        if i != d - 1:
            if backend_callback is not None:
                circuit_mps = qtn.circuit.CircuitMPS.from_openqasm2_str(
                    qasm_str, psi0=evolved_mps, max_bond=max_bond, progbar=False,
                    to_backend=backend_callback
                )
            else:
                circuit_mps = qtn.circuit.CircuitMPS.from_openqasm2_str(
                    qasm_str, psi0=evolved_mps, max_bond=max_bond, progbar=False
                )
            evolved_mps = circuit_mps.psi
    return d_path_dict


def fill_subspace_matrices_from_fname_dict(
    fname_dict: Dict[int, str], ham_mpo: MatrixProductOperator, d: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill subspace matrices from a dictionary mapping integers (the power of the unitary)
    to the filename where the MPS is stored."""

    h = np.zeros((d, d), dtype=complex)
    s = np.zeros((d, d), dtype=complex)
    for i in range(d):
        state_i = quimb.load_from_disk(fname_dict[i])
        for j in range(i+1, d):
            state_j = quimb.load_from_disk(fname_dict[j])
            h[i, j] = state_i.H @ ham_mpo.apply(state_j)
            s[i, j] = state_i.H @ state_j
    h += h.conj().T
    s += s.conj().T
    for i in range(d):
        state_i = quimb.load_from_disk(fname_dict[i])
        h[i, i] = state_i.H @ ham_mpo.apply(state_i)
        s[i, i] = state_i.H @ state_i
    return (h, s)


@dataclass 
class RTESubspaceResult:
    """Store the results of getting the H and S matrices."""
    tau: float
    nsteps: int
    h: np.ndarray
    s: np.ndarray
    ds: List[int]
    energies: List[float]


def threshold_eigenvalues(h: np.ndarray, s: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Remove all eigenvalues below a positive threshold eps.
    See Epperly et al. sec. 1.2."""

    # Build a matrix whose columns correspond to the positive eigenvectors of s.
    evals, evecs = la.eigh(s)
    positive_evals = []
    positive_evecs = []
    for i, ev in enumerate(evals):
        assert abs(ev.imag) < 1e-7
        if ev.real > eps:
            positive_evals.append(ev.real)
            positive_evecs.append(evecs[:, i])
    pos_evec_mat = np.vstack(positive_evecs).T
    # Project h and s into this subspace.
    new_s =  pos_evec_mat.conj().T @ s @ pos_evec_mat
    new_h = pos_evec_mat.conj().T @ h @ pos_evec_mat
    return new_h, new_s
