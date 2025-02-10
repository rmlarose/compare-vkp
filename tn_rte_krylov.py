"""Use tensor network methods to quantify subspace error for RTE quantum Krylov."""

from typing import List, Tuple, Dict
from copy import deepcopy
import numpy as np
import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
import openfermion as of
from openfermionpyscf import run_pyscf
from openfermion.chem import geometry_from_pubchem, MolecularData
import openfermion_helper
import quimb
import quimb.tensor as qtn
from quimb.tensor.tensor_1d_compress import tensor_network_1d_compress_direct
import qiskit
import qiskit.qasm3
import qiskit_ibm_runtime
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit.qasm2 import dumps
import convert

def load_hamiltonian() -> Tuple[of.QubitOperator, cirq.PauliSum]:
    """Load the water Hamiltonian and convert to both openfermion
    and cirq representations."""

    # Set parameters to make a simple molecule.
    geometry = geometry_from_pubchem("water")
    basis = "sto-3g"
    multiplicity = 1
    charge = 0

    # Make molecule and print out a few interesting facts about it.
    molecule = MolecularData(geometry, basis, multiplicity, charge)

    mol = run_pyscf(molecule, run_mp2=True, run_cisd=True, run_ccsd=True, run_fci=True)
    mol.save()
    water = MolecularData(filename=molecule.filename)
    hamiltonian = water.get_molecular_hamiltonian()
    hamiltonian = of.get_fermion_operator(hamiltonian)
    hamiltonian_openfermion = of.jordan_wigner(hamiltonian)
    hamiltonian = openfermion_helper.preprocess_hamiltonian(
        hamiltonian_openfermion, drop_term_if=[lambda term: term == ()]
    )  # Drop identity.
    return hamiltonian_openfermion, hamiltonian


def psum_to_mpo(psum: cirq.PauliSum, qs: List[cirq.Qid], max_bond: int) -> qtn.MatrixProductOperator:
    """Convert a Pauli sum into a matrix product operator."""

    assert max_bond > 0
    # Convert each string into an MPO, and add them together.
    for i, term in enumerate(psum):
        dense_string = term.dense(qs)
        # Convert each Pauli in the string to a list of matrix representations.
        operators = []
        for j, p in enumerate(dense_string.pauli_mask):
            syms = ['I', 'X', 'Y', 'Z']
            op = quimb.pauli(syms[p])
            # We must reshape the array to work in the MPO (add bond dimension 1 for Kronecker product)
            if j == 0 or j == len(qs) - 1:
                operators.append(op.reshape((1,) + op.shape))
            else:
                operators.append(op.reshape((1,1,) + op.shape))
        # This list of operators becomes an mpo.
        if i == 0:
            mpo = qtn.tensor_1d.MatrixProductOperator(operators)
        else:
            mpo += qtn.tensor_1d.MatrixProductOperator(operators)
            mpo = tensor_network_1d_compress_direct(mpo, max_bond=max_bond)
    return mpo


def dmrg_energy(mpo: quimb.tensor.tensor_1d.MatrixProductOperator) -> float:
    """Get energy with quimb's DMRG2 implementation."""

    dmrg = qtn.tensor_dmrg.DMRG2(hamiltonian_mpo)
    converged = dmrg.solve(bond_dims=10)
    assert converged
    return dmrg.energy


def ham_to_trotter_circuit(hamiltonian_qiskit) -> cirq.Circuit:
    """Use paulihedral to get a Trotter circuit, then convert to cirq format
    so it can be simulated with quimb."""

    order: int = 1
    cx_structure = "chain"  # "fountain"
    trotter_step = PauliEvolutionGate(hamiltonian_qiskit, time=1, synthesis=LieTrotter(cx_structure=cx_structure) if order == 1 else SuzukiTrotter(order, cx_structure=cx_structure))
    circuit = qiskit.QuantumCircuit(hamiltonian_qiskit.num_qubits)
    circuit.append(trotter_step, range(hamiltonian_qiskit.num_qubits))
    circuit = circuit.decompose(reps=2)
    circuit_qasm = dumps(circuit)
    return circuit_from_qasm(circuit_qasm)


def simulate_circuit_on_mps(
    circuit: cirq.Circuit,
    mps: qtn.tensor_1d.MatrixProductState,
    dtype: str = "float64",
    verbose: bool = False
) -> qtn.MatrixProductState:
    """Simulate the action of a circuit on an MPS."""

    qubits_to_indices = {q: i for i, q in enumerate(sorted(circuit.all_qubits()))}
    nqubits = len(qubits_to_indices)

    num_ops = len(list(circuit.all_operations()))
    for i, op in enumerate(circuit.all_operations()):
        mps.gate_(
            quimb.qarray(cirq.unitary(op)),
            [qubits_to_indices[q] for q in op.qubits],
            contract="swap+split"
        )
        if verbose:
            print(f"\rOp {i + 1} / {num_ops}", end="")

    return mps


def rte_subspace_matrices(
    reference_state: qtn.MatrixProductState, mpo: qtn.MatrixProductOperator, 
    trotter_circuit: cirq.Circuit, d: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the H and S matrices in the real-time basis of dimension d.
    Let U = exp(-i H dt). We want basis function |phi_m> = U^m |ref>,
    where |ref> is the reference state. Then, we compute the matrices
    H_mn = <phi_m|H|phi_n>, S_mn = <phi_m|phi_n>.
    Note that <phi_m|A|phi_n> = <ref|U^dagger^m A U^n |ref>
    = <ref|A U^(n - m) |ref>, for A = H,I."""

    assert d > 1
    # Get all matrix elements of the form H_k = <ref|H U^k |ref>
    # and S_k = <ref| U^k |ref>.
    h_dict: Dict[int, complex] = {}
    s_dict: Dict[int, complex] = {}
    mps = reference_state.copy()
    for i in range(d):
        # Update the mps.
        mps = simulate_circuit_on_mps(trotter_circuit, mps)
        # Evaluate <ref|H|psi> and <ref|psi>.
        h_psi = mps.copy().gate_with_mpo(mpo)
        h_dict[i] = reference_state.H @ h_psi
        s_dict[i] =  reference_state.H @ mps
    #  Fill H and S from the dictionaries.
    h = np.zeros((d, d), dtype=complex)
    s = np.zeros((d, d), dtype=complex)
    for i in range(d):
        for j in range(d):
            if i <= j:
                # <ref| U^dagger^i A U^j |ref> = <ref| A U^dagger^i U^j |ref>
                # = <ref| A U^(j-i) |ref>
                h[i,j] = h_dict[j - i]
                s[i,j] = s_dict[j - i]
            else:
                # <ref| U^dagger^i A U^j |ref> = <ref| U^dagger^i U^j A |ref>
                # = <ref| U^dagger^(i-j) A |ref> = (<ref| A U^(i-j) |ref>)*
                h[i,j] = h_dict[i - j].conjugate()
                s[i,j] = s_dict[i - j].conjugate()
    return h, s


def main() -> None:
    hamiltonian_of, hamiltonian = load_hamiltonian()
    qs = hamiltonian.qubits
    hamiltonian_mpo = psum_to_mpo(hamiltonian, qs, 8)
    hamiltonian_qiskit = convert.cirq_pauli_sum_to_qiskit_pauli_op(hamiltonian)
    trotter_circuit = ham_to_trotter_circuit(hamiltonian_qiskit)
    nqubits = len(hamiltonian.qubits)
    reference_state = qtn.MPS_computational_state("0" * nqubits, cyclic=False) 
    h, s = rte_subspace_matrices(reference_state, hamiltonian_mpo, trotter_circuit, 4)
    

if __name__ == "__main__":
    main()
