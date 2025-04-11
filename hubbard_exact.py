"""Get ground state energy, energy gap, and reference state overlap for the water molecule."""

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import eigsh
import h5py
import openfermion as of
import cirq
from openfermion.functionals.get_one_norm_test import qubit_hamiltonian
from krylov_common import neel_state_circuit, load_water_hamiltonian

def total_number_qubit_operator(norbitals: int) -> of.QubitOperator:
    """Get a qubit operator corresponding to N = sum_i a_i^ a_i.

    Arguments:
    norbitals: The number of spin-orbitals in the system."""

    number_operator = of.FermionOperator()
    for i in range(norbitals):
        number_operator += of.FermionOperator(f"{i}^ {i}")
    return of.transforms.jordan_wigner(number_operator)


def main():
    # Build the Fermi-Hubbard Hamiltonian.
    #ham: of.QubitOperator = load_water_hamiltonian()
    ham_fermi = of.hamiltonians.fermi_hubbard(2, 2, 1.0, 2.0, spinless=True)
    ham: of.QubitOperator = of.transforms.jordan_wigner(ham_fermi)
    nq = of.count_qubits(ham)
    nterms = len(ham.terms)
    # Get a circuit and vector for the reference state.
    #ref_circuit = hf_ref_circuit(nq, nq)
    ref_circuit = neel_state_circuit(nq)
    sim = cirq.Simulator()
    sim_result = sim.simulate(ref_circuit)
    ref_state = sim_result.final_state_vector
    # We will solve the eigenvalue problem with H -> H + alpha (N - N_occ)^2,
    # where N_occ is the number of the
    number_operator: of.QubitOperator = total_number_qubit_operator(nq)
    number_operator_sparse = of.linalg.qubit_operator_sparse(number_operator)
    reference_number_expectation = np.vdot(ref_state, number_operator_sparse @ ref_state)
    regularizer = (number_operator - reference_number_expectation) ** 2
    alpha = 10.0
    ham_regularized = ham + alpha * regularizer
    # Use scipy to get eigenvectors of the number-regularized Hamiltonian.
    ham_sparse = of.linalg.qubit_operator_sparse(ham, n_qubits=nq)
    ham_reg_sparse = of.linalg.qubit_operator_sparse(ham_regularized, n_qubits=nq)
    reg_energies, eigenvectors = eigsh(
        ham_reg_sparse,
        k=6,
        which='SA',
        maxiter=10_000
    )
    # Get the expectation of the original Hamiltonian.
    energies = np.zeros(2, dtype=complex)
    norm0 = np.vdot(eigenvectors[:, 0], eigenvectors[:, 0])
    energies[0] = np.vdot(eigenvectors[:, 0], ham_sparse @ eigenvectors[:, 0]) / norm0
    exc_idx = 2
    norm1 = np.vdot(eigenvectors[:, exc_idx], eigenvectors[:, exc_idx])
    energies[1] = np.vdot(eigenvectors[:, exc_idx], ham_sparse @ eigenvectors[:, exc_idx]) / norm1
    ref_norm = np.vdot(ref_state, ref_state)
    ref_energy = np.vdot(ref_state, ham_sparse @ ref_state)
    assert abs(energies[0] - energies[1]) >= 1e-14, \
        f"Energies must be non-degenerate, but got gap = {abs(energies[0] - energies[1])}."
    print(f"Energy = {energies[0]}")
    print(f"Energy gap = {abs(energies[1] - energies[0])}.")
    print(f"Reference energy = {ref_energy}.")
    # Get the expectation of the number operator.
    number_expectation = np.vdot(eigenvectors[:, 0], number_operator_sparse @ eigenvectors[:, 0])
    print(f"Number expectation = {number_expectation}")
    print(f"Reference state number expectation = {reference_number_expectation}")
    # Output to HDF5
    f = h5py.File("hubbard_exact.h5", "w")
    nq_dset = f.create_dataset("nq", data=nq)
    nterms_dset = f.create_dataset("nterms", data=nterms)
    ref_state_dset = f.create_dataset("ref_state", data=ref_state)
    energies_dset = f.create_dataset("energies", data=energies)
    eigenvector_dset = f.create_dataset("eigenvectors", data=eigenvectors)
    number_exp_dset = f.create_dataset("number_expectation", data=number_expectation)
    ref_num_dset = f.create_dataset("reference_number_expectation", data=reference_number_expectation)
    ref_energy_dset = f.create_dataset("reference_energy", data=ref_energy)
    f.close()

if __name__ == "__main__":
    main()