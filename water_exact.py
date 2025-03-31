"""Get ground state energy, energy gap, and reference state overlap for the water molecule."""

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import eigsh
import h5py
import openfermion as of
import cirq
from openfermion.functionals.get_one_norm_test import qubit_hamiltonian
from krylov_common import hf_ref_circuit, load_water_hamiltonian

def total_number_qubit_operator(norbitals: int) -> of.QubitOperator:
    """Get a qubit operator corresponding to N = sum_i a_i^ a_i.

    Arguments:
    norbitals: The number of spin-orbitals in the system."""

    number_operator = of.FermionOperator()
    for i in range(norbitals):
        number_operator += of.FermionOperator(f"{i}^ {i}")
    return of.transforms.jordan_wigner(number_operator)


def main():
    ham: of.QubitOperator = load_water_hamiltonian()
    nq = of.count_qubits(ham)
    nterms = len(ham.terms)
    # Get a circuit and vector for the reference state.
    ref_circuit = hf_ref_circuit(nq, nq)
    sim = cirq.Simulator()
    sim_result = sim.simulate(ref_circuit)
    ref_state = sim_result.final_state_vector
    # Use scipy to get eigenvectors.
    ham_sparse = of.linalg.qubit_operator_sparse(ham, n_qubits=nq)
    energies, eigenvectors = eigsh(
        ham_sparse,
        k=2,
        which='SA',
        maxiter=10_000
    )
    print(f"Final energy: {energies[0]}")
    # Get the expectation of the number operator.
    number_operator: of.QubitOperator = total_number_qubit_operator(nq)
    number_operator_sparse = of.linalg.qubit_operator_sparse(number_operator)
    number_expectation = np.vdot(eigenvectors[:, 0], number_operator_sparse @ eigenvectors[:, 0])
    print(f"Number expectation = {number_expectation}")
    # Output to HDF5
    f = h5py.File("water_exact.h5", "w")
    nq_dset = f.create_dataset("nq", data=nq)
    nterms_dset = f.create_dataset("nterms", data=nterms)
    ref_state_dset = f.create_dataset("ref_state", data=ref_state)
    energies_dset = f.create_dataset("energies", data=energies)
    eigenvector_dset = f.create_dataset("eigenvectors", data=eigenvectors)
    number_operator_dset = f.create_dataset("number_operator", data=number_expectation)
    f.close()

if __name__ == "__main__":
    main()