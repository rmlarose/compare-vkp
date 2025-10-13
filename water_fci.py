import openfermion as of
import numpy as np
from scipy.sparse.linalg import eigsh
import quimb
from quimb.tensor.tensor_1d import MatrixProductState
from tensor_network_common import total_number_qubit_operator, mpo_mps_exepctation

def main():
    hamiltonian_file = "monomer_eqb.hdf5"
    n_occ = 10 # Number of electrons.
    alpha = 1. # Term to ensure number of qubits.
    hamiltonian = of.jordan_wigner(
            of.get_fermion_operator(
        of.chem.MolecularData(filename=hamiltonian_file).get_molecular_hamiltonian()
        )
    )
    nqubits = of.utils.count_qubits(hamiltonian)
    number = total_number_qubit_operator(nqubits)
    occupation_term = alpha * (number - n_occ) ** 2
    total_hamiltonian = hamiltonian + occupation_term

    hamiltonian_sparse = of.linalg.get_sparse_operator(total_hamiltonian)
    eigvals, eigvecs = eigsh(hamiltonian_sparse, which="SA")
    i_min = np.argmin(eigvals.real)
    ground_energy = eigvals[i_min].real
    ground_state = eigvecs[:, i_min]
    print(f"Ground state energy = {ground_energy:10.9e}")

    # Test taking the mpo-mps expectation uinsg quimb.
    ground_state_mps = MatrixProductState.from_dense(ground_state)
    ham_mpo = quimb.load_from_disk("water_mpo.data")
    mps_energy = mpo_mps_exepctation(ham_mpo, ground_state_mps).real
    print(f"Ground state energy from MPO = {mps_energy:10.9e}")
    abs_err = abs(ground_energy - mps_energy)
    print(f"Absolute error = {abs_err:10.9e}")


if __name__ == "__main__":
    main()