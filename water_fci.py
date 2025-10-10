import openfermion as of
import numpy as np
from scipy.sparse.linalg import eigsh
from tensor_network_common import total_number_qubit_operator

def main():
    hamiltonian_file = "monomer_eqb.hdf5"
    n_occ = 10 # Number of electrons.
    alpha = 10. # Term to ensure number of qubits.
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
    eigvals, _ = eigsh(hamiltonian_sparse, which="SA")
    ground_energy = np.min(eigvals).real
    print(f"Ground state energy = {ground_energy:10.9e}")


if __name__ == "__main__":
    main()