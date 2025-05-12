import h5py
import numpy as np
import scipy.sparse.linalg as la
import openfermion as of
from krylov_common import load_xyz_hamiltonian

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

def main():
    ham = load_xyz_hamiltonian()
    ham_matrix = of.linalg.get_sparse_operator(ham)

    evals, evecs = la.eigsh(ham_matrix, k=4, which="SA")
    ground_energy = np.min(evals)
    ground_state = evecs[:, 0]
    print(f"Ground state energy = {ground_energy}")

    f = h5py.File("xyz_exact.hdf5", "w")
    f.create_dataset("ground_energy", data=ground_energy)
    f.create_dataset("ground_state", data=ground_state)
    f.close()

if __name__ == "__main__":
    main()
