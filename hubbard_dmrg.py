from typing import List
import argparse
import json
import h5py
import numpy as np
from scipy.linalg import norm
import cirq
import openfermion as of
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductOperator, MatrixProductState
from quimb.tensor.tensor_1d_compress import tensor_network_1d_compress_direct
from krylov_common import load_hubbard_hamiltonian

def pauli_string_to_mpo(pstring: cirq.PauliString, qs: List[cirq.Qid]) -> MatrixProductOperator:
    """Convert a Pauli string to a matrix product operator."""

    # Make a list of matrices for each operator in the string.
    ps_dense = pstring.dense(qs)
    matrices: List[np.ndarray] = []
    for pauli_int in ps_dense.pauli_mask:
        if pauli_int == 0:
            matrices.append(np.eye(2))
        elif pauli_int == 1:
            matrices.append(cirq.unitary(cirq.X))
        elif pauli_int == 2:
            matrices.append(cirq.unitary(cirq.Y))
        else: # pauli_int == 3
            matrices.append(cirq.unitary(cirq.Z))
    # Convert the matrices into tensors. We have a bond dim chi=1 for a Pauli string MPO.
    tensors: List[np.ndarray] = []
    for i, m in enumerate(matrices):
        if i == 0:
            tensors.append(m.reshape((2, 2, 1)))
        elif i == len(matrices) - 1:
            tensors.append(m.reshape((1, 2, 2)))
        else:
            tensors.append(m.reshape((1, 2, 2, 1)))
    return pstring.coefficient * MatrixProductOperator(tensors, shape="ludr")


def pauli_sum_to_mpo(psum: cirq.PauliSum, qs: List[cirq.Qid], max_bond: int) -> MatrixProductOperator:
    """Convert a Pauli sum to an MPO."""

    for i, p in enumerate(psum):
        if i == 0:
            mpo = pauli_string_to_mpo(p, qs)
        else:
            mpo += pauli_string_to_mpo(p, qs)
            tensor_network_1d_compress_direct(mpo, max_bond=max_bond, inplace=True)
    return mpo


def mpo_mps_exepctation(mpo: MatrixProductOperator, mps: MatrixProductState) -> complex:
    """Get the expectation of an operator given the state."""

    mpo_times_mps = mpo.apply(mps)
    return mps.H @ mpo_times_mps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_filename", type=str, help="JSON file with simulation configuration")
    parser.add_argument("output_filename", type=str, help="HDF5 ouptut file for ground state.")
    args = parser.parse_args()

    with open(args.input_filename) as f:
        input_dict = json.load(f)
    l = input_dict["l"] # Number of lattice sites on each side.
    n_occ = input_dict["n_occ"] # Number of occupied orbitals.
    t = input_dict["t"] # Hopping rate
    u = input_dict["u"] # Interaction strength.
    alpha = input_dict["alpha"] # Factor to regulat occupation number.
    max_bond = input_dict["max_bond"] # Max bond for all calculations.

    # Load the fermionic Hamiltonian. 
    ham_fermi = of.hamiltonians.fermi_hubbard(l, l, t, u, spinless=True)
    ham_fermi_jw = of.transforms.jordan_wigner(ham_fermi)
    ham_cirq = of.transforms.qubit_operator_to_pauli_sum(ham_fermi_jw)
    nq = of.utils.count_qubits(ham_fermi_jw)
    qs = cirq.LineQubit.range(nq)
    ham_mpo = pauli_sum_to_mpo(ham_cirq, qs, max_bond)
    # Add a term alpha (\hat{N} - N_occ)^2 to ensure the occupation number.
    total_number = of.FermionOperator.zero()
    for i in range(l * l):
        total_number += of.FermionOperator(f"{i}^ {i}", 1.0)
    total_number_jw = of.transforms.jordan_wigner(total_number)
    total_number_cirq = of.transforms.qubit_operator_to_pauli_sum(total_number_jw)
    total_number_mpo = pauli_sum_to_mpo(total_number_cirq, qs, max_bond)
    ham_total = ham_fermi + alpha * (total_number - n_occ * of.FermionOperator.identity()) ** 2
    ham_total_jw = of.transforms.jordan_wigner(ham_total)
    ham_total_cirq = of.transforms.qubit_operator_to_pauli_sum(ham_total_jw)
    ham_total_mpo = pauli_sum_to_mpo(ham_total_cirq, qs, max_bond)

    # Solve with DMRG.
    dmrg = qtn.DMRG(ham_total_mpo, bond_dims=max_bond)
    converged = dmrg.solve()
    if not converged:
        print("DMRG failed to converge.")
    ground_state = dmrg.state

    # Convert the ground state MPS to a vector.
    gs_tensor = ground_state.contract()
    gs_numpy = gs_tensor.data
    gs_vector = gs_numpy.reshape(gs_numpy.size)

    energy = mpo_mps_exepctation(ham_mpo, ground_state)
    number = mpo_mps_exepctation(total_number_mpo, ground_state)
    qubit_map = {q: i for i, q in enumerate(qs)}
    energy_cirq = ham_cirq.expectation_from_state_vector(gs_vector, qubit_map)
    number_cirq = total_number_cirq.expectation_from_state_vector(gs_vector, qubit_map)
    print("Quimb result:")
    print(f"Energy = {energy}, number = {number}")
    print("Cirq result:")
    print(f"Energy = {energy_cirq}, number = {number_cirq}")

    f = h5py.File(args.output_filename, "w")
    f.create_dataset("ground_state", data=gs_vector)
    f.create_dataset("energy", data=energy)
    f.create_dataset("number", data=number)
    f.create_dataset("energy_cirq", data=energy_cirq)
    f.create_dataset("number_cirq", data=number_cirq)
    f.close()

if __name__ == "__main__":
    main()
