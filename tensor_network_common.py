from typing import List
import numpy as np
import cirq
import openfermion as of
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductOperator, MatrixProductState
from quimb.tensor.tensor_1d_compress import tensor_network_1d_compress_direct

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


def total_number_qubit_operator(n_orbitals: int, use_jw=True) -> of.QubitOperator:
    """Get a Pauli sum representing the total number operator.
    
    Arugments:
    n_orbitals - Number of orbitals in the system.
    use_jw - Whether to use Jordan-Wigner. Otherwise, use Bravyi-Kitaev.
    
    Returns:
    A QubitOperator representation of the total number operator."""

    total_number = of.FermionOperator.zero()
    for i in range(n_orbitals):
        total_number += of.FermionOperator(f"{i}^ {i}", 1.0)
    if use_jw:
        total_number_qubit = of.transforms.jordan_wigner(total_number)
    else:
        total_number_qubit = of.transforms.bravyi_kitaev(total_number)
    return total_number_qubit
