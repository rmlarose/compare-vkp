from typing import List
import numpy as np
import openfermion as of

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


def add_number_term(
    hamiltonian: of.QubitOperator, n_occ: int,
    alpha: float, use_jw: bool = True
) -> of.QubitOperator:
    """Transforms the Hamiltonian as H -> H + alpha (N - N_occ) ** 2, where
    N is the total number operator, and N_occ is the desired number of electrons."""

    assert n_occ >= 0
    assert alpha >= 0.

    nq = of.utils.count_qubits(hamiltonian)
    num = total_number_qubit_operator(nq, use_jw=use_jw)
    augment_term = alpha * (num - n_occ) ** 2
    return hamiltonian + augment_term


def fock_state(bools: List[bool]) -> np.ndarray:
    """Get a fock state given occupations."""

    idx = 0
    for k, b in enumerate(bools[::-1]):
        if b:
            idx += 2 ** k
    psi = np.zeros(2 ** len(bools), dtype=complex)
    psi[idx] = 1.
    return psi