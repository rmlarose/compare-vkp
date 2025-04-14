from typing import List
import h5py
import numpy as np
from scipy.optimize import minimize
import cirq
import openfermion as of
from krylov_common import load_hubbard_hamiltonian

def brickwall_ansatz(theta: np.ndarray, qs: List[cirq.Qid]) -> np.ndarray:
    """Build the circuit for the brick wall Ansatz.

    Arguments:
    theta - vector of parameters.
    qs - Qubits for Ansatz.

    Returns:
    circuit - Brick wall circuit for given parameters."""

    assert theta.shape == (3, len(qs))

    ckt = cirq.Circuit()
    for i in range(len(qs)):
        if i % 2 == 0:
    return ckt


def main():
    ham = load_hubbard_hamiltonian()
    ham_cirq = of.transforms.qubit_operator_to_pauli_sum(ham)