from typing import List
import cirq
from kcommute import get_si_sets
from diagonalize import diagonalize_pauli_strings

def tfim_hamiltonian(l: int, h: float, j: List[float]) -> cirq.PauliSum:
    """Get Hamiltonian H = \\sum_i h sigma^x_i + \\sum_i j sigma^z_i simga^z_{i+1}
    acting on l qubits."""

    qs = cirq.LineQubit.range(l)
    ham = cirq.PauliSum()
    for i in range(l):
        ham += h * cirq.X(qs[i])
    for i in range(l):
        if i != l - 1:
            ham += j[0] * cirq.Z(qs[i]) * cirq.Z(qs[i+1])
            ham += j[1] * cirq.X(qs[i]) * cirq.X(qs[i+1])
            ham += j[2] * cirq.Y(qs[i]) * cirq.Y(qs[i+1])
    return ham


def main():
    h = 4.5
    jj = [1.0, 2.0, 3.0]
    ham = tfim_hamiltonian(4, h, jj)
    groups = get_si_sets(ham, 1)
    for i, group in enumerate(groups):
        print(f"group {i}\n", group)
    for group in groups:
        diag_ckt, diagonalized_paulis = diagonalize_pauli_strings(group, ham.qubits)
        print(diag_ckt)
        print(diagonalized_paulis)

if __name__ == "__main__":
    main()
