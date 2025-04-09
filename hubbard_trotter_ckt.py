import openfermion as of
import cirq
from kcommute import get_si_sets
from diagonalize import diagonalize_pauli_strings

def main():
    ham_of = of.hamiltonians.fermi_hubbard(3, 3, 1.0, 2.0, spinless=False)
    ham_qubit = of.transforms.jordan_wigner(ham_of)
    ham_cirq = of.transforms.qubit_operator_to_pauli_sum(ham_qubit)
    groups = get_si_sets(ham_cirq, len(ham_cirq.qubits))
    for i, group in enumerate(groups):
        print(f"group {i}")
        for ps in group:
            print(ps)
        diag_ckt, diag_strings = diagonalize_pauli_strings(group, ham_cirq.qubits)
        print(diag_ckt)
        for ps in diag_strings:
            print(ps)

if __name__ == "__main__":
    main()
