import numpy as np
import cirq
import openfermion as of
import qiskit
from quimb.tensor.tensor_1d import MatrixProductState
import krylov_common as kc
from convert import cirq_pauli_sum_to_qiskit_pauli_op


def main():
    l = 2
    t = 1.0
    u = 2.0
    tau = 1e-1
    steps = 200
    d = 2

    ham_fermi = of.hamiltonians.fermi_hubbard(l, l, t, u, spinless=True)
    hamiltonian = of.transforms.jordan_wigner(ham_fermi)
    nq = of.count_qubits(hamiltonian)
    ham_paulisum = of.transforms.qubit_operator_to_pauli_sum(hamiltonian)
    qs = cirq.LineQubit.range(nq)

    ham_qiskit = cirq_pauli_sum_to_qiskit_pauli_op(ham_paulisum)
    dt = tau / float(steps)
    ev_gate = qiskit.circuit.library.PauliEvolutionGate(ham_qiskit, time=dt)
    ev_ckt_qiskit = qiskit.QuantumCircuit(nq)
    for _ in range(steps):
        ev_ckt_qiskit.append(ev_gate, range(nq))

    # Use a computational basis state for the reference.
    psi = np.zeros(2 ** nq, dtype=complex)
    psi[0] = 1.0
    ref_mps = MatrixProductState.from_dense(psi)

    max_mpo_bond = 200
    bond_mat_elem_dict = {}
    bond_dims = [100, 200, 400, 500]
    for max_circuit_bond in bond_dims:
        mat_elem, overlap = kc.tebd_matrix_element_and_overlap(
            ham_paulisum, ev_ckt_qiskit, ref_mps, d, max_circuit_bond, max_mpo_bond
        )
        bond_mat_elem_dict[max_circuit_bond] = mat_elem
        print(f"chi={max_circuit_bond} mat_elem={mat_elem}")

if __name__ == "__main__":
    main()
