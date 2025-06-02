import cirq
import openfermion as of
import qiskit
from qiskit.qasm2 import dump
import krylov_common as kc
from convert import cirq_pauli_sum_to_qiskit_pauli_op


def main():
    tau = 0.1
    steps = 1

    for n in [2, 3, 4]:
        print(f"n={n}")
        hamiltonian: of.QubitOperator = kc.load_hubbard_hamiltonian(n)
        nq = of.count_qubits(hamiltonian)
        qs = cirq.LineQubit.range(nq)

        ham_paulisum = of.transforms.qubit_operator_to_pauli_sum(hamiltonian)
        ham_qiskit = cirq_pauli_sum_to_qiskit_pauli_op(ham_paulisum)
        dt = tau / float(steps)
        ev_gate = qiskit.circuit.library.PauliEvolutionGate(ham_qiskit, time=dt)
        ev_ckt_qiskit = qiskit.QuantumCircuit(nq)
        for _ in range(steps):
            ev_ckt_qiskit.append(ev_gate, range(nq))

        fname = f"trotter_ckt_{n}_steps.qasm"
        dump(ev_ckt_qiskit, fname)
        compiled = qiskit.transpile(
            ev_ckt_qiskit,
            optimization_level=3,
            basis_gates=["u3", "cx"]
        )
        print(
            f"""
        Depth: {compiled.depth()}
        Gates: {", ".join([f"{k.upper()}: {v}" for k, v in compiled.count_ops().items()])}
        """
        )

if __name__ == "__main__":
    main()
