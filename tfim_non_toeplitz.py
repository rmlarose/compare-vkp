import argparse
import json
from mpi4py import MPI
import h5py
import numpy as np
import scipy.linalg as la
import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
import qiskit
from qiskit.qasm2 import dumps
from quimb.tensor.tensor_1d import MatrixProductState, MatrixProductOperator
import quimb
import quimb.tensor as qtn
from quimb.tensor.circuit import CircuitMPS
from krylov_common import tebd_states_to_scratch, fill_subspace_matrices_from_fname_dict
from convert import cirq_pauli_sum_to_qiskit_pauli_op
from tensor_network_common import pauli_sum_to_mpo

def tfim_hamiltonian(L: int, j: float, mu: float, delta: float) -> cirq.PauliSum:
    """Get the L-qubit TFIM Hamiltonian
    H = j sum_{i}  z_i z_{i+1} + mu sum_i z_i + delta sum_i x_i"""

    qs = cirq.LineQubit.range(L)
    ham = cirq.PauliSum()
    for i in range(len(qs)):
        if i != len(qs) - 1:
            ham += j * cirq.Z.on(qs[i]) * cirq.Z.on(qs[i + 1])
        ham += mu * cirq.Z.on(qs[i]) + delta * cirq.X.on(qs[i])
    return ham


def main():
    L = 8
    j = -1.0
    mu = 0.1
    delta = 0.3
    max_mpo_bond = 100
    max_circuit_bond = 100
    tau = 0.2
    steps = 10
    d = 20

    ham = tfim_hamiltonian(L, j, mu, delta)
    qs = ham.qubits
    nq = len(qs)
    ham_qiskit = cirq_pauli_sum_to_qiskit_pauli_op(ham)
    ham_mpo = pauli_sum_to_mpo(ham, qs, max_mpo_bond)
    dt = tau / float(steps)
    ev_gate = qiskit.circuit.library.PauliEvolutionGate(ham_qiskit, time=dt)
    ev_ckt_qiskit = qiskit.QuantumCircuit(nq)
    for _ in range(steps):
        ev_ckt_qiskit.append(ev_gate, range(nq))
    ev_circuit_transpiled = qiskit.transpile(ev_ckt_qiskit, basis_gates=["u3", "cx"])
    
    ref_circuit = qiskit.QuantumCircuit(nq)
    for i in range(nq):
        ref_circuit.x(i)
    ref_circuit_qasm = dumps(ref_circuit)
    quimb_circuit = CircuitMPS.from_openqasm2_str(ref_circuit_qasm)
    reference_mps = quimb_circuit.psi

    scratch_dir = "tfim_scratch"
    fname_dict = tebd_states_to_scratch(
        ev_circuit_transpiled, reference_mps, max_circuit_bond,
        d, scratch_dir, None
    )
    h, s = fill_subspace_matrices_from_fname_dict(fname_dict, ham_mpo, d)

    output_file = f"{scratch_dir}/subspace.hdf5"
    f = h5py.File(output_file, "w")
    f.create_dataset("h", data=h)
    f.create_dataset("s", data=s)
    f.close()

    # Get the exact energy.
    ham_mat = ham.matrix(qs)
    eigvals, _ = la.eig(ham_mat)
    ground_energy = np.min(eigvals)
    print(f"Exact ground state energy: {ground_energy}")

if __name__ == "__main__":
    main()