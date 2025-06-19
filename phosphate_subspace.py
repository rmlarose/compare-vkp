from typing import List
import argparse
import pickle
import json
import h5py
import numpy as np
import scipy.linalg as la
import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
import qiskit
import openfermion as of
import qiskit
from qiskit import qpy
from quimb.tensor.tensor_1d import MatrixProductState
from kcommute import get_si_sets
from trotter_circuit import trotter_multistep_from_groups
import krylov_common as kc
from convert import cirq_pauli_sum_to_qiskit_pauli_op

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="JSON input file with parameters.")
    parser.add_argument("hamiltonian_file", tye=str, help="File with Hamiltonian data.")
    parser.add_argument("output_file", type=str, help="Output filename.")
    args = parser.parse_args()

    with open(args.input_file) as f:
        input_dict = json.load(f)
    n_occ = input_dict["n_occ"] # Number of occupied orbitals.
    steps = input_dict["steps"]
    d = input_dict["d"]
    tau = input_dict["tau"]
    max_circuit_bond = input_dict["max_circuit_bond"]
    max_mpo_bond = input_dict["max_mpo_bond"]

    ham_fermi = of.utils.load_operator(file_name=args.hamiltonian_file, data_directory=".")
    hamiltonian = of.transforms.jordan_wigner(ham_fermi)
    nq = of.utils.count_qubits(hamiltonian)
    qs = cirq.LineQubit.range(nq)

    # Get the first-order Trotter circuit.
    use_paulihedral = True
    ham_paulisum = of.transforms.qubit_operator_to_pauli_sum(hamiltonian)
    if use_paulihedral:
        ham_qiskit = cirq_pauli_sum_to_qiskit_pauli_op(ham_paulisum)
        dt = tau / float(steps)
        ev_gate = qiskit.circuit.library.PauliEvolutionGate(ham_qiskit, time=dt)
        ev_ckt_qiskit = qiskit.QuantumCircuit(nq)
        for _ in range(steps):
            ev_ckt_qiskit.append(ev_gate, range(nq))
    else:
        assert len(ham_paulisum.qubits) == nq
        assert set(ham_paulisum.qubits).issubset(set(qs))
        tau = tau
        steps = steps
        dt = tau / float(steps)
        groups = get_si_sets(ham_paulisum, k=1)
        with open("hubbard_groups.pkl", "wb") as f:
            pickle.dump(groups, f)
        evolution_ckt = trotter_multistep_from_groups(
            groups, qs, tau, steps
        )
        ev_ckt_qasm = evolution_ckt.to_qasm()
        ev_ckt_qiskit = qiskit.QuantumCircuit.from_qasm_str(ev_ckt_qasm)
    with open("hubbard_trotter_ckt.qpy", "wb") as f:
        qpy.dump(ev_ckt_qiskit, f)

    # We will use a Hartree-Fock state as our reference.
    reference_circuit = qiskit.QuantumCircuit(nq)
    for i in range(nq):
        if i < n_occ:
            reference_circuit.x(i)
        else:
            reference_circuit.id(i)
    ref_state = qiskit.quantum_info.Statevector(reference_circuit).data
    ref_state_energy = hamiltonian.expectation_from_state_vector(ref_state)

    # Compute the subspace matrices.
    d_max = d
    use_tebd = True
    if not use_tebd:
        h, s = kc.subspace_matrices_from_ref_state(hamiltonian, ref_state, ev_ckt_qiskit, d_max)
    else:
        reference_mps = MatrixProductState.from_dense(ref_state)
        h, s = kc.tebd_subspace_matrices(ham_paulisum, ev_ckt_qiskit, reference_mps,
                                         d_max, max_circuit_bond, max_mpo_bond)
    # Write to file.
    f = h5py.File(args.output_file, "w")
    f.create_dataset("tau", data=tau)
    f.create_dataset("steps", data=steps)
    f.create_dataset("d_max", data=d_max)
    f.create_dataset("h", data=h)
    f.create_dataset("s", data=s)
    f.create_dataset("reference_energy", data=ref_state_energy)
    f.close()

if __name__ == "__main__":
    main()
