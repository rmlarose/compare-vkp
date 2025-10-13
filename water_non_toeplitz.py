import argparse
import h5py
import json
import numpy as np
import torch
import cirq
import openfermion as of
import qiskit
from qiskit.qasm2 import dumps
import quimb
from quimb.tensor.circuit import CircuitMPS
from convert import cirq_pauli_sum_to_qiskit_pauli_op
from krylov_common import tebd_states_to_scratch, fill_subspace_matrices_from_fname_dict

def to_torch(x, device_name: str="cuda"):
    return torch.tensor(x, dtype=torch.complex64, device=device_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="JSON input file with parameters.")
    parser.add_argument("output_file", type=str, help="HDF5 file for subspace matrix output.")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        input_dict = json.load(f)
    
    hamiltonian_file = input_dict["hamiltonian_file"]
    hamiltonian_mpo_filename = input_dict["hamiltonian_mpo_filename"]
    tau = input_dict["tau"]
    d = input_dict["d"]
    steps = input_dict["steps"]
    n_elec = input_dict["n_elec"]
    max_circuit_bond = input_dict["max_circuit_bond"]

    hamiltonian = of.jordan_wigner(
            of.get_fermion_operator(
        of.chem.MolecularData(filename=hamiltonian_file).get_molecular_hamiltonian()
        )
    )
    # hamiltonian.compress(abs_tol=threshold_tolerance)
    ham_cirq = of.transforms.qubit_operator_to_pauli_sum(hamiltonian)
    nq = of.utils.count_qubits(hamiltonian)
    qs = cirq.LineQubit.range(nq)
    ham_mpo = quimb.load_from_disk(hamiltonian_mpo_filename)
    for tensor in ham_mpo.tensors:
        tensor.modify(apply=lambda x: to_torch(x))
    for tensor in ham_mpo.tensors:
        tensor.modify(apply=lambda x: to_torch(x))
    nq = len(ham_mpo.tensors)
    qs = cirq.LineQubit.range(nq)

    ham_paulisum = of.transforms.qubit_operator_to_pauli_sum(hamiltonian)
    ham_qiskit = cirq_pauli_sum_to_qiskit_pauli_op(ham_paulisum)
    dt = tau / float(steps)
    ev_gate = qiskit.circuit.library.PauliEvolutionGate(ham_qiskit, time=dt)
    ev_ckt_qiskit = qiskit.QuantumCircuit(nq)
    for _ in range(steps):
        ev_ckt_qiskit.append(ev_gate, range(nq))
    ev_circuit_transpiled = qiskit.transpile(ev_ckt_qiskit, basis_gates=["u3", "cx"])

    reference_circuit = qiskit.QuantumCircuit(nq)
    for i in range(nq):
        if i < n_elec:
            reference_circuit.x(i)
    ref_circuit_qasm = dumps(reference_circuit)
    quimb_circuit = CircuitMPS.from_openqasm2_str(ref_circuit_qasm)
    reference_mps = quimb_circuit.psi
    for tensor in reference_mps.tensors:
        tensor.modify(apply=lambda x: to_torch(x))

    fname_dict = tebd_states_to_scratch(
        ev_circuit_transpiled, reference_mps, max_circuit_bond,
        d, args.scratch_dir, to_torch
    )
    h, s = fill_subspace_matrices_from_fname_dict(fname_dict, ham_mpo, d)

    f = h5py.File(args.output_file, "w")
    f.create_dataset("h", data=h)
    f.create_dataset("s", data=s)
    f.close()

if __name__ == "__main__":
    main()
