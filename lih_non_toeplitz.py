import argparse
import json
import h5py
import numpy as np
import openfermion as of
from openfermionpyscf import run_pyscf
import qiskit
from qiskit.qasm2 import dumps
from quimb.tensor.circuit import CircuitMPS
from krylov_common import tebd_states_to_scratch, fill_subspace_matrices_from_fname_dict
from tensor_network_common import pauli_sum_to_mpo
from convert import cirq_pauli_sum_to_qiskit_pauli_op

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="JSON file with parameters.")
    parser.add_argument("scratch_dir", type=str, help="scratch data directory.")
    parser.add_argument("output_file", type=str, help="HDF5 file with subspace matrix ouptput.")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        input_dict = json.load(f)

    molec = "LiH"
    basis = "sto-3g"
    n_elec = 4
    mpo_max_bond = input_dict["max_mpo_bond"]
    max_circuit_bond = input_dict["max_tebd_bond"]
    steps = input_dict["steps"]
    tau = input_dict["tau"]
    d = input_dict["d"]

    geometry = of.chem.geometry_from_pubchem(molec)
    multiplicity = 1
    molecule = of.chem.MolecularData(
        geometry, basis, multiplicity
    )
    molecule = run_pyscf(molecule, run_scf=1)
    print(f"HF energy:", molecule.hf_energy)
    hamiltonian = molecule.get_molecular_hamiltonian()
    hamiltonian_qubop = of.transforms.jordan_wigner(hamiltonian)
    nterms = len(hamiltonian_qubop.terms)
    print(f"Model has {nterms} terms.")
    hamiltonian_psum = of.transforms.qubit_operator_to_pauli_sum(hamiltonian_qubop)
    qs = hamiltonian_psum.qubits
    nq = len(qs)
    print(f"Model has {nq} qubits.")
    hamiltonian_mpo = pauli_sum_to_mpo(hamiltonian_psum, qs, mpo_max_bond)
    ham_qiskit = cirq_pauli_sum_to_qiskit_pauli_op(hamiltonian_psum)

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

    scratch_dir = "lih_scratch"
    fname_dict = tebd_states_to_scratch(
        ev_circuit_transpiled, reference_mps, max_circuit_bond,
        d, args.scratch_dir, None
    )
    h, s = fill_subspace_matrices_from_fname_dict(fname_dict, hamiltonian_mpo, d)

    f = h5py.File(args.output_file, "w")
    f.create_dataset("h", data=h)
    f.create_dataset("s", data=s)
    f.close()

if __name__ == "__main__":
    main()
