from typing import List
from time import perf_counter_ns
import argparse
import pickle
import json
from mpi4py import MPI
import h5py
import numpy as np
import scipy.linalg as la
import torch
import quimb
import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
import qiskit
import openfermion as of
import qiskit
from qiskit import qpy
from qiskit.qasm2 import dumps
from quimb.tensor.tensor_1d import MatrixProductState
from quimb.tensor.circuit import CircuitMPS
from kcommute import get_si_sets
from trotter_circuit import trotter_multistep_from_groups
import krylov_common as kc
from convert import cirq_pauli_sum_to_qiskit_pauli_op
from tensor_network_common import pauli_sum_to_mpo

def to_torch(x, device_name: str="cuda"):
    return torch.tensor(x, dtype=torch.complex64, device=device_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="JSON input file with parameters.")
    parser.add_argument("output_file", type=str, help="Output filename.")
    args = parser.parse_args()

    print("Reading input.")
    with open(args.input_file) as f:
        input_dict = json.load(f)
    n_occ = input_dict["n_occ"] # Number of occupied orbitals.
    steps = input_dict["steps"]
    d_vals = input_dict["d"]
    tau = input_dict["tau"]
    max_circuit_bond = input_dict["max_circuit_bond"]
    max_mpo_bond = input_dict["max_mpo_bond"]
    hamiltonian_mpo_filename = input_dict["mpo_filename"] # This is for reading an MPO.
    hamiltonian_file = input_dict["hamiltonian_file"]

    comm = MPI.COMM_WORLD
    mpi_comm_rank = comm.Get_rank()
    mpi_comm_size = comm.Get_size()

    # threshold_tolerance: float = 1e-2
    # ham_fermi = of.utils.load_operator(file_name=hamiltonian_file, data_directory=hamiltonian_directory)
    # hamiltonian = of.transforms.jordan_wigner(ham_fermi)
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
    nq = len(ham_mpo.tensors)
    qs = cirq.LineQubit.range(nq)

    print("Compiling circuits.")
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
    # with open("hubbard_trotter_ckt.qpy", "wb") as f:
    #     qpy.dump(ev_ckt_qiskit, f)

    # We will use a Hartree-Fock state as our reference.
    reference_circuit = qiskit.QuantumCircuit(nq)
    for i in range(nq):
        if i < n_occ:
            reference_circuit.x(i)
        # else:
        #     reference_circuit.id(i)
    # ref_state = qiskit.quantum_info.Statevector(reference_circuit).data
    # ref_state_energy = hamiltonian.expectation_from_state_vector(ref_state)
    ref_circuit_qasm = dumps(reference_circuit)
    quimb_circuit = CircuitMPS.from_openqasm2_str(ref_circuit_qasm)
    reference_mps = quimb_circuit.psi
    for tensor in reference_mps.tensors:
        tensor.modify(apply=lambda x: to_torch(x))
    assert len(reference_mps.tensor_map) == nq

    print(f"Untranspiled circuit has depth {ev_ckt_qiskit.depth()}")
    ev_ckt_transpiled = qiskit.transpile(ev_ckt_qiskit, basis_gates=["u3", "cx"])
    print(f"Transpiled circuit has depth {ev_ckt_transpiled.depth()}")

    mat_elems = []
    overlaps = []
    contract_start_time = perf_counter_ns()
    for d in d_vals:
        print(f"on d = {d}.")
        # Compute the subspace matrices.
        mat_elem, overlap = kc.tebd_matrix_element_and_overlap(
            ham_mpo, ev_ckt_transpiled, reference_mps,
            d, max_circuit_bond, to_torch
        )
        mat_elems.append(mat_elem)
        overlaps.append(overlap)
    contract_end_time = perf_counter_ns()
    contract_elapsed_time = contract_end_time - contract_start_time
    print(f"Contraction elapsed time = {contract_elapsed_time}")

    # TODO There are Torch tensors in overlaps and mat_elems. Make them serializable.
    overlap_data = [t.data.tolist() for t in overlaps]
    mat_elem_data = [t.data.tolist() for t in mat_elems]
    output_dict = {
        "input": input_dict,
        "d": d_vals,
        "overlaps": overlap_data,
        "mat_elems": mat_elem_data,
    }
    
    with open(args.output_file, "w") as f:
        json.dump(output_dict, f)

if __name__ == "__main__":
    main()