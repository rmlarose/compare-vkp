from time import process_time_ns
import numpy as np
import scipy.linalg as la
import torch
import cirq
import qiskit
import openfermion as of
import qiskit
from qiskit.qasm2 import dumps
import quimb
from quimb.tensor.circuit import CircuitMPS
import krylov_common as kc
from convert import cirq_pauli_sum_to_qiskit_pauli_op

def to_torch(x):
    return torch.tensor(x, dtype=torch.complex64, device="cuda")

tau = 1e-1
steps = 1
n_occ = 32
threshold_tolerance: float = 1e-2
print(f"Simulating tau={tau} with {steps} steps.")

print("Loading and converting Hamiltonian.")
ham_start_time = process_time_ns()
ham_fermi = of.utils.load_operator(file_name="owp_631gd_22_ducc.data", data_directory=".")
hamiltonian = of.transforms.jordan_wigner(ham_fermi)
hamiltonian.compress(abs_tol=threshold_tolerance)
nq = of.utils.count_qubits(hamiltonian)
qs = cirq.LineQubit.range(nq)
ham_end_time = process_time_ns()
ham_elapsed_time = ham_end_time - ham_start_time
print(f"Time to load {ham_elapsed_time:1.6e}")

print("Compiling circuit.")
compile_start_time = process_time_ns()
ham_paulisum = of.transforms.qubit_operator_to_pauli_sum(hamiltonian)
ham_qiskit = cirq_pauli_sum_to_qiskit_pauli_op(ham_paulisum)
dt = tau / float(steps)
ev_gate = qiskit.circuit.library.PauliEvolutionGate(ham_qiskit, time=dt)
ev_ckt_qiskit = qiskit.QuantumCircuit(nq)
for _ in range(steps):
    ev_ckt_qiskit.append(ev_gate, range(nq))
compile_end_time = process_time_ns()
compile_elapsed_time = compile_end_time - compile_start_time
transpiled_circuit = qiskit.transpile(ev_ckt_qiskit, basis_gates=["u3", "cx"])
print(f"Time to compile {compile_elapsed_time:1.6e}")
print(f"Circuit depth {transpiled_circuit.depth()}")

reference_circuit = qiskit.QuantumCircuit(nq)
for i in range(nq):
    if i < n_occ:
        reference_circuit.x(i)
ref_circuit_qasm = dumps(reference_circuit)
quimb_circuit = CircuitMPS.from_openqasm2_str(ref_circuit_qasm)
reference_mps = quimb_circuit.psi
for tensor in reference_mps.tensors:
    tensor.modify(apply=lambda x: to_torch(x))
assert len(reference_mps.tensor_map) == nq

max_circuit_bond = 64
max_mpo_bond = 80
ham_mpo = quimb.load_from_disk("phosphate_mpo_chi597.data")
for tensor in ham_mpo.tensors:
    tensor.modify(apply=lambda x: to_torch(x))
d = 1
print("Computing overlap and matrix element.")
print(f"d = {d} Max circuit bond dim = {max_circuit_bond} Max MPO bond = {max_mpo_bond}")
contract_start_time = process_time_ns()
kc.tebd_matrix_element_and_overlap(
    ham_mpo, transpiled_circuit, reference_mps,
    d, max_circuit_bond, to_torch
)
contract_end_time = process_time_ns()
contract_elapsed_time = contract_end_time - contract_start_time
print(f"Time to contract {contract_elapsed_time:1.6e}")
