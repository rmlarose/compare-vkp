from typing import List, Dict, Tuple
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
import openfermion as of
from quimb.tensor.tensor_1d import MatrixProductState, MatrixProductOperator
import quimb
import quimb.tensor as qtn
import krylov_common as kc
from convert import cirq_pauli_sum_to_qiskit_pauli_op
from tensor_network_common import pauli_sum_to_mpo

def perturb_state_with_cb_state(ref_state: np.ndarray, cb_state: List[bool], ratio: float) -> np.ndarray:
    """Perturb the state with a computational basis state, similar to the above."""

    assert 0.0 <= ratio <= 1.0

    # Build a circuit that prepares the C.B. state.
    ckt = cirq.Circuit()
    qs = cirq.LineQubit.range(len(cb_state))
    for i, b in enumerate(cb_state):
        if b:
            ckt.append(cirq.X(qs[i]))
        else:
            ckt.append(cirq.I(qs[i]))
    # Get the C.B. statevector.
    sim = cirq.Simulator()
    result = sim.simulate(ckt)
    cb_vector = result.final_state_vector

    new_state = (1.0 - ratio) * ref_state + ratio * cb_vector
    return new_state / la.norm(new_state)


def tebd_states_to_scratch(
    ev_circuit: qiskit.QuantumCircuit,
    ref_state: MatrixProductState, max_bond: int, d: int,
    scratch_dir: str, backend_callback
) -> Dict[int, str]:
    """Do successive steps of TEBD with the same circuit, storing the intermediate MPS's
    in a scratch directory."""

    qasm_str = dumps(ev_circuit)
    d_path_dict: Dict[int, str] = {}
    evolved_mps = ref_state.copy()
    for i in range(d):
        fname = f"{scratch_dir}/state_{i}.dump"
        quimb.save_to_disk(evolved_mps, fname)
        d_path_dict[i] = fname
        if i != d - 1:
            if backend_callback is not None:
                circuit_mps = qtn.circuit.CircuitMPS.from_openqasm2_str(
                    qasm_str, psi0=evolved_mps, max_bond=max_bond, progbar=False,
                    to_backend=backend_callback
                )
            else:
                circuit_mps = qtn.circuit.CircuitMPS.from_openqasm2_str(
                    qasm_str, psi0=evolved_mps, max_bond=max_bond, progbar=False
                )
            evolved_mps = circuit_mps.psi
    return d_path_dict


def fill_subspace_matrices_from_fname_dict(
    fname_dict: Dict[int, str], ham_mpo: MatrixProductOperator, d: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill subspace matrices from a dictionary mapping integers (the power of the unitary)
    to the filename where the MPS is stored."""

    h = np.zeros((d, d), dtype=complex)
    s = np.zeros((d, d), dtype=complex)
    for i in range(d):
        state_i = quimb.load_from_disk(fname_dict[i])
        for j in range(i+1, d):
            state_j = quimb.load_from_disk(fname_dict[j])
            h[i, j] = state_i.H @ ham_mpo.apply(state_j)
            s[i, j] = state_i.H @ state_j
    h += h.conj().T
    s += s.conj().T
    for i in range(d):
        state_i = quimb.load_from_disk(fname_dict[i])
        h[i, i] = state_i.H @ ham_mpo.apply(state_i)
        s[i, i] = state_i.H @ state_i
    return (h, s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="JSON input file with parameters.")
    parser.add_argument("exact_input_file", type=str, help="HDF5 file with ground state.")
    parser.add_argument("output_file", type=str, help="Output filename.")
    args = parser.parse_args()

    with open(args.input_file) as f:
        input_dict = json.load(f)
    l = input_dict["l"] # Number of lattice sites on each side.
    n_occ = input_dict["n_occ"] # Number of occupied orbitals.
    t = input_dict["t"] # Hopping rate
    u = input_dict["u"] # Interaction strength.
    steps = input_dict["steps"]
    ratio = input_dict["ratio"]
    d = input_dict["d"]
    tau = input_dict["tau"]
    max_circuit_bond = input_dict["max_circuit_bond"]
    max_mpo_bond = input_dict["max_mpo_bond"]

    ham_fermi = of.hamiltonians.fermi_hubbard(l, l, t, u, spinless=True)
    hamiltonian = of.transforms.jordan_wigner(ham_fermi)
    ham_cirq = of.transforms.qubit_operator_to_pauli_sum(hamiltonian)
    nq = of.count_qubits(hamiltonian)
    qs = cirq.LineQubit.range(nq)

    ham_paulisum = of.transforms.qubit_operator_to_pauli_sum(hamiltonian)
    ham_qiskit = cirq_pauli_sum_to_qiskit_pauli_op(ham_paulisum)
    ham_mpo = pauli_sum_to_mpo(ham_paulisum, qs, max_mpo_bond)
    dt = tau / float(steps)
    ev_gate = qiskit.circuit.library.PauliEvolutionGate(ham_qiskit, time=dt)
    ev_ckt_qiskit = qiskit.QuantumCircuit(nq)
    for _ in range(steps):
        ev_ckt_qiskit.append(ev_gate, range(nq))

    f_in = h5py.File(args.exact_input_file, "r")
    ground_state = f_in["ground_state"][:]
    n_occ = round(f_in["number_expectation"][()].real)
    f_in.close()
    print(f"{n_occ} occupied states in reference.")
    b = [True] * n_occ + [False] * (nq - n_occ)
    assert 0.0 <= ratio <= 1.0
    ref_state = perturb_state_with_cb_state(ground_state, b, ratio)
    qubit_map = dict(zip(qs, range(len(qs))))
    ref_state_energy = ham_paulisum.expectation_from_state_vector(ref_state, qubit_map)
    ref_state_mps = MatrixProductState.from_dense(ref_state)
    print(f"Reference energy = {ref_state_energy}")
    del ground_state

    ev_circuit_transpiled = qiskit.transpile(ev_ckt_qiskit, basis_gates=["u3", "cx"])
    scratch_dir = "hubbard_scratch"
    fname_dict = tebd_states_to_scratch(
        ev_circuit_transpiled, ref_state_mps, max_circuit_bond,
        d, scratch_dir, None
    )
    h, s = fill_subspace_matrices_from_fname_dict(fname_dict, ham_mpo, d)

    f = h5py.File(args.output_file, "w")
    f.create_dataset("l", data=l)
    f.create_dataset("tau", data=tau)
    f.create_dataset("steps", data=steps)
    f.create_dataset("ratio", data=ratio)
    f.create_dataset("d_max", data=d)
    f.create_dataset("h", data=h)
    f.create_dataset("s", data=s)
    f.create_dataset("ref_state", data=ref_state)
    f.create_dataset("reference_energy", data=ref_state_energy)
    f.close()


if __name__ == "__main__":
    main()