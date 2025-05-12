import h5py
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import eigsh
import pandas as pd
import qiskit
import cirq
import openfermion as of
from trotter_circuit import trotter_multistep_from_groups
from kcommute import get_si_sets
import krylov_common as kc
import convert



def main():
    ham = kc.load_xyz_hamiltonian()

    # get a the reference state circuit.
    nq = of.utils.count_qubits(ham)
    qs = cirq.LineQubit.range(nq)
    use_vqe = True
    if use_vqe:
        # Load reference state from the VQE results file.
        reference_file = h5py.File("xyz_vqe.hdf5", "r")
        ref_state = np.array(reference_file["state"])
    else:
        # Use the exact ground state and perturb it by superposing with a random state.
        ref_ckt_qiskit = kc.cb_state_circuit_qiskit(nq, [False, False, False])
        ref_ckt = convert.qiskit_circuit_to_cirq(ref_ckt_qiskit)
        sim = cirq.Simulator()
        result = sim.simulate(ref_ckt)
        ref_state = result.final_state_vector
        rand_state = np.random.rand(ref_state.size)
        rand_state = rand_state / la.norm(rand_state)
        ref_state = ref_state + 0.9 * rand_state
        ref_state = ref_state / la.norm(ref_state)
    ham_matrix = of.linalg.get_sparse_operator(ham)
    print("Reference energy:", np.vdot(ref_state, ham_matrix @ ref_state))

    evals, evecs = eigsh(ham_matrix, k=4, which="SA")
    ground_energy = np.min(evals)
    ground_state = evecs[:, 0]
    print(f"Ground state energy = {ground_energy}")

    overlap = np.vdot(ref_state, ground_state)
    print("|<ref|ground state>|^2=", abs(overlap)**2)

    # Get the Trotter circuit.
    tau = 1e-1
    steps = 100
    ham_paulisum = of.transforms.qubit_operator_to_pauli_sum(ham)
    assert len(ham_paulisum.qubits) == nq
    assert set(ham_paulisum.qubits).issubset(set(qs))
    use_paulihedral = False
    if not use_paulihedral:
        print("Using fully-commuting Trotter circuits.")
        groups = get_si_sets(ham_paulisum, k=1)
        evolution_ckt = trotter_multistep_from_groups(
            groups, qs, tau, steps
        )
        ev_ckt_qasm = evolution_ckt.to_qasm()
        ev_ckt_qiskit = qiskit.QuantumCircuit.from_qasm_str(ev_ckt_qasm)
    else:
        print("Using Paulihedral")
        ham_qiskit = convert.cirq_pauli_sum_to_qiskit_pauli_op(ham_paulisum)
        ev_gate = qiskit.circuit.library.PauliEvolutionGate(ham_qiskit, time=tau/float(steps))
        ev_ckt_qiskit = qiskit.QuantumCircuit(nq)
        for _ in range(steps):
            ev_ckt_qiskit.append(ev_gate, range(nq))

    # Get subspace matrices.
    d_max = 15
    h, s = kc.subspace_matrices_from_ref_state(ham, ref_state, ev_ckt_qiskit, d_max)

    # Compute eigenvalues at different subspace sizes.
    eps = 1e-5
    results = []
    for d in range(3, d_max):
        h_d = h[:d, :d]
        s_d = s[:d, :d]
        h_pos, s_pos = kc.threshold_eigenvalues(h_d, s_d, eps)
        evals, evecs = la.eigh(h_pos, s_pos)
        results.append((d, np.min(evals)))
        print(d, np.min(evals))
    df = pd.DataFrame.from_records(results)
    df.index.name = "i"

    # File output.
    output_fname = "xyz_krylov.hdf5"
    df.to_hdf(output_fname, key="energies", mode="w")
    f = h5py.File(output_fname, "w")
    f.create_dataset("ref_state", data=ref_state)
    f.create_dataset("tau", data=tau)
    f.create_dataset("steps", data=steps)
    f.create_dataset("h", data=h)
    f.create_dataset("s", data=s)
    f.create_dataset("ev_ckt_qasm", data=qiskit.qasm2.dumps(ev_ckt_qiskit))

if __name__ == "__main__":
    main()
