import argparse
import pickle
import h5py
import cirq
import openfermion as of
from kcommute import get_si_sets
from trotter_circuit import trotter_multistep_from_groups
import krylov_common as kc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tau", type=float, help="Evolution time for U.")
    parser.add_argument("steps", type=int, help="Number of steps for U.")
    parser.add_argument("d", type=int, help="Subspace dimension")
    parser.add_argument("output_file", type=str, help="Output filename.")
    args = parser.parse_args()

    #hamiltonian = kc.load_water_hamiltonian()
    ham_fermi = of.hamiltonians.fermi_hubbard(2, 2, 1.0, 2.0, spinless=True)
    hamiltonian: of.QubitOperator = of.transforms.jordan_wigner(ham_fermi)
    nq = of.count_qubits(hamiltonian)
    qs = cirq.LineQubit.range(nq)
    # We start in the same state as the exact diagonalization study.
    #state_prep_ckt = kc.hf_ref_circuit(nq, 10)
    state_prep_ckt = kc.neel_state_circuit(nq)
    # Get the first-order Trotter circuit.
    ham_paulisum = of.transforms.qubit_operator_to_pauli_sum(hamiltonian)
    assert len(ham_paulisum.qubits) == nq
    assert set(ham_paulisum.qubits).issubset(set(qs))
    groups = get_si_sets(ham_paulisum, k=1)
    with open("hubbard_groups.pkl", "wb") as f:
        pickle.dump(groups, f)
    tau = args.tau
    steps = args.steps
    evolution_ckt = trotter_multistep_from_groups(
        groups, qs, tau, steps
    )
    # Convert circuits to QASM.
    prep_ckt_qasm = cirq.qasm(state_prep_ckt)
    evolution_ckt_qasm = cirq.qasm(evolution_ckt)
    # Compute the subspace matrices.
    d_max = args.d
    h, s = kc.subspace_matrices(hamiltonian, state_prep_ckt, evolution_ckt, d_max)
    # Write to file.
    f = h5py.File(args.output_file, "w")
    f.create_dataset("tau", data=tau)
    f.create_dataset("steps", data=steps)
    f.create_dataset("prep_circuit_qasm", data=prep_ckt_qasm)
    f.create_dataset("evolution_circuit_qasm", data=evolution_ckt_qasm)
    f.create_dataset("groups", data=str(groups))
    f.create_dataset("d_max", data=d_max)
    f.create_dataset("h", data=h)
    f.create_dataset("s", data=s)
    f.close()

if __name__ == "__main__":
    main()
