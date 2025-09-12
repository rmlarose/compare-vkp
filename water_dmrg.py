import cirq
import openfermion as of
import quimb
import quimb.tensor as qtn
import tensor_network_common as tc

def main():
    ham_mpo = quimb.utils.load_from_disk("water_mpo.data")

    max_mps_bond = 100
    dmrg = qtn.tensor_dmrg.DMRG(ham_mpo, max_mps_bond)
    converged = dmrg.solve()
    if not converged:
        print("DMRG did not converge!")
    final_energy = dmrg.energy
    print(f"Final DMRG energy = {final_energy}.")
    final_mps = dmrg.state
    quimb.utils.save_to_disk(final_mps, "water_dmrg_mps.data")

    total_number = tc.total_number_qubit_operator(len(ham_mpo.tensor_map))
    total_number_cirq = of.transforms.qubit_operator_to_pauli_sum(total_number)
    qs = cirq.LineQubit.range(len(ham_mpo.tensor_map))
    total_number_mpo = tc.pauli_sum_to_mpo(total_number_cirq, qs, 1_000)
    number_expectation = tc.mpo_mps_exepctation(total_number_mpo, final_mps)
    print(f"Expectation of N in ground state: {number_expectation}.")

if __name__ == "__main__":
    main()