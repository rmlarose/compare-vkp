import numpy as np
import quimb
import quimb.tensor as qtn

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

if __name__ == "__main__":
    main()