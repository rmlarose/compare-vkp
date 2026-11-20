"""DMRG ground states of the TS, product and reactant Hamiltonians, saved as portable .npz files
for compare_hamiltonians.py.

For each Hamiltonian:
  1. Build the MPO of the Jordan-Wigner Hamiltonian from the normal-ordered fermionic terms with
     |c| >= mpo_cutoff (as in ts_counts.ipynb). This takes ~35 min and the MPO is cached as a
     pickle in --work-dir, which is only reused on the machine that wrote it.
  2. DMRG with a (N - N_e)^2 penalty, starting from Hartree-Fock plus a small perturbation.
  3. Save <output-dir>/<name>_ground_state.npz: the MPS site tensors as plain NumPy arrays plus the
     energy and DMRG settings (see ts_tools.save_mps_npz / load_mps_npz). Only this file needs to
     be copied to another machine.

If <work-dir>/<name>_ground_state.pkl already exists (e.g. the TS ground state from the notebook),
it is converted to .npz instead of re-running DMRG.

Usage:
    python ground_states.py [--hamiltonian-dir ../hamiltonians] [--output-dir ground_states]
                            [--names ts product reactant]
"""

import argparse
import gc
import os
import pickle
import time

import numpy as np
import openfermion as of
import quimb
import quimb.tensor as qtn

import ts_tools

# DMRG settings (same as ts_counts.ipynb).
mpo_cutoff = 1e-3  # Fermionic-coefficient cutoff for the MPO.
bond_dims = [16, 32, 64]  # Bond-dimension schedule (last value repeats).
penalty_mu = 2.0  # Ha, strength of the (N - N_e)^2 penalty.
dmrg_tol = 1e-6
dmrg_max_sweeps = 10

hamiltonian_files = {
    "ts": "ts.npz",
    "product": "po3_1w_product_22o.npz",
    "reactant": "po3_1w_reactant_22o.npz",
}


def build_mpo(name, fermion_hamiltonian, nqubits, work_dir):
    """MPO of the cut fermionic Hamiltonian and its (dropped) constant, cached in work_dir."""
    mpo_file = f"{work_dir}/{name}_mpo.pkl"
    if os.path.exists(mpo_file):
        with open(mpo_file, "rb") as f:
            mpo_data = pickle.load(f)
        return mpo_data["mpo"], mpo_data["constant"]

    start = time.time()
    fermion_hamiltonian_no = of.normal_ordered(fermion_hamiltonian)
    constant = fermion_hamiltonian_no.terms.get((), 0.0).real
    fermion_hamiltonian_cut = of.FermionOperator()
    fermion_hamiltonian_cut.terms = {
        t: c.real for t, c in fermion_hamiltonian_no.terms.items() if t != () and abs(c) >= mpo_cutoff
    }
    del fermion_hamiltonian_no
    print(f"[{name}] fermionic terms after MPO cutoff: {len(fermion_hamiltonian_cut.terms)}")
    mpo = ts_tools.fermion_operator_to_mpo(fermion_hamiltonian_cut, nqubits)
    with open(mpo_file, "wb") as f:
        pickle.dump({"mpo": mpo, "constant": constant}, f)
    print(f"[{name}] built MPO in {time.time() - start:.0f} s (max bond {mpo.max_bond()})")
    return mpo, constant


def run_dmrg(name, mpo, constant, nqubits, nelec):
    dmrg_mpo = mpo.add_MPO(ts_tools.number_penalty_mpo(nqubits, nelec, mu=penalty_mu))

    # Hartree-Fock (lowest nelec spin-orbitals occupied, |1> = occupied in JW) plus a small random
    # perturbation: pure HF is a fixed point of the two-site update because of number conservation.
    hf = qtn.MPS_computational_state("1" * nelec + "0" * (nqubits - nelec), dtype="float64")
    noise = qtn.MPS_rand_state(nqubits, bond_dim=bond_dims[0], dtype="float64", seed=0)
    p0 = hf.add_MPS(noise.multiply(1e-2))
    p0.compress(max_bond=bond_dims[0])
    p0.normalize()

    dmrg = qtn.DMRG2(dmrg_mpo, bond_dims=bond_dims, cutoffs=1e-10, p0=p0)
    start = time.time()
    dmrg.solve(tol=dmrg_tol, max_sweeps=dmrg_max_sweeps, verbosity=1)
    return {
        "psi": dmrg.state,
        "energy": dmrg.energy.real + constant,
        "sweep_energies": [e.real + constant for e in dmrg.energies],
        "bond_dims": bond_dims,
        "time": time.time() - start,
    }


def main():
    parser = argparse.ArgumentParser("ground_states")
    parser.add_argument("--hamiltonian-dir", default="../hamiltonians", help="Directory with the Hamiltonian npz files.")
    parser.add_argument("--output-dir", default="ground_states", help="Where to write <name>_ground_state.npz.")
    parser.add_argument("--work-dir", default=f"results_cutoff_{mpo_cutoff:g}",
                        help="Machine-local cache for the MPO pickles (and existing ground-state pickles).")
    parser.add_argument("--names", nargs="+", default=list(hamiltonian_files), choices=list(hamiltonian_files),
                        help="Which Hamiltonians to run.")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)

    for name in args.names:
        out_file = f"{args.output_dir}/{name}_ground_state.npz"
        if os.path.exists(out_file):
            print(f"[{name}] {out_file} exists, skipping")
            continue

        fermion_hamiltonian, qubit_hamiltonian, nelec = ts_tools.load_hamiltonian(
            f"{args.hamiltonian_dir}/{hamiltonian_files[name]}"
        )
        nqubits = of.count_qubits(qubit_hamiltonian)
        del qubit_hamiltonian

        legacy_file = f"{args.work_dir}/{name}_ground_state.pkl"
        if os.path.exists(legacy_file):
            print(f"[{name}] converting existing {legacy_file}")
            with open(legacy_file, "rb") as f:
                gs_data = pickle.load(f)
        else:
            mpo, constant = build_mpo(name, fermion_hamiltonian, nqubits, args.work_dir)
            del fermion_hamiltonian
            gc.collect()
            gs_data = run_dmrg(name, mpo, constant, nqubits, nelec)
            del mpo
            # Keep the pickle too, so a lost .npz can be regenerated without re-running DMRG.
            with open(legacy_file, "wb") as f:
                pickle.dump(gs_data, f)

        psi = gs_data["psi"]
        psi_arrays = ts_tools.mps_to_arrays(psi)
        n_expect = sum(
            (1 - ts_tools.pauli_expectation(psi_arrays, ((i, "Z"),)).real) / 2 for i in range(nqubits)
        )
        ts_tools.save_mps_npz(
            out_file,
            psi,
            energy=gs_data["energy"],
            sweep_energies=np.array(gs_data["sweep_energies"]),
            bond_dims=np.array(gs_data["bond_dims"]),
            dmrg_time=gs_data["time"],
            number_expectation=n_expect,
            nelec=nelec,
            nqubits=nqubits,
            mpo_cutoff=mpo_cutoff,
            penalty_mu=penalty_mu,
            quimb_version=quimb.__version__,
            numpy_version=np.__version__,
        )
        print(f"[{name}] E = {gs_data['energy']:.6f} Ha, <N> = {n_expect:.6f} (expected {nelec}), "
              f"max bond {psi.max_bond()} -> {out_file}")
        del gs_data, psi, psi_arrays
        gc.collect()


if __name__ == "__main__":
    main()
