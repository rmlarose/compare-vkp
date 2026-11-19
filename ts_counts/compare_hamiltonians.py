"""Shot counts and numbers of groups for the TS, product and reactant Hamiltonians
over a range of Pauli-coefficient cutoffs.

Same workflow as ts_counts.ipynb, for each Hamiltonian:
  1. Jordan-Wigner qubit Hamiltonian, non-identity strings sorted by decreasing |c|.
  2. Ground state from DMRG on an MPO built with the fermionic cutoff `mpo_cutoff`
     (cached; the TS MPO/ground state from the notebook in results_cutoff_0.001/ are reused).
  3. For each grouping cutoff and each k: sorted-insertion k-commuting grouping of the strings
     with |c| >= cutoff, exact group variances from the MPS, and shots
     N = (sum_g sqrt(Var_g))^2 / epsilon^2. Only k = 44 = nqubits (full commutation) is used.
  4. Cut-off strings re-incorporated as in the notebook: Z-only cut strings in one group, every
     other cut string measured individually.

Because the strings are sorted once, the kept strings for any cutoff are a prefix of the full
sorted list, so term indices (and therefore the variance cache) are shared between cutoffs.

Output: one row per (hamiltonian, cutoff, k) in `output_csv`.
"""

import gc
import os
import pickle
import time
import argparse

import numpy as np
import openfermion as of
import pandas as pd
import quimb.tensor as qtn
from openfermion import InteractionOperator, get_fermion_operator, jordan_wigner
from openfermion.ops.representations import get_tensors_from_integrals

import ts_tools

# ----------------------------------------------------------------------------
# Parameters.
# ----------------------------------------------------------------------------

cutoffs = [1e-2, 1e-3, 1e-4]  # Pauli-coefficient cutoffs (Ha) for grouping.
k_values = [44]  # k = nqubits (44): fully-commuting sorted insertion.
epsilon = 1.6e-3  # Target precision in Ha (chemical accuracy).

# DMRG settings (same as ts_counts.ipynb so the cached TS ground state is reused).
mpo_cutoff = 1e-3
bond_dims = [16, 32, 64]
penalty_mu = 2.0
nworkers = 4  # Processes for multi-string group variances (~1.5 GB each for the largest groups).

dmrg_dir = f"results_cutoff_{mpo_cutoff:g}"  # <name>_mpo.pkl, <name>_ground_state.pkl
results_dir = "results_compare"  # Variance caches.
output_csv = "hamiltonian_comparison.csv"
os.makedirs(dmrg_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)


# ----------------------------------------------------------------------------
# Helpers.
# ----------------------------------------------------------------------------

def load_hamiltonian(path):
    """Returns (fermion Hamiltonian, qubit Hamiltonian, number of electrons)."""
    h_data = np.load(path)
    # ts.npz uses ECORE/H1/H2/NELEC; the product/reactant files use e_nuc/h1/h2/nelectron
    # (same chemist-notation H2 and the same role for the constant).
    if "H1" in h_data.files:
        ecore, h1, h2, nelec = h_data["ECORE"], h_data["H1"], h_data["H2"], h_data["NELEC"]
    else:
        ecore, h1, h2, nelec = h_data["e_nuc"], h_data["h1"], h_data["h2"], h_data["nelectron"]
    ecore = float(ecore)  # Stored as a 0-d array; OpenFermion needs a scalar.
    # H2 is in chemist notation, (pq|rs). get_tensors_from_integrals expects
    # OpenFermion's MolecularData ordering, two_body_integrals[p,q,r,s] = (ps|qr).
    h2 = h2.transpose(0, 2, 3, 1)
    one_body_tensor, two_body_tensor = get_tensors_from_integrals(h1, h2)
    hamiltonian_op = InteractionOperator(
        constant=ecore, one_body_tensor=one_body_tensor, two_body_tensor=two_body_tensor
    )
    fermion_hamiltonian = get_fermion_operator(hamiltonian_op)
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
    return fermion_hamiltonian, qubit_hamiltonian, int(nelec)


def ground_state(name, fermion_hamiltonian, nqubits, nelec):
    """DMRG ground state (cached in dmrg_dir). Returns the gs_data dict."""
    gs_file = f"{dmrg_dir}/{name}_ground_state.pkl"
    if os.path.exists(gs_file):
        with open(gs_file, "rb") as f:
            return pickle.load(f)

    mpo_file = f"{dmrg_dir}/{name}_mpo.pkl"
    if os.path.exists(mpo_file):
        with open(mpo_file, "rb") as f:
            mpo_data = pickle.load(f)
    else:
        start = time.time()
        fermion_hamiltonian_no = of.normal_ordered(fermion_hamiltonian)
        constant = fermion_hamiltonian_no.terms.get((), 0.0).real
        fermion_hamiltonian_cut = of.FermionOperator()
        fermion_hamiltonian_cut.terms = {
            t: c.real for t, c in fermion_hamiltonian_no.terms.items() if t != () and abs(c) >= mpo_cutoff
        }
        print(f"[{name}] fermionic terms after MPO cutoff: {len(fermion_hamiltonian_cut.terms)}")
        mpo = ts_tools.fermion_operator_to_mpo(fermion_hamiltonian_cut, nqubits)
        mpo_data = {"mpo": mpo, "constant": constant}
        with open(mpo_file, "wb") as f:
            pickle.dump(mpo_data, f)
        print(f"[{name}] built MPO in {time.time() - start:.0f} s")

    mpo, constant = mpo_data["mpo"], mpo_data["constant"]
    dmrg_mpo = mpo.add_MPO(ts_tools.number_penalty_mpo(nqubits, nelec, mu=penalty_mu))

    # Hartree-Fock plus a small random perturbation (see ts_counts.ipynb).
    hf = qtn.MPS_computational_state("1" * nelec + "0" * (nqubits - nelec), dtype="float64")
    noise = qtn.MPS_rand_state(nqubits, bond_dim=bond_dims[0], dtype="float64", seed=0)
    p0 = hf.add_MPS(noise.multiply(1e-2))
    p0.compress(max_bond=bond_dims[0])
    p0.normalize()

    dmrg = qtn.DMRG2(dmrg_mpo, bond_dims=bond_dims, cutoffs=1e-10, p0=p0)
    start = time.time()
    dmrg.solve(tol=1e-6, max_sweeps=10, verbosity=1)
    gs_data = {
        "psi": dmrg.state,
        "energy": dmrg.energy.real + constant,
        "sweep_energies": [e.real + constant for e in dmrg.energies],
        "bond_dims": bond_dims,
        "time": time.time() - start,
    }
    with open(gs_file, "wb") as f:
        pickle.dump(gs_data, f)
    return gs_data


def load_cache(path, energy):
    """Variance cache keyed by sorted term indices, reused only for the same ground state."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            saved = pickle.load(f)
        if np.isclose(saved["energy"], energy):
            return saved["variances"]
    return {}


def save_cache(path, energy, cache):
    with open(path, "wb") as f:
        pickle.dump({"energy": energy, "variances": cache}, f)


def sqrt_sum(variances):
    return float(np.sum(np.sqrt(np.clip(variances, 0.0, None))))


# ----------------------------------------------------------------------------
# Main loop.
# ----------------------------------------------------------------------------

def run_hamiltonian(name, path):
    fermion_hamiltonian, qubit_hamiltonian, nelec = load_hamiltonian(path)
    nqubits = of.count_qubits(qubit_hamiltonian)

    # All non-identity strings, sorted by decreasing |c|. Kept strings for a cutoff are a prefix.
    terms, coeffs, x, z = ts_tools.qubit_operator_to_arrays(qubit_hamiltonian)
    assert np.max(np.abs(coeffs.imag)) < 1e-12
    coeffs = coeffs.real
    abs_coeffs = np.abs(coeffs)
    z_only = np.array([all(p == "Z" for _, p in t) for t in terms])
    print(f"[{name}] {nqubits} qubits, {nelec} electrons, {len(terms)} non-identity strings")

    gs_data = ground_state(name, fermion_hamiltonian, nqubits, nelec)
    del fermion_hamiltonian, qubit_hamiltonian
    gc.collect()
    psi, energy = gs_data["psi"], gs_data["energy"]
    print(f"[{name}] DMRG energy {energy:.6f} Ha (max bond {psi.max_bond()})")

    cache_file = f"{results_dir}/{name}_variance_cache.pkl"
    cache = load_cache(cache_file, energy)

    rows = []
    for cutoff in cutoffs:
        nkept = int(np.sum(abs_coeffs >= cutoff))  # abs_coeffs is sorted in decreasing order.
        dropped_l1 = float(abs_coeffs[nkept:].sum())

        # Cut-off strings (k-independent): Z-only strings in one group, the rest individually.
        dropped_idx = np.arange(nkept, len(terms))
        dropped_groups = [np.array([i]) for i in dropped_idx[~z_only[nkept:]]]
        if z_only[nkept:].any():
            dropped_groups.append(dropped_idx[z_only[nkept:]])
        if dropped_groups:
            start = time.time()
            dropped_variances = ts_tools.group_variances(
                dropped_groups, terms, coeffs, nqubits, psi,
                cache=cache, nworkers=os.cpu_count(), chunksize=2000,
            )
            save_cache(cache_file, energy, cache)
            sqrt_sum_dropped = sqrt_sum(dropped_variances)
            print(f"[{name}] cutoff {cutoff:.2e}: {len(dropped_idx)} cut strings, "
                  f"sum sqrt(Var) = {sqrt_sum_dropped:.4f} Ha ({time.time() - start:.0f} s)")
        else:
            sqrt_sum_dropped = 0.0

        for k in k_values:
            start = time.time()
            groups = ts_tools.k_commuting_groups(x[:nkept], z[:nkept], nqubits, k)
            variances = ts_tools.group_variances(
                groups, terms, coeffs, nqubits, psi, cache=cache, nworkers=nworkers
            )
            save_cache(cache_file, energy, cache)
            sqrt_sum_kept = sqrt_sum(variances)
            row = {
                "hamiltonian": name,
                "cutoff": cutoff,
                "k": k,
                "nqubits": nqubits,
                "nterms_kept": nkept,
                "nterms_dropped": len(terms) - nkept,
                "dropped_l1": dropped_l1,
                "ngroups": len(groups),
                "shots": sqrt_sum_kept**2 / epsilon**2,
                "ngroups_reincorporated": len(groups) + len(dropped_groups),
                "shots_reincorporated": (sqrt_sum_kept + sqrt_sum_dropped) ** 2 / epsilon**2,
                "epsilon": epsilon,
                "dmrg_energy": energy,
            }
            rows.append(row)
            print(f"[{name}] cutoff {cutoff:.2e}, k = {k:2d}: {len(groups):6d} groups, "
                  f"{row['shots']:.3e} shots ({time.time() - start:.0f} s)")
    return rows


def main():
    parser = argparse.ArgumentParser("compare_hamiltonians")
    parser.add_argument("--hamiltonian-dir", type=str, default="../hamiltonians", help="DIR with npz files.")
    args = parser.parse_args()
    
    hamiltonians = {
        "ts": f"{args.hamiltonian_dir}/ts.npz",
        "product": f"{args.hamiltonian_dir}/po3_1w_product_22o.npz",
        "reactant": f"{args.hamiltonian_dir}/po3_1w_reactant_22o.npz",
    }
    all_rows = []
    for name, path in hamiltonians.items():
        all_rows += run_hamiltonian(name, path)
        # Write after each Hamiltonian so partial results survive an interruption.
        pd.DataFrame(all_rows).to_csv(output_csv, index=False)
        gc.collect()
    df = pd.DataFrame(all_rows)
    print(df.to_string(index=False))
    print(f"Saved {output_csv}")


if __name__ == "__main__":
    main()
