"""Shot counts and numbers of groups for the TS, product and reactant Hamiltonians
over a range of Pauli-coefficient cutoffs.

Same workflow as ts_counts.ipynb, for each Hamiltonian:
  1. Jordan-Wigner qubit Hamiltonian, non-identity strings sorted by decreasing |c|.
  2. Ground state read from <ground-state-dir>/<name>_ground_state.npz, written by
     ground_states.py (run that first; the .npz files are portable between machines).
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

import ts_tools

# ----------------------------------------------------------------------------
# Parameters.
# ----------------------------------------------------------------------------

cutoffs = [1e-2, 1e-3, 1e-4]  # Pauli-coefficient cutoffs (Ha) for grouping.
k_values = [44]  # k = nqubits (44): fully-commuting sorted insertion.
epsilon = 1.6e-3  # Target precision in Ha (chemical accuracy).

# APPROXIMATION: group MPOs with bond dimension above max_mpo_bond are SVD-truncated to it before
# computing the group variance (--max-mpo-bond; 0 = exact). Exact <H_g^2> needs memory
# ~ (MPS bond)^2 * (MPO bond)^2: at cutoff 1e-4, k = 44 the TS groups reach MPO bond ~600 (~90 GB).
# Tested on TS groups with exact bond 160: truncating to 96 changed sqrt(Var) by <= 0.6%,
# to 64 by <= 5%, to 32 by up to 90%. On the worst TS group (exact bond 593) 64 vs 128 differ by <1%.
# Peak memory per worker on that group (macOS): 3.7 GB at 64, 6.8 GB at 96, 9.0 GB at 128.
# The CSV records max_mpo_bond and how many groups were truncated.
max_mpo_bond = 64

# Worker processes for the variances (set with --nworkers / --nworkers-singles). Each worker is a
# separate process holding its own copy of the Pauli strings and imports (~1 GB), plus the group
# contraction (see above), so these, not the core count, set the memory footprint.
nworkers = 4  # Multi-string groups.
nworkers_singles = 4  # Cut-off strings measured individually (cheap, low memory per task).

ground_state_dir = "ground_states"  # <name>_ground_state.npz from ground_states.py (--ground-state-dir).
results_dir = "results_compare"  # Variance caches.
output_csv = "hamiltonian_comparison.csv"
os.makedirs(results_dir, exist_ok=True)


# ----------------------------------------------------------------------------
# Helpers.
# ----------------------------------------------------------------------------

def load_cache(path, energy):
    """Returns (variances, exact MPO bonds), both keyed by sorted term indices.

    Reused only for the same ground state and either the same max_mpo_bond or an older,
    fully exact cache (no "max_mpo_bond" entry; exact values are at least as good)."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            saved = pickle.load(f)
        same_state = abs(saved["energy"] - energy) < 1e-9  # np.isclose is too loose at ~600 Ha.
        exact_cache = saved.get("max_mpo_bond", None) is None
        if same_state and (exact_cache or saved["max_mpo_bond"] == max_mpo_bond):
            return saved["variances"], saved.get("mpo_bonds", {})
        print(f"Ignoring {path}: different ground state or max_mpo_bond")
    return {}, {}


def save_cache(path, energy, cache, bonds):
    with open(path, "wb") as f:
        pickle.dump({"energy": energy, "max_mpo_bond": max_mpo_bond, "variances": cache, "mpo_bonds": bonds}, f)


def sqrt_sum(variances):
    return float(np.sum(np.sqrt(np.clip(variances, 0.0, None))))


# ----------------------------------------------------------------------------
# Main loop.
# ----------------------------------------------------------------------------

def available_cpus():
    """CPUs allocated to this job. os.cpu_count() reports every core on a cluster node,
    not the job's allocation."""
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # macOS
        return os.cpu_count()


def run_hamiltonian(name, path):
    # Fail early, before the expensive Hamiltonian construction, if the ground state is missing.
    gs_file = f"{ground_state_dir}/{name}_ground_state.npz"
    if not os.path.exists(gs_file):
        raise FileNotFoundError(f"{gs_file} not found: run ground_states.py first and copy it here.")

    fermion_hamiltonian, qubit_hamiltonian, nelec = ts_tools.load_hamiltonian(path)
    del fermion_hamiltonian
    nqubits = of.count_qubits(qubit_hamiltonian)

    # All non-identity strings, sorted by decreasing |c|. Kept strings for a cutoff are a prefix.
    terms, coeffs, x, z = ts_tools.qubit_operator_to_arrays(qubit_hamiltonian)
    # assert np.max(np.abs(coeffs.imag)) < 1e-12
    coeffs = coeffs.real
    abs_coeffs = np.abs(coeffs)
    z_only = np.array([all(p == "Z" for _, p in t) for t in terms])
    print(f"[{name}] {nqubits} qubits, {nelec} electrons, {len(terms)} non-identity strings")

    del qubit_hamiltonian
    gc.collect()
    psi, gs_meta = ts_tools.load_mps_npz(gs_file)
    energy = float(gs_meta["energy"])
    assert psi.L == nqubits and int(gs_meta["nelec"]) == nelec, f"{gs_file} does not match {path}"
    print(f"[{name}] DMRG energy {energy:.6f} Ha (max bond {psi.max_bond()})")

    cache_file = f"{results_dir}/{name}_variance_cache.pkl"
    cache, bonds = load_cache(cache_file, energy)

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
                cache=cache, nworkers=nworkers_singles, chunksize=2000,
            )
            save_cache(cache_file, energy, cache, bonds)
            sqrt_sum_dropped = sqrt_sum(dropped_variances)
            print(f"[{name}] cutoff {cutoff:.2e}: {len(dropped_idx)} cut strings, "
                  f"sum sqrt(Var) = {sqrt_sum_dropped:.4f} Ha ({time.time() - start:.0f} s)")
        else:
            sqrt_sum_dropped = 0.0

        for k in k_values:
            start = time.time()
            groups = ts_tools.k_commuting_groups(x[:nkept], z[:nkept], nqubits, k)
            variances = ts_tools.group_variances(
                groups, terms, coeffs, nqubits, psi, cache=cache, nworkers=nworkers,
                max_mpo_bond=max_mpo_bond, bond_cache=bonds,
            )
            save_cache(cache_file, energy, cache, bonds)
            sqrt_sum_kept = sqrt_sum(variances)
            # Groups whose variance used a truncated MPO (entries from an older exact cache have no
            # recorded bond and were exact). The Z-only cut-off group is a sum of Z strings with a
            # small MPO, so it is not truncated in practice; it is counted here if it was.
            keys = [tuple(sorted(int(j) for j in g)) for g in groups] + \
                   [tuple(sorted(int(j) for j in g)) for g in dropped_groups if len(g) > 1]
            ntruncated = 0 if max_mpo_bond is None else sum(bonds.get(key, 0) > max_mpo_bond for key in keys)
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
                "max_mpo_bond": max_mpo_bond if max_mpo_bond is not None else "exact",
                "ngroups_truncated": ntruncated,
                "epsilon": epsilon,
                "dmrg_energy": energy,
            }
            rows.append(row)
            print(f"[{name}] cutoff {cutoff:.2e}, k = {k:2d}: {len(groups):6d} groups, "
                  f"{row['shots']:.3e} shots, {ntruncated} group MPOs truncated to bond {max_mpo_bond} "
                  f"({time.time() - start:.0f} s)")
    return rows


def main():
    global nworkers, nworkers_singles, ground_state_dir, max_mpo_bond  # run_hamiltonian reads these.
    parser = argparse.ArgumentParser("compare_hamiltonians")
    parser.add_argument("--hamiltonian-dir", type=str, default="../hamiltonians", help="DIR with npz files.")
    parser.add_argument("--nworkers", type=int, default=nworkers,
                        help="Worker processes for multi-string group variances.")
    parser.add_argument("--nworkers-singles", type=int, default=nworkers_singles,
                        help="Worker processes for the individually measured cut-off strings.")
    parser.add_argument("--ground-state-dir", type=str, default=ground_state_dir,
                        help="DIR with <name>_ground_state.npz files from ground_states.py.")
    parser.add_argument("--max-mpo-bond", type=int, default=max_mpo_bond,
                        help="Truncate group MPOs above this bond dimension (approximate; bounds memory). 0 = exact.")
    args = parser.parse_args()
    ground_state_dir = args.ground_state_dir
    max_mpo_bond = args.max_mpo_bond or None
    print("Group variances: " + ("exact" if max_mpo_bond is None else
          f"APPROXIMATE, group MPOs truncated to bond dimension {max_mpo_bond}"))
    # Never use more workers than allocated CPUs.
    nworkers = min(args.nworkers, available_cpus())
    nworkers_singles = min(args.nworkers_singles, available_cpus())
    print(f"{available_cpus()} CPUs available; {nworkers} workers for groups, "
          f"{nworkers_singles} for single strings")
    
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
