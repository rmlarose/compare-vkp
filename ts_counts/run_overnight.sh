#!/bin/bash
# End-to-end workflow on the Mac (16 GB): ground states, then shot counts / group counts for the
# TS, product and reactant Hamiltonians at cutoffs 1e-2, 1e-3, 1e-4 with k = 44.
#
#   cd ts_counts && ./run_overnight.sh
#
# Output: hamiltonian_comparison.csv, and a log in logs/. Both steps skip work that is already
# done (ground_states/*.npz, variance caches in results_compare/), so re-running after an
# interruption resumes. Keep the Mac plugged in; caffeinate stops it from sleeping while this runs.
#
# APPROXIMATION: group MPOs with bond dimension above MAX_MPO_BOND are SVD-truncated before the
# group variance is computed (exact groups need up to ~90 GB). The CSV records max_mpo_bond and
# ngroups_truncated for each row.

set -euo pipefail

MAX_MPO_BOND=64   # Tested: <= 5% error in sqrt(Var) per group vs. exact (96: <= 0.6%, but ~7 GB/worker).
NWORKERS=3        # Processes for multi-string group variances (worst group measured at 3.7 GB/worker; 3 fit in 16 GB).
NWORKERS_SINGLES=8  # Processes for the individually measured cut-off strings (light, ~0.5 GB each).

cd "$(dirname "$0")"
source ../.venv/bin/activate
mkdir -p logs
LOG="logs/run_$(date +%Y%m%d_%H%M%S).log"
caffeinate -i -w $$ &  # Prevent idle sleep for as long as this script runs.

echo "Logging to $LOG"
{
    echo "=== $(date): ground states ==="
    # TS is skipped if ground_states/ts_ground_state.npz exists. Product/reactant: ~35 min MPO build
    # + a few minutes of DMRG each. MPO pickles (~1.3 GB each) are cached in results_cutoff_0.001/.
    time python -u ground_states.py --hamiltonian-dir ../hamiltonians --output-dir ground_states

    echo "=== $(date): shot counts (max MPO bond $MAX_MPO_BOND) ==="
    time python -u compare_hamiltonians.py \
        --hamiltonian-dir ../hamiltonians \
        --ground-state-dir ground_states \
        --max-mpo-bond "$MAX_MPO_BOND" \
        --nworkers "$NWORKERS" \
        --nworkers-singles "$NWORKERS_SINGLES"

    echo "=== $(date): done ==="
} 2>&1 | tee "$LOG"
