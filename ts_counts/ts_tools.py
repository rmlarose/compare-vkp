"""Fast tools for the transition-state shot-count workflow.

* k-commuting sorted-insertion grouping on bit-packed Pauli strings (numba),
  equivalent to ``kcommute.get_si_sets`` but fast enough for ~10^5 terms.
* Conversion of (groups of) OpenFermion QubitOperators to quimb MPOs.
* Group variances with respect to an MPS, and the resulting shot counts.

Pauli strings on n <= 64 qubits are encoded as a pair of uint64 bit masks
(x, z): X -> (1, 0), Y -> (1, 1), Z -> (0, 1). Qubit i is bit i.
"""

from typing import Dict, List, Sequence, Tuple

import numba
import numpy as np
import openfermion as of
import quimb.tensor as qtn
from quimb.operator import HilbertSpace, SparseOperatorBuilder

PauliTerm = Tuple[Tuple[int, str], ...]  # OpenFermion term key, e.g. ((0, 'X'), (3, 'Z'))


# ----------------------------------------------------------------------------
# Encoding.
# ----------------------------------------------------------------------------

def qubit_operator_to_arrays(
    qubop: of.QubitOperator,
) -> Tuple[List[PauliTerm], np.ndarray, np.ndarray, np.ndarray]:
    """Returns (terms, coeffs, x, z) for the non-identity terms of `qubop`,
    sorted by decreasing |coefficient| (the sorted-insertion order)."""
    items = [(t, c) for t, c in qubop.terms.items() if t != ()]
    items.sort(key=lambda tc: abs(tc[1]), reverse=True)
    terms = [t for t, _ in items]
    coeffs = np.array([c for _, c in items])
    x = np.zeros(len(terms), dtype=np.uint64)
    z = np.zeros(len(terms), dtype=np.uint64)
    for i, term in enumerate(terms):
        xi = zi = 0
        for q, p in term:
            if p in ("X", "Y"):
                xi |= 1 << q
            if p in ("Z", "Y"):
                zi |= 1 << q
        x[i] = xi
        z[i] = zi
    return terms, coeffs, x, z


# ----------------------------------------------------------------------------
# k-commuting sorted insertion.
# ----------------------------------------------------------------------------

@numba.njit(cache=True)
def _prefix_parity(m):
    # Bit i of the result is the parity of bits 0..i of m.
    m ^= m << np.uint64(1)
    m ^= m << np.uint64(2)
    m ^= m << np.uint64(4)
    m ^= m << np.uint64(8)
    m ^= m << np.uint64(16)
    m ^= m << np.uint64(32)
    return m


@numba.njit(cache=True)
def _k_commute(x1, z1, x2, z2, block_ends):
    """True iff the two Paulis commute on every block. Blocks are contiguous
    ranges of qubits; block_ends holds the last qubit index of each block."""
    m = (x1 & z2) ^ (z1 & x2)  # Bit i set <=> the Paulis anticommute on qubit i.
    if m == 0:
        return True
    p = _prefix_parity(m)
    prev = np.uint64(0)
    for e in block_ends:
        cur = (p >> np.uint64(e)) & np.uint64(1)
        if cur != prev:  # Odd number of anticommuting sites in this block.
            return False
        prev = cur
    return True


@numba.njit(cache=True)
def _sorted_insertion(x, z, block_ends):
    n = x.shape[0]
    group_of = np.empty(n, dtype=np.int64)
    # Members of each group are stored as a linked list through `nxt`.
    head = np.empty(n, dtype=np.int64)
    tail = np.empty(n, dtype=np.int64)
    nxt = np.full(n, -1, dtype=np.int64)
    ngroups = 0
    for t in range(n):
        placed = False
        for g in range(ngroups):
            ok = True
            j = head[g]
            while j != -1:
                if not _k_commute(x[t], z[t], x[j], z[j], block_ends):
                    ok = False
                    break
                j = nxt[j]
            if ok:
                nxt[tail[g]] = t
                tail[g] = t
                group_of[t] = g
                placed = True
                break
        if not placed:
            head[ngroups] = t
            tail[ngroups] = t
            group_of[t] = ngroups
            ngroups += 1
    return group_of, ngroups


def block_ends_for(nqubits: int, k: int) -> np.ndarray:
    """Last qubit of each block, matching kcommute.compute_blocks (contiguous chunks of size k)."""
    return np.array([min(k * (b + 1), nqubits) - 1 for b in range(-(-nqubits // k))], dtype=np.int64)


def k_commuting_groups(x: np.ndarray, z: np.ndarray, nqubits: int, k: int) -> List[np.ndarray]:
    """Sorted-insertion k-commuting grouping. Terms must already be sorted by
    decreasing |coefficient|. Returns a list of index arrays, one per group."""
    if nqubits > 64:
        raise ValueError("Bit-packed encoding supports at most 64 qubits.")
    group_of, ngroups = _sorted_insertion(x, z, block_ends_for(nqubits, k))
    order = np.argsort(group_of, kind="stable")
    splits = np.cumsum(np.bincount(group_of, minlength=ngroups))[:-1]
    return np.split(order, splits)


# ----------------------------------------------------------------------------
# MPOs and variances.
# ----------------------------------------------------------------------------

def terms_to_mpo(
    terms: Sequence[PauliTerm], coeffs: Sequence[complex], nqubits: int, constant: float = 0.0
) -> qtn.MatrixProductOperator:
    """Builds an exact MPO (via quimb's SparseOperatorBuilder) for sum_i coeffs[i] * terms[i]."""
    builder = SparseOperatorBuilder(hilbert_space=HilbertSpace(range(nqubits)))
    if constant != 0.0:
        builder.add_term(constant)
    for term, c in zip(terms, coeffs):
        builder.add_term(c, *[(p.lower(), q) for q, p in term])
    return builder.build_mpo()


def fermion_operator_to_mpo(fermion_op: of.FermionOperator, nqubits: int) -> qtn.MatrixProductOperator:
    """MPO of jordan_wigner(fermion_op) (OpenFermion convention, verified
    against openfermion.get_sparse_operator), built from the fermionic terms so
    the MPO bond dimension is much smaller than building from Pauli strings.
    The constant term is dropped."""
    builder = SparseOperatorBuilder(hilbert_space=HilbertSpace(range(nqubits)), jordan_wigner=True)
    for term, c in fermion_op.terms.items():
        if term == ():
            continue
        builder.add_term(c, *[("+" if action else "-", q) for q, action in term])
    return builder.build_mpo()


def number_penalty_mpo(nqubits: int, nelec: int, mu: float) -> qtn.MatrixProductOperator:
    """MPO for mu * (N - nelec)^2 with N = sum_i (1 - Z_i) / 2 (JW, |1> = occupied).

    Adding this to the Hamiltonian keeps DMRG (which has no symmetry here) in the
    nelec-electron sector. Since [H, N] = 0, eigenstates in that sector are
    unchanged."""
    # (N - n0)^2 = (a - S/2)^2 with S = sum_i Z_i, a = nqubits/2 - n0.
    a = nqubits / 2 - nelec
    builder = SparseOperatorBuilder(hilbert_space=HilbertSpace(range(nqubits)))
    builder.add_term(mu * (a**2 + nqubits / 4))  # Includes the Z_i^2 = I diagonal of S^2 / 4.
    for i in range(nqubits):
        builder.add_term(-mu * a, ("z", i))
        for j in range(i + 1, nqubits):
            builder.add_term(mu / 2, ("z", i), ("z", j))
    return builder.build_mpo()


def mps_to_arrays(psi: qtn.MatrixProductState) -> List[np.ndarray]:
    """Site arrays of an MPS as (left, phys, right), boundary bonds of size 1."""
    L = psi.L
    arrays = []
    for i in range(L):
        t = psi[i]
        inds = []
        if i > 0:
            inds.append(psi.bond(i - 1, i))
        inds.append(psi.site_ind(i))
        if i < L - 1:
            inds.append(psi.bond(i, i + 1))
        a = t.transpose(*inds).data
        if i == 0:
            a = a[None]
        if i == L - 1:
            a = a[..., None]
        arrays.append(np.asarray(a))
    return arrays


def mpo_to_arrays(mpo: qtn.MatrixProductOperator) -> List[np.ndarray]:
    """Site arrays of an MPO as (left, out, in, right), boundary bonds of size 1."""
    L = mpo.L
    arrays = []
    for i in range(L):
        t = mpo[i]
        inds = []
        if i > 0:
            inds.append(mpo.bond(i - 1, i))
        inds += [mpo.upper_ind(i), mpo.lower_ind(i)]
        if i < L - 1:
            inds.append(mpo.bond(i, i + 1))
        w = t.transpose(*inds).data
        if i == 0:
            w = w[None]
        if i == L - 1:
            w = w[..., None]
        arrays.append(np.asarray(w))
    return arrays


def expectation_and_square(
    psi_arrays: List[np.ndarray], w_arrays: List[np.ndarray], slice_size: int = 8
) -> Tuple[complex, complex]:
    """Returns (<psi|O|psi>, <psi|O^2|psi>) by left-to-right environment sweeps.

    The <O^2> update is done in slices of `slice_size` along the bra bond, accumulating
    into the new environment. This is exact (same arithmetic) but keeps the 5-index
    intermediates ~(bra bond / slice_size) times smaller: for MPS bond 64 and MPO bond 96
    the unsliced update peaks at ~9 GB."""
    E1 = np.ones((1, 1, 1))  # (bra, mpo, ket)
    E2 = np.ones((1, 1, 1, 1))  # (bra, mpo (outer), mpo (inner), ket)
    for A, W in zip(psi_arrays, w_arrays):
        Ac = A.conj()
        # <O>
        T = np.tensordot(E1, A, axes=([2], [0]))  # a w t d
        T = np.tensordot(T, W, axes=([1, 2], [0, 2]))  # a d u v
        E1 = np.tensordot(Ac, T, axes=([0, 1], [0, 2]))  # c d v -> reorder
        E1 = E1.transpose(0, 2, 1)
        # <O^2>
        dtype = np.result_type(E2, A, W)
        E2_new = np.zeros((A.shape[2], W.shape[3], W.shape[3], A.shape[2]), dtype=dtype)
        for s in range(0, E2.shape[0], slice_size):
            T = np.tensordot(E2[s : s + slice_size], A, axes=([3], [0]))  # a w x t d
            T = np.tensordot(T, W, axes=([2, 3], [0, 2]))  # a w d u y   (inner MPO applied to ket)
            T = np.tensordot(T, W, axes=([1, 3], [0, 2]))  # a d y s z   (outer MPO)
            E2_new += np.tensordot(Ac[s : s + slice_size], T, axes=([0, 1], [0, 3])).transpose(0, 3, 2, 1)
            del T
        E2 = E2_new
    return complex(E1.ravel()[0]), complex(E2.ravel()[0])


_PAULI = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_expectation(psi_arrays: List[np.ndarray], term: PauliTerm, right_canonical: bool = False) -> complex:
    """<psi|P|psi> for a single Pauli string.

    If `right_canonical` is True, psi_arrays must be in right-canonical form;
    the contraction then stops at the last non-identity site (the rest of the
    chain contracts to the identity), which roughly halves the cost."""
    ops = dict(term)
    last = max(ops) if (right_canonical and ops) else len(psi_arrays) - 1
    E = np.ones((1, 1))  # (bra, ket)
    for i in range(last + 1):
        A = psi_arrays[i]
        B = np.tensordot(_PAULI[ops[i]], A, axes=([1], [1])).transpose(1, 0, 2) if i in ops else A
        E = np.tensordot(np.tensordot(E, B, axes=([1], [0])), A.conj(), axes=([0, 1], [0, 1])).T
    return complex(np.trace(E)) if right_canonical else complex(E.ravel()[0])


def _single_group_variance(group_terms, group_coeffs, nqubits, psi_arrays, max_mpo_bond=None) -> Tuple[float, int]:
    """Returns (variance, exact MPO bond dimension of the group operator).

    If max_mpo_bond is set and the exact MPO is larger, the MPO is SVD-truncated to
    max_mpo_bond before computing the variance (approximate). The memory of the <O^2>
    contraction scales as (MPS bond)^2 * (MPO bond)^2, e.g. ~20 GB for MPO bond ~300."""
    if len(group_terms) == 1:
        # Var(cP) = |c|^2 (1 - <P>^2) since P^2 = I. psi_arrays is right-canonical (see group_variances).
        ev = pauli_expectation(psi_arrays, group_terms[0], right_canonical=True).real
        return max(abs(group_coeffs[0]) ** 2 * (1.0 - ev**2), 0.0), 1
    mpo = terms_to_mpo(group_terms, group_coeffs, nqubits)
    bond = mpo.max_bond()
    if max_mpo_bond is not None and bond > max_mpo_bond:
        mpo.compress(max_bond=max_mpo_bond)
    mean, sq = expectation_and_square(psi_arrays, mpo_to_arrays(mpo))
    return float(np.real(sq - mean**2)), bond


# Per-worker state for the process pool.
_WORKER = {}


def _init_worker(terms, coeffs, nqubits, psi_arrays, max_mpo_bond):
    _WORKER.update(terms=terms, coeffs=coeffs, nqubits=nqubits, psi=psi_arrays, max_mpo_bond=max_mpo_bond)


def _worker_chunk(chunk):
    w = _WORKER
    return [
        _single_group_variance(
            [w["terms"][j] for j in key], [w["coeffs"][j] for j in key], w["nqubits"], w["psi"], w["max_mpo_bond"]
        )
        for key in chunk
    ]


def group_variances(
    groups: List[np.ndarray],
    terms: Sequence[PauliTerm],
    coeffs: np.ndarray,
    nqubits: int,
    psi: qtn.MatrixProductState,
    cache: Dict[Tuple[int, ...], float] = None,
    nworkers: int = None,
    chunksize: int = 50,
    verbose: bool = True,
    max_mpo_bond: int = None,
    bond_cache: Dict[Tuple[int, ...], int] = None,
) -> np.ndarray:
    """Variance of each group's operator w.r.t. psi, computed by MPO
    sandwiches in a process pool. `cache` (keyed by sorted term indices) is
    updated in place, so groups shared between different k (or already
    computed in an earlier, interrupted run) are not recomputed.

    Exact unless `max_mpo_bond` is set: then group MPOs with a larger bond
    dimension are truncated to it (approximate, but bounds the memory). The
    exact MPO bond dimension of each newly computed group is stored in
    `bond_cache` if given. Callers must not mix caches computed with
    different `max_mpo_bond`."""
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if cache is None:
        cache = {}
    keys = [tuple(sorted(int(j) for j in g)) for g in groups]
    todo = list(dict.fromkeys(k for k in keys if k not in cache))
    # Big groups first so the slow ones don't straggle at the end.
    todo.sort(key=len, reverse=True)
    if todo:
        nworkers = nworkers or os.cpu_count()
        chunks = [todo[i : i + chunksize] for i in range(0, len(todo), chunksize)]
        # One BLAS thread per worker; parallelism comes from the pool.
        env_backup = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
        for v in env_backup:
            os.environ[v] = "1"
        t0 = time.time()
        try:
            psi_rc = psi.copy()
            psi_rc.right_canonize()
            psi_arrays = mps_to_arrays(psi_rc)
            with ProcessPoolExecutor(nworkers, initializer=_init_worker, initargs=(list(terms), coeffs, nqubits, psi_arrays, max_mpo_bond)) as ex:
                futures = {ex.submit(_worker_chunk, c): c for c in chunks}
                for n, fut in enumerate(as_completed(futures), 1):
                    for key, (var, bond) in zip(futures[fut], fut.result()):
                        cache[key] = var
                        if bond_cache is not None:
                            bond_cache[key] = bond
                    if verbose:
                        print(f"  chunk {n} / {len(chunks)}  ({time.time() - t0:.0f} s)", end="\r")
        finally:
            for v, val in env_backup.items():
                if val is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = val
        if verbose:
            print()
    return np.array([cache[k] for k in keys])


def shots_from_variances(variances: np.ndarray, epsilon: float) -> float:
    """Shots for precision epsilon with optimal shot allocation: (sum_g sqrt(Var_g))^2 / eps^2."""
    return float(np.sum(np.sqrt(np.clip(variances, 0.0, None))) ** 2 / epsilon**2)


# ----------------------------------------------------------------------------
# Hamiltonian loading and portable ground-state files.
# ----------------------------------------------------------------------------

def load_hamiltonian(path: str) -> Tuple[of.FermionOperator, of.QubitOperator, int]:
    """Returns (fermion Hamiltonian, Jordan-Wigner qubit Hamiltonian, number of electrons).

    ts.npz uses ECORE/H1/H2/NELEC; the product/reactant files use e_nuc/h1/h2/nelectron
    (same chemist-notation H2 and the same role for the constant)."""
    from openfermion.ops.representations import get_tensors_from_integrals

    h_data = np.load(path)
    if "H1" in h_data.files:
        ecore, h1, h2, nelec = h_data["ECORE"], h_data["H1"], h_data["H2"], h_data["NELEC"]
    else:
        ecore, h1, h2, nelec = h_data["e_nuc"], h_data["h1"], h_data["h2"], h_data["nelectron"]
    ecore = float(ecore)  # Stored as a 0-d array; OpenFermion needs a scalar.
    # H2 is in chemist notation, (pq|rs). get_tensors_from_integrals expects
    # OpenFermion's MolecularData ordering, two_body_integrals[p,q,r,s] = (ps|qr).
    h2 = h2.transpose(0, 2, 3, 1)
    one_body_tensor, two_body_tensor = get_tensors_from_integrals(h1, h2)
    hamiltonian_op = of.InteractionOperator(
        constant=ecore, one_body_tensor=one_body_tensor, two_body_tensor=two_body_tensor
    )
    fermion_hamiltonian = of.get_fermion_operator(hamiltonian_op)
    qubit_hamiltonian = of.jordan_wigner(fermion_hamiltonian)
    return fermion_hamiltonian, qubit_hamiltonian, int(nelec)


def save_mps_npz(path: str, psi: qtn.MatrixProductState, **metadata) -> None:
    """Saves an MPS as plain NumPy arrays (site_i with shape (left, phys, right)) plus
    scalar/array metadata. Unlike a pickle of the quimb object, this loads with any
    NumPy/quimb version on any machine."""
    arrays = mps_to_arrays(psi)
    np.savez(
        path,
        nsites=len(arrays),
        **{f"site_{i}": a for i, a in enumerate(arrays)},
        **{k: np.asarray(v) for k, v in metadata.items()},
    )


def load_mps_npz(path: str) -> Tuple[qtn.MatrixProductState, Dict[str, np.ndarray]]:
    """Inverse of save_mps_npz. Returns (psi, metadata)."""
    with np.load(path) as data:
        n = int(data["nsites"])
        arrays = [data[f"site_{i}"] for i in range(n)]
        metadata = {k: data[k] for k in data.files if k != "nsites" and not k.startswith("site_")}
    arrays[0] = arrays[0][0]  # Drop the size-1 boundary bonds.
    arrays[-1] = arrays[-1][..., 0]
    return qtn.MatrixProductState(arrays, shape="lpr"), metadata
