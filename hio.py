"""We will try the HIO algorithm from https://arxiv.org/abs/2410.21517.
As a candidate problem, let's try estimating the phase field of
<psi| e^{-i ZZ r} e^{-i XX s} |psi>, where r and s are real numbers,
and psi is some random state."""

from typing import Tuple, List
import h5py
import numpy as np
import cirq
from scipy import fft
import scipy.linalg as la

def hybrid_input_output(
    abs_f: np.ndarray, beta: float, L: int, F: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Iteratively predicts phases from the absolute values.
    
    Arguments:
    abs_f: The 2D field |f| as a matrix.
    beta: Regularization parameter.
    L: Number of iterations.
    F: Initial guess as the spectrum.
    
    Returns:
    f: The field with the phase."""

    f_tilde_old = np.zeros(abs_f.shape)
    deltas: List[float] = [] # ||f_tilde - f_tilde_old||
    for i in range(L):
        f = fft.ifft2(F)
        f_tilde = abs_f * np.exp(1j * np.angle(f))
        if i != 0:
            deltas.append(la.norm(f_tilde - f_tilde_old))
        F_tilde = np.real(fft.fft2(f_tilde))
        new_F = np.zeros(F_tilde.shape, dtype=complex)
        for k in range(F_tilde.shape[0]):
            for m in range(F_tilde.shape[1]):
                if F_tilde[k, m] <= 0.:
                    new_F[k, m] = F[k, m] - beta * F_tilde[k, m]
                else:
                    new_F[k, m] = F_tilde[k, m]
        f_tilde_old = f_tilde.copy()
    return (f_tilde, np.array(deltas))


def main():
    # Generate a random two-qubit state.
    qs = cirq.LineQubit.range(2)
    nq = len(qs)
    psi_real = np.random.rand(2 ** nq).astype(complex)
    psi_imag = np.random.rand(2 ** nq).astype(complex)
    psi = psi_real + 1j * psi_imag
    psi = psi / la.norm(psi)

    # Generate the M * N array of amplitudes.
    M = 10 # Number of values of r.
    N = 10 # Number of values of s.
    dr = 0.1
    ds = 0.1
    f = np.zeros((M, N), dtype=complex)
    for i in range(M):
        for j in range(N):
            r = i * dr - (float(M) * dr / 2)
            s = j * ds - (float(N) * ds / 2)
            xx = 1.0 * cirq.X.on(qs[0]) * cirq.X.on(qs[1])
            zz = 1.0 * cirq.Z.on(qs[0]) * cirq.Z.on(qs[1])
            u1 = la.expm(-1j * cirq.unitary(xx) * r)
            u2 = la.expm(-1j * cirq.unitary(zz) * s)
            f[i, j] = psi.conj().T @ u1 @ u2 @ psi
    
    abs_f = np.abs(f)
    F_real = fft.fft2(f)
    F = F_real.copy() + 1e-2 * np.random.rand(*F_real.shape)
    beta = 0.1
    L = 100
    f_hio, deltas = hybrid_input_output(abs_f, beta, L, F)
    
    fp = h5py.File("data/phases.hdf5", "w")
    fp.create_dataset("amplitudes", data=f)
    fp.create_dataset("F_real", data=F_real)
    fp.create_dataset("amplitudes_hio", data=f_hio)
    fp.create_dataset("delta", data=deltas)
    fp.close()

if __name__ == "__main__":
    main()
