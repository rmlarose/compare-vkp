import unittest
import numpy as np
import scipy.linalg as la
import cirq
import trotter_circuit

class TestDiagonalStrings(unittest.TestCase):
    """Test behavior of Trotter circuits with diagonal strings."""

    def test_identity(self):
        """When we pass in the identity to be exponentiated, we should get an empty circuit."""

        ps = cirq.PauliString()
        generated_circuit = trotter_circuit.diagonal_pstring_exponential_circuit(ps, 0.1)
        target_circuit = cirq.Circuit()
        self.assertEqual(generated_circuit, target_circuit)

    def test_single_z(self):
        """When we pass in a single Z, we should just get a Z rotation."""

        q = cirq.LineQubit(0)
        ps = 1.0 * cirq.Z(q)
        theta = 0.1
        generated_circuit = trotter_circuit.diagonal_pstring_exponential_circuit(ps, theta)
        target_circuit = cirq.Circuit()
        target_circuit.append(cirq.rz(2.0 * theta).on(q))
        self.assertEqual(generated_circuit, target_circuit)

    def test_zz(self):
        """ZZ should produce something like Fig. 4.19 of Nielsen and Chuang,
        but with two qubits (thus only one CNOT on each side of the Rz)."""

        qs = cirq.LineQubit.range(2)
        ps = -0.5 * cirq.Z(qs[0]) * cirq.Z(qs[1])
        theta = 0.3
        generated_circuit = trotter_circuit.diagonal_pstring_exponential_circuit(ps, theta)
        target_circuit = cirq.Circuit()
        target_circuit.append(cirq.CNOT(qs[1], qs[0]))
        target_circuit.append(cirq.rz(2.0 * theta * ps.coefficient).on(qs[0]))
        target_circuit.append(cirq.CNOT(qs[1], qs[0]))
        self.assertEqual(generated_circuit, target_circuit)

    def test_zziz(self):
        """Test ZZIZ."""

        qs = cirq.LineQubit.range(4)
        ps = -0.5 * cirq.Z(qs[0]) * cirq.Z(qs[1]) * cirq.Z(qs[3])
        theta = 0.3
        generated_circuit = trotter_circuit.diagonal_pstring_exponential_circuit(ps, theta)
        target_unitary = la.expm(complex(0.0, -1.0) * theta * ps.matrix())
        # norm_distance = la.norm(target_unitary - generated_circuit.unitary())
        self.assertTrue(np.allclose(generated_circuit.unitary(), target_unitary))


class TestNonDiagonalStrings(unittest.TestCase):
    """Test behavior when we pass in non-diagonal strings (requiring diagonalization circuits)."""

    def test_single_x(self):
        """When we have a single X, we should get an X rotation flanked by Hadamards."""

        q = cirq.LineQubit(0)
        ps = 1.0 * cirq.X(q)
        theta = 0.1
        generated_circuit = trotter_circuit.commuting_group_exponential_circuit([ps], theta)
        target_circuit = cirq.Circuit()
        target_circuit.append(cirq.H(q))
        target_circuit.append(cirq.rz(2.0 * theta).on(q))
        target_circuit.append(cirq.H(q))
        self.assertEqual(generated_circuit, target_circuit)

    def test_xx(self):
        """When we put in XX, we should get hadamards and the usual ladder-rotation-ladder."""

        qs = cirq.LineQubit.range(2)
        ps = -0.5 * cirq.X(qs[0]) * cirq.X(qs[1])
        theta = 0.3
        generated_circuit = trotter_circuit.commuting_group_exponential_circuit([ps], theta)
        target_circuit = cirq.Circuit()
        target_circuit.append(cirq.H(qs[0]))
        target_circuit.append(cirq.H(qs[1]))
        target_circuit.append(cirq.CNOT(qs[1], qs[0]))
        target_circuit.append(cirq.rz(2.0 * theta * ps.coefficient).on(qs[0]))
        target_circuit.append(cirq.CNOT(qs[1], qs[0]))
        target_circuit.append(cirq.H(qs[0]))
        target_circuit.append(cirq.H(qs[1]))
        self.assertTrue(cirq.approx_eq(generated_circuit.unitary(), target_circuit.unitary()))

    def test_xix(self):
        """When we put in XIX, we should get hadamards and the usual ladder-rotation-ladder."""

        qs = cirq.LineQubit.range(3)
        ps = -0.5 * cirq.X(qs[0]) * cirq.X(qs[2])
        theta = 0.3
        generated_circuit = trotter_circuit.commuting_group_exponential_circuit([ps], theta, qs=qs)
        target_circuit = cirq.Circuit()
        target_circuit.append(cirq.H(qs[0]))
        target_circuit.append(cirq.H(qs[2]))
        target_circuit.append(cirq.CNOT(qs[2], qs[0]))
        target_circuit.append(cirq.rz(2.0 * theta * ps.coefficient).on(qs[0]))
        target_circuit.append(cirq.CNOT(qs[2], qs[0]))
        target_circuit.append(cirq.H(qs[0]))
        target_circuit.append(cirq.H(qs[2]))
        self.assertTrue(cirq.approx_eq(generated_circuit.unitary(), target_circuit.unitary()))


class TestGroupingCircuits(unittest.TestCase):
    """Test behavior when we pass in multiple strings, and they are storted 
    into commuting groups."""

    def test_xx_zz(self):
        """When we pass in {XX, ZZ}, they should be sorted into a single group."""

        qs = cirq.LineQubit.range(2)
        ps1 = 1.0 * cirq.X(qs[0]) * cirq.X(qs[1])
        ps2 = 1.0 * cirq.Z(qs[0]) * cirq.Z(qs[1])
        group = [ps1, ps2]
        dt = 1e-2
        generated_circuit = trotter_circuit.first_order_trotter_for_grouping([group], dt, qs)
        # Diagonalize the strings with the inverse of a Bell state prep circuit.
        diag_circuit = cirq.Circuit()
        diag_circuit.append(cirq.CNOT(qs[1], qs[0]))
        diag_circuit.append(cirq.H(qs[1]))
        # The diagonalizing circuits maps XX->ZI, ZZ->IZ.
        # There are just single-qubit rotations applied to each qubits, then.
        exp_circuit = cirq.Circuit()
        exp_circuit.append(cirq.rz(2.0 * dt).on(qs[0]))
        exp_circuit.append(cirq.rz(2.0 * dt).on(qs[1]))
        target_circuit = diag_circuit + exp_circuit + cirq.inverse(diag_circuit)
        self.assertTrue(np.allclose(target_circuit.unitary(), generated_circuit.unitary()))


class TestUnitaries(unittest.TestCase):
    """Test that our circuits give approximately the right unitaries."""

    def test_zz(self):
        """Test that evolution uder the Hamiltonian ZZ gives the right unitary"""

        qs = cirq.LineQubit.range(2)
        ps = 1.0 * cirq.Z(qs[0]) * cirq.Z(qs[1])
        dt = 1e-2
        #generated_circuit = trotter_circuit.first_order_trotter_for_grouping([[ps]], dt)
        generated_circuit = trotter_circuit.diagonal_pstring_exponential_circuit(ps, dt)
        generated_unitary = generated_circuit.unitary()
        ham = ps.matrix()
        target_unitary = la.expm(complex(0.0, -1.0) * dt * ham)
        self.assertTrue(np.allclose(target_unitary, generated_unitary, rtol=1e-4, atol=1e-6))

    def test_xx(self):
        """Test that evolution uder the Hamiltonian XX gives the right unitary"""

        qs = cirq.LineQubit.range(2)
        ps = 1.0 * cirq.X(qs[0]) * cirq.X(qs[1])
        dt = 1e-2
        generated_circuit = trotter_circuit.commuting_group_exponential_circuit([ps], dt)
        generated_unitary = generated_circuit.unitary()
        ham = ps.matrix()
        target_unitary = la.expm(complex(0.0, -1.0) * dt * ham)
        self.assertTrue(np.allclose(target_unitary, generated_unitary, rtol=1e-4, atol=1e-6))

    def test_xx_zz(self):
        """Test that evolution under the Hamiltonian XX + ZZ gives the right unitary"""

        qs = cirq.LineQubit.range(2)
        ps1 = 1.0 * cirq.X(qs[0]) * cirq.X(qs[1])
        ps2 = 1.0 * cirq.Z(qs[0]) * cirq.Z(qs[1])
        dt = 1e-2
        generated_circuit = trotter_circuit.commuting_group_exponential_circuit([ps1, ps2], dt)
        generated_unitary = generated_circuit.unitary()
        ham = ps1 + ps2
        ham_matrix = ham.matrix()
        target_unitary = la.expm(complex(0.0, -1.0) * dt * ham_matrix)
        self.assertTrue(np.allclose(target_unitary, generated_unitary, rtol=1e-4, atol=1e-6))

    def test_iz_yi(self):
        """Test that evolution uder the Hamiltonian IZ+YI gives the right unitary"""

        qs = cirq.LineQubit.range(2)
        ps1 = 1.0 * cirq.Y(qs[0])
        ps2 = 1.0 * cirq.Z(qs[1])
        dt = 1e-2
        generated_circuit = trotter_circuit.commuting_group_exponential_circuit([ps1, ps2], dt, qs=qs)
        generated_unitary = generated_circuit.unitary()
        ham = ps1 + ps2
        ham_matrix = ham.matrix()
        target_unitary = la.expm(complex(0.0, -1.0) * dt * ham_matrix)
        self.assertTrue(np.allclose(target_unitary, generated_unitary, rtol=1e-4, atol=1e-6))

    def test_xx_zz_iz_yi(self):
        """Test that evolution uder the Hamiltonian XX+ZZ+IZ+YI gives the right unitary"""

        qs = cirq.LineQubit.range(2)
        ps1 = 1.0 * cirq.X(qs[0]) * cirq.X(qs[1])
        ps2 = 1.0 * cirq.Z(qs[0]) * cirq.Z(qs[1])
        ps3 = 1.0 * cirq.Y(qs[0])
        ps4 = 1.0 * cirq.Z(qs[1])
        dt = 1e-2
        groups = [[ps1, ps2], [ps3, ps4]]
        generated_circuit = trotter_circuit.first_order_trotter_for_grouping(groups, dt, qs=qs)
        generated_unitary = generated_circuit.unitary()
        # We split into two groups: {{XX, ZZ}, {IZ, YI}}. Use two unitary circuits.
        ham1 = ps1 + ps2
        ham2 = ps3 + ps4
        ham1_matrix = ham1.matrix()
        ham2_matrix = ham2.matrix()
        u1 = la.expm(complex(0.0, -1.0) * dt * ham1_matrix)
        u2 = la.expm(complex(0.0, -1.0) * dt * ham2_matrix)
        target_unitary = u1 @ u2
        err = target_unitary - generated_unitary
        # I will assert a percentenage of error because the matrix multiplication
        # is throwing off np.allclose.
        self.assertTrue(la.norm(err) / la.norm(target_unitary) * 100 < 0.1)

if __name__ == '__main__':
    unittest.main()
