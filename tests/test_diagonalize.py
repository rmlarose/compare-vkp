import unittest
import numpy as np
import cirq
import diagonalize

class TestSigns(unittest.TestCase):
    """Test if we get the correct signs for groups of Paulis."""

    def test_yi_iz(self):
        """Test the group {YI, IZ}. The first should have sign -1 (True),
        and the second should have sign +1 (False)."""

        qs = cirq.LineQubit.range(2)
        ps1 = 1.0 * cirq.Y(qs[0])
        ps2 = 1.0 * cirq.Z(qs[1])
        paulis = [ps1, ps2]
        stabilizer_matrix = diagonalize.get_stabilizer_matrix_from_paulis(paulis, qs)
        signs = diagonalize.get_stabilizer_matrix_signs(stabilizer_matrix)
        target_signs = [True, False]
        self.assertEqual(target_signs, signs)

    def test_yy(self):
        """If we pass in YY, we should get the sign +1 (False),
        because each Y contributes a -1 sign, and they are mutliplied together."""

        qs = cirq.LineQubit.range(2)
        ps = 1.0 * cirq.Y(qs[0]) * cirq.Y(qs[1])
        stabilizer_matrix = diagonalize.get_stabilizer_matrix_from_paulis([ps], qs)
        signs = diagonalize.get_stabilizer_matrix_signs(stabilizer_matrix)
        target_signs = [False]
        self.assertEqual(target_signs, signs)

    def test_yyiy(self):
        """If we pass in YYIY, we should get the sign -1 (True),
        because each Y contributes a -1 sign, and they are mutliplied together."""

        qs = cirq.LineQubit.range(4)
        ps = 1.0 * cirq.Y(qs[0]) * cirq.Y(qs[1]) * cirq.Y(qs[3])
        stabilizer_matrix = diagonalize.get_stabilizer_matrix_from_paulis([ps], qs)
        signs = diagonalize.get_stabilizer_matrix_signs(stabilizer_matrix)
        target_signs = [True]
        self.assertEqual(target_signs, signs)


class TestBinaryGaussian(unittest.TestCase):

    def test_id(self):
        """The identity matrix should remain invariant under elimination."""

        input_matrix = np.eye(3).astype(bool)
        output_matrix = diagonalize.binary_gaussian_elimination(input_matrix)
        test_matrix = input_matrix.copy()
        self.assertTrue(np.all(output_matrix == test_matrix))
    
    def test_110_011_101(self):
        """If the matrix has columns 110, 011, and 101, the pivots
        should be at (0, 0) and (1, 1), with (2, 2) being False."""

        input_matrix = np.array([
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1]
        ])
        output_matrix = diagonalize.binary_gaussian_elimination(input_matrix)
        # Check that the first two columns contain the first two pivots, but that
        # there is not pivot in the third column.
        pivots = output_matrix[0, 0] and output_matrix[1, 1] and not output_matrix[2, 2]
        # Check that we are in RREF.
        zeros = not output_matrix[1, 0] and not output_matrix[2, 0] and not output_matrix[2, 1]
        self.assertTrue(pivots and zeros)


class TestLinearlyIndependentSet(unittest.TestCase):

    def test_xx_yy_zz(self):
        """If we put in {XX, YY, ZZ}, we should only get XX and YY."""

        qs = cirq.LineQubit.range(2)
        p1 = 1.0 * cirq.X(qs[0]) * cirq.X(qs[1])
        p2 = 1.0 * cirq.Y(qs[0]) * cirq.Y(qs[1])
        p3 = 1.0 * cirq.Z(qs[0]) * cirq.Z(qs[1])
        ps = [p1, p2, p3]
        stabilizer_matrix = diagonalize.get_stabilizer_matrix_from_paulis(ps, qs)
        independent_columns = diagonalize.get_linearly_independent_set(stabilizer_matrix.astype(bool))
        independent_strings = diagonalize.get_paulis_from_stabilizer_matrix(independent_columns)
        self.assertTrue({p1, p2} == set(independent_strings))

    def test_crawford(self):
        """Test the example from Crawford pg. 15"""

        qs = cirq.LineQubit.range(4)
        p1 = 1.0 * cirq.Z(qs[0]) * cirq.Z(qs[1]) * cirq.Z(qs[2]) * cirq.Z(qs[3])
        p2 = 1.0 * cirq.X(qs[0]) * cirq.X(qs[1]) * cirq.Y(qs[2]) * cirq.Y(qs[3])
        p3 = 1.0 * cirq.Y(qs[0]) * cirq.Y(qs[1]) * cirq.X(qs[2]) * cirq.X(qs[3])
        p4 = 1.0 * cirq.Y(qs[1]) * cirq.X(qs[2])
        p5 = 1.0 * cirq.Y(qs[0]) * cirq.X(qs[3])
        p6 = 1.0 * cirq.X(qs[0]) * cirq.Z(qs[1]) * cirq.Z(qs[2]) * cirq.Y(qs[3])
        ps = [p1, p2, p3, p4, p5, p6]
        stabilizer_matrix = diagonalize.get_stabilizer_matrix_from_paulis(ps, qs)
        # reduced_matrix = diagonalize.binary_gaussian_elimination(stabilizer_matrix.astype(bool))
        independent_columns = diagonalize.get_linearly_independent_set(stabilizer_matrix)
        independent_strings = diagonalize.get_paulis_from_stabilizer_matrix(independent_columns)
        # for ps in independent_strings:
        #     print(ps)
        self.assertTrue({p1, p2, p4} == set(independent_strings))


class TestIsDiagonal(unittest.TestCase):

    def test_i(self):
        """I should be diagonal."""

        ps = cirq.PauliString()
        self.assertTrue(diagonalize.is_pauli_diagonal(ps))

    def test_z(self):
        """Z should be diagonal."""

        q = cirq.LineQubit(3)
        ps = 1.0 * cirq.Z(q)
        self.assertTrue(diagonalize.is_pauli_diagonal(ps))

    def test_x(self):
        """X should not be diagonal."""

        q = cirq.LineQubit(3)
        ps = 1.0 * cirq.X(q)
        self.assertFalse(diagonalize.is_pauli_diagonal(ps))

    def test_y(self):
        """Y should not be diagonal."""

        q = cirq.LineQubit(3)
        ps = 1.0 * cirq.Y(q)
        self.assertFalse(diagonalize.is_pauli_diagonal(ps))

    def test_zizz(self):
        """ZIZZ should be diagonal."""

        qs = cirq.LineQubit.range(4)
        ps = 1.0 * cirq.Z(qs[0]) * cirq.Z(qs[2]) * cirq.Z(qs[3])
        self.assertTrue(diagonalize.is_pauli_diagonal(ps))

    def test_xyz(self):
        """XYZ should not be diagonal."""

        qs = cirq.LineQubit.range(3)
        ps = 1.0 * cirq.X(qs[0]) * cirq.Y(qs[1]) * cirq.Z(qs[2])
        self.assertFalse(diagonalize.is_pauli_diagonal(ps))




if __name__ == "__main__":
    unittest.main()