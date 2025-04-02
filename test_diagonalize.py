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


class TestGramSchmidt(unittest.TestCase):
    """Test the Gram-Schmidt procedure on simple cases."""

    def test_xxi_ixx(self):
        """Test the set {XXI, IXX}, which should leave it untouched
        (the two are linearly independent)."""

        stabilizer_matrix = np.zeros((6, 2), dtype=bool)
        stabilizer_matrix[3, 0] = True
        stabilizer_matrix[4, 0] = True
        stabilizer_matrix[4, 1] = True
        stabilizer_matrix[5, 1] = True
        independent_indices = diagonalize.get_linearly_independent_set(stabilizer_matrix)
        self.assertTrue(independent_indices == [0, 1])

    def test_xxi_ixx_xxx(self):
        """Test the set {XXI, IXX, XXX}. The matrix should be the same.
        N.b. XXI + IXX = IXI."""

        stabilizer_matrix = np.zeros((6, 3), dtype=bool)
        stabilizer_matrix[3, 0] = True
        stabilizer_matrix[4, 0] = True
        stabilizer_matrix[4, 1] = True
        stabilizer_matrix[5, 1] = True
        stabilizer_matrix[3, 2] = True
        stabilizer_matrix[4, 2] = True
        stabilizer_matrix[5, 2] = True
        independent_indices = diagonalize.get_linearly_independent_set(stabilizer_matrix)
        self.assertTrue(independent_indices == [0, 1, 2])

    def test_xii_iix_xix(self):
        """Test the set {XII, IIX, XIX}. The last column should go away."""

        stabilizer_matrix = np.zeros((6, 3), dtype=bool)
        stabilizer_matrix[3, 0] = True
        stabilizer_matrix[5, 1] = True
        stabilizer_matrix[3, 2] = True
        stabilizer_matrix[5, 2] = True
        independent_indices = diagonalize.get_linearly_independent_set(stabilizer_matrix)
        target_indices = [0, 1]
        self.assertTrue(independent_indices == target_indices)
    
    def test_ii_iz(self):
        """If a vector of all identity is in the matrix, this is just a matrix of zeros.
        It should be thrown out."""

        stabilizer_matrix = np.zeros((4, 2), dtype=bool)
        stabilizer_matrix[1, 1] = True
        independent_indices = diagonalize.get_linearly_independent_set(stabilizer_matrix)
        target_indices = [1]
        self.assertTrue(independent_indices == target_indices)
    
    def test_x_y_z(self):
        """If we take the set {X, Y, Z}, the second one should be thrown out (Y prop. XZ)."""

        stabilizer_matrix = np.zeros((2, 3), dtype=bool)
        stabilizer_matrix[1, 0] = True
        stabilizer_matrix[0, 1] = True
        stabilizer_matrix[0, 2] = True
        stabilizer_matrix[1, 2] = True
        independent_indices = diagonalize.get_linearly_independent_set(stabilizer_matrix)
        self.assertEqual(independent_indices, [0, 1])


if __name__ == "__main__":
    unittest.main()