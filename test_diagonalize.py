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

    def test_110_011_101(self):
        """If we put in the binary vectors [110], [011], and [101],
        we know that [110] ^ [011] = [101], so we should eliminate the last row."""

        input_matrix = np.array([[True, True, False], [False, True, True], [True, False, True]])
        output_matrix = diagonalize.binary_gaussian_elimination(input_matrix)
        target_matrix = input_matrix[:2, :]
        self.assertTrue(np.all(target_matrix == output_matrix))

    def test_011_110_101(self):
        """If we put in the binary vectors [011], [110], and [101],
        we know that [110] ^ [011] = [101], so we should eliminate the last row."""

        input_matrix = np.array([[False, True, True], [True, True, False], [True, False, True]])
        output_matrix = diagonalize.binary_gaussian_elimination(input_matrix)
        #print(f"ouptut_matrix=\n", output_matrix)
        target_matrix = np.vstack([input_matrix[1, :], input_matrix[0, :]])
        self.assertTrue(np.all(target_matrix == output_matrix))
    
    def test_100_010_001(self):
        """If we pass in a diagonal matrix, it should remain untouched."""

        input_matrix = np.eye(5).astype(bool)
        output_matrix = diagonalize.binary_gaussian_elimination(input_matrix)
        target_matrix = input_matrix.copy()
        self.assertTrue(np.all(target_matrix == output_matrix))
    
    def test_110_001_111(self):
        """In the set [110], [001], and [111], [111] should be cast out."""

        input_matrix = np.array([[True, True, False], [False, False, True], [True, True, True]])
        output_matrix = diagonalize.binary_gaussian_elimination(input_matrix)
        target_matrix = input_matrix[:2, :]
        self.assertTrue(np.all(target_matrix == output_matrix))
    
    def test_1100_0011_1111(self):
        """The last row should be thrown out!"""

        input_matrix = np.array([
            [True, True, False, False],
            [False, False, True, True,],
            [True, True, True, True]
        ])
        output_matrix = diagonalize.binary_gaussian_elimination(input_matrix)
        target_matrix = input_matrix[:2, :]
        self.assertTrue(np.all(output_matrix == target_matrix))
    
    def test_four_qubit_xx_yy_zz(self):
        """Test the set {X_1 X_2, Y_1 Y_2, Z_1 Z_2, X_3 X_4, Y_3 Y_4, Z_3 Z_4}"""

        input_matrix = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0], # X_1 X_2
            [1, 1, 0, 0, 1, 1, 0, 0], # Y_1 Y_2
            [0, 0, 0, 0, 1, 1, 0, 0], # Z_1 Z_2
            [0, 0, 1, 1, 0, 0, 0, 0], # X_3 X_4
            [0, 0, 1, 1, 0, 0, 1, 1], # Y_3 Y_4
            [0, 0, 0, 0, 0, 0, 1, 1], # Z_3 Z_4
        ]).astype(bool)
        output_matrix = diagonalize.binary_gaussian_elimination(input_matrix)
        test_matrix = np.vstack([input_matrix[0, :], input_matrix[3, :], input_matrix[2, :], input_matrix[5, :]])
        self.assertTrue(np.all(test_matrix == output_matrix))

if __name__ == "__main__":
    unittest.main()