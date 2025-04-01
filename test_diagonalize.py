import unittest
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


if __name__ == "__main__":
    unittest.main()