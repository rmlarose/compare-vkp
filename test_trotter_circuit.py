import unittest
import cirq
import trotter_circuit

class TestDiagonalStrings(unittest.TestCase):

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
        target_circuit.append(cirq.rz(theta).on(q))
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
        target_circuit.append(cirq.rz(theta * ps.coefficient).on(qs[0]))
        target_circuit.append(cirq.CNOT(qs[1], qs[0]))
        self.assertEqual(generated_circuit, target_circuit)

class TestNonDiagonalStrings(unittest.TestCase):

    def test_single_x(self):
        """When we have a single X, we should get an X rotation flanked by Hadamards."""

        q = cirq.LineQubit(0)
        ps = 1.0 * cirq.X(q)
        theta = 0.1
        generated_circuit = trotter_circuit.commuting_group_exponential_circuit([ps], theta)
        target_circuit = cirq.Circuit()
        target_circuit.append(cirq.H(q))
        target_circuit.append(cirq.rz(theta).on(q))
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
        target_circuit.append(cirq.rz(theta * ps.coefficient).on(qs[0]))
        target_circuit.append(cirq.CNOT(qs[1], qs[0]))
        target_circuit.append(cirq.H(qs[0]))
        target_circuit.append(cirq.H(qs[1]))
        self.assertEqual(generated_circuit, target_circuit)

if __name__ == '__main__':
    unittest.main()