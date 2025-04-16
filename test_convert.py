import unittest
import cirq
import qiskit
from convert import cirq_circuit_to_qiskit

class TestCirqToQiskit(unittest.TestCase):

    def test_bell_circuit(self):
        """Test converting a Bell circuit."""

        qs = cirq.LineQubit.range(2)
        cirq_circuit = cirq.Circuit()
        cirq_circuit.append(cirq.H(qs[0]))
        cirq_circuit.append(cirq.CNOT(qs[0], qs[1]))
        qiskit_circuit = qiskit.QuantumCircuit(2)
        qiskit_circuit.h(0)
        qiskit_circuit.cx(0, 1)
        generated_circuit = cirq_circuit_to_qiskit(cirq_circuit)
        self.assertEqual(generated_circuit, qiskit_circuit)

if __name__ == "__main__":
    unittest.main()
