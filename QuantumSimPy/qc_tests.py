import unittest
from ideal_qc import QuantumCircuit
import numpy as np

def prod3(psi1, psi2, psi3):
    psi = np.tensordot(psi1, psi2, axes=0)
    psi = np.tensordot(psi, psi3, axes=0)
    return psi

def prod5(psi1, psi2, psi3, psi4, psi5):
    psi = np.tensordot(psi1, psi2, axes=0)
    psi = np.tensordot(psi, psi3, axes=0)
    psi = np.tensordot(psi, psi4, axes=0)
    psi = np.tensordot(psi, psi5, axes=0)
    return psi

class rxd_gate_almost_ideal(unittest.TestCase):
    def setUp(self):
        self.psi_Z_0 = np.array([1, 0])
        self.psi_Z_1 = np.array([0, 1])
        self.psi_Y_0 = np.array([1, 1j])/np.sqrt(2)
        self.psi_Y_1 = np.array([1, -1j])/np.sqrt(2)
        self.psi_X_0 = np.array([1, 1])/np.sqrt(2)
        self.psi_X_1 = np.array([1, -1])/np.sqrt(2)
        
    def test_one_qubit_Z(self):
        self.qc = QuantumCircuit(1)
        self.qc.rxd([0], np.pi, 0.000001)
        self.assertEqual(np.round(abs(self.qc.run(self.psi_Z_0)[0].conj() @ self.psi_Z_1), 5), 1)
        self.assertEqual(np.round(abs(self.qc.run(self.psi_Z_1)[0].conj() @ self.psi_Z_0), 5), 1)
        self.assertEqual(np.round(abs(self.qc.run(self.psi_X_0)[0].conj() @ self.psi_X_0), 5), 1)
        self.assertEqual(np.round(abs(self.qc.run(self.psi_X_1)[0].conj() @ self.psi_X_1), 5), 1)
        self.assertEqual(np.round(abs(self.qc.run(self.psi_Y_1)[0].conj() @ self.psi_Y_0), 5), 1)
        self.assertEqual(np.round(abs(self.qc.run(self.psi_Y_0)[0].conj() @ self.psi_Y_1), 5), 1)

    def test_three_qubit_cirq(self):

        self.qc = QuantumCircuit(3)
        self.qc.rxd([0, 1, 2], np.pi, 0.000001)
        test_psi = prod3(self.psi_Z_0, self.psi_Y_0, self.psi_X_0)
        xxx_test_psi = prod3(self.psi_Z_1, self.psi_Y_1, self.psi_X_0).reshape(8)
        evolved_psi = self.qc.run(test_psi, 1)[0].reshape(8)
        self.assertEqual(np.round(abs(evolved_psi.conj() @ xxx_test_psi), 5), 1)

        self.qc = QuantumCircuit(3)
        self.qc.rxd([0, 2], np.pi, 0.000001)
        test_psi = prod3(self.psi_Z_0, self.psi_Y_0, self.psi_Y_0)
        xxx_test_psi = prod3(self.psi_Z_1, self.psi_Y_0, self.psi_Y_1).reshape(8)
        evolved_psi = self.qc.run(test_psi, 1)[0].reshape(8)
        self.assertEqual(np.round(abs(evolved_psi.conj() @ xxx_test_psi), 5), 1)

    def two_layer(self):
        self.qc = QuantumCircuit(5)
        self.qc.rxd([0, 1, 2], np.pi, 0.000001)
        self.qc.rxd([0, 1, 2], np.pi, 0.000001)

        test_psi = prod5(self.psi_Z_1, self.psi_Y_0, self.psi_Y_1, self.psi_Z_0, self.psi_X_0)
        xxx_test_psi = test_psi.reshape(32)
        evolved_psi = self.qc.run(test_psi, 1)[0].reshape(32)
        self.assertEqual(np.round(abs(evolved_psi.conj() @ xxx_test_psi), 5), 1)


class test_CZ_gate(unittest.TestCase):
    def setUp(self):
        self.psi_Z_0 = np.array([1, 0])
        self.psi_Z_1 = np.array([0, 1])
        self.psi_Y_0 = np.array([1, 1j])/np.sqrt(2)
        self.psi_Y_1 = np.array([1, -1j])/np.sqrt(2)
        self.psi_X_0 = np.array([1, 1])/np.sqrt(2)
        self.psi_X_1 = np.array([1, -1])/np.sqrt(2)
        
    def test_CZ(self):
        self.qc = QuantumCircuit(3)
        self.qc.cz(0, 1)
        
        test_psi = prod3(self.psi_Z_1, self.psi_X_1, self.psi_Y_0)
        xxx_test_psi = prod3(self.psi_Z_1, self.psi_X_0, self.psi_Y_0).reshape(8)
        evolved_psi = self.qc.run(test_psi, 1)[0].reshape(8)        
        self.assertEqual(np.round(abs(evolved_psi.conj() @ xxx_test_psi), 5), 1)

        test_psi = prod3(self.psi_Z_0, self.psi_X_1, self.psi_Y_0)
        xxx_test_psi = prod3(self.psi_Z_0, self.psi_X_1, self.psi_Y_0).reshape(8)
        evolved_psi = self.qc.run(test_psi, 1)[0].reshape(8)
        self.assertEqual(np.round(abs(evolved_psi.conj() @ xxx_test_psi), 5), 1)
       
        self.qc.cz(0, 2)
        test_psi = prod3(self.psi_Z_1, self.psi_X_1, self.psi_Y_0)
        xxx_test_psi = prod3(self.psi_Z_1, self.psi_X_0, self.psi_Y_1).reshape(8)
        evolved_psi = self.qc.run(test_psi, 1)[0].reshape(8)
        self.assertEqual(np.round(abs(evolved_psi.conj() @ xxx_test_psi), 5), 1)

if __name__ == "__main__":
    unittest.main()