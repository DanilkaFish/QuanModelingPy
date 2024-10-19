from simple_sim import *
from numpy import exp
import numpy as np
# from numpy.linalg import j
# from numpy.linalg import 
dic = {1:4, 2: 8, 3: 12, 6: 9, 7: 13, 11: 14}
class LCG:
    a = 75
    c = 74
    l = 100
    m = 2**16 + 1
    n0 = 100
    def __init__(self, seed):
        self.seed = seed
        self.x = seed
        for i in range(self.n0):
            self.get_double()
            
    # def get_next(self, x):
    #     return 
    def get_num(self, nmax):
        x = self.x
        self.x = (self.a*self.x + self.c) % self.m
        return x % nmax
    
    def get_double(self):
        x = self.x
        self.x = (self.a*self.x + self.c) % self.m
        return (x / self.m - 0.5) * 2 * self.l
        
class generator_qc:
    def __init__(self, n_qubits, seed):
        self.lcg = LCG(seed)
        self._n_qubits = n_qubits
    
    def gen_circ(self, depth) -> list[Operator]:
        op_list = []
        for i in range(depth):
            if self.lcg.get_num(2):
                op_list.append(self.generate_2_qubit())
            else:
                op_list.append(self.generate_1_qubit())
        return op_list
       
    @property
    def n_qubits(self):
        return self._n_qubits
    
    def generate_gate(self):
        pass 
    
    def generate_1_qubit(self)-> Operator:
        op = self.generate_n_qubit_op(1)
        op.data = np.array(op.data).reshape(2,2)
        op.data = exp(1j*op.data)
        return op
        
    def generate_2_qubit(self) -> Operator:
        op = self.generate_n_qubit_op(2)
        op.data = np.matrix(np.array(op.data).reshape(4,4))
        # op.data = np.array(op.data).reshape(4,4)
        op.data = exp(1j*op.data)
        return op
    
    def generate_n_qubit_op(self, n)-> Operator:
        nh = (1 << n)
        her = [0]*nh*nh
        for i in range(0, nh ):
            for j in range(i , nh):
                if (i == j):
                    her[i*nh + i] = self.lcg.get_double()
                else:  
                    her[i*nh + j] = self.lcg.get_double() + 1j*self.lcg.get_double()
                    her[j*nh + i] = her[i*nh + j].conjugate()
        
        return Operator(self.generate_n_qubits(n), her)

    def generate_n_qubits(self, n) -> list[int]:
        qubits = [-1] * n
        i = 0
        buf = self.lcg.get_num(self.n_qubits)
        while (buf in qubits) and (i < n):
            qubits[i] = i
            i += 1
        return qubits
    
    
if __name__ == "__main__":
    gqc = generator_qc(5, 27)
    # lcg = LCG(27)
    # print(lcg.get_double())
    # print(gqc.generate_1_qubit())
    # print(gqc.generate_2_qubit())
    
    qc = gqc.gen_circ(3)
    for el in qc:
        print(el)
