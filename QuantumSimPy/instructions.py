from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike 

CZ =  np.diag([1, 1, 1, -1]).reshape([2, 2, 2, 2])

class Intstruction(ABC):
    def __init__(self, qubits: list, **kwargs):
        self.qubits = qubits
        self.n_qubits = len(qubits)
        self.axes_psi = np.array(qubits) + 1
        self.axes_ops = np.arange(1, self.n_qubits + 1)  
    
    def evolve_states(self, psi_array):
        counts = psi_array.shape[0]
        return np.moveaxis(np.diagonal(np.tensordot(psi_array, 
                                                    self.ops(counts), 
                                                    axes=(self.axes_psi, self.axes_ops)),
                                        axis1=0,
                                        axis2=-self.n_qubits - 1), 
                            source=np.arange(-1, -self.n_qubits - 2, -1), 
                            destination=[0] + list(reversed(self.axes_psi)))
    
    @abstractmethod
    def ops(self, counts) -> np.ndarray:
        pass

    @property
    def name(self):
        return "gate"


class CZgate(Intstruction):
    def ops(self):
        return CZ 
    
    def evolve_states(self, psi_array):
        return np.moveaxis(np.tensordot(psi_array, self.ops(), axes=(self.axes_psi, self.axes_ops - 1)), 
                            source=-1*(self.axes_ops), 
                            destination=reversed(self.axes_psi))
        
class Xgate(Intstruction):
    def __init__(self, 
                 qubits: ArrayLike,
                 theta0: float = 0):
        """
            Generate layer of random X gates for array of qubits
        """
        self.theta0 = theta0
        super().__init__(qubits)
        self.axes_ops = 1

    def ops(self):
        return np.array([[np.cos(self.theta0/2), -1j*np.sin(self.theta0/2)],
                         [-1j*np.sin(self.theta0/2), np.cos(self.theta0/2)]])
    
    def evolve_states(self, psi_array):
        return np.moveaxis(np.tensordot(psi_array, self.ops(), axes=(self.axes_psi, self.axes_ops - 1)), 
                            source=-1*(self.axes_ops), 
                            destination=reversed(self.axes_psi))


class XgateNoisyD(Intstruction):
    def __init__(self, 
                 qubits: ArrayLike,
                 theta0: float = 0, 
                 sigma2: float = 0.0001):
        """
            Generate layer of random X gates for array of qubits
        """
        self.theta0 = theta0
        self.sigma2 = sigma2
        super().__init__(qubits)
        self.axes_ops = np.arange(1, self.n_qubits + 1) * 2

    def ops(self, counts):
        kraus, probs = self.get_an_kraus()
        choice = lambda: np.random.default_rng().choice(kraus, size=counts, p=probs)
        operators = choice()

        for _ in range(1, self.n_qubits):
            operators = np.einsum("i...,ijk->i...jk", operators, choice())
        return operators
    
    def get_an_kraus(self):
        mean = np.exp(((self.theta0 + 1j*self.sigma2)**2 - self.theta0**2)/2/self.sigma2)
        m = np.abs(mean)
        phi = self.theta0/2
        re = np.cos(phi)
        im = np.sin(phi)
        E = np.array([[[-1j*im, re],
                       [re, -1j*im]],
                      [[1j*re, im],
                       [im, 1j*re]]])
        probs = np.array([1/2*(1 - m), 1/2*(1 + m)])
        return E, probs
    
    @property
    def name(self):
        return "rxd"
    

class XgateNoisyC(Intstruction):
    def __init__(self, 
                 qubits: ArrayLike,
                 theta0: float = 0, 
                 sigma2: float = 0.0001):
        """
            Generate layer of random X gates for array of qubits
        """
        self.theta0 = theta0
        self.sigma2 = sigma2
        super().__init__(qubits)
        self.axes_ops = np.arange(1, self.n_qubits + 1) * 2


    def ops(self, counts):        
        def rx(theta):
            return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                             [-1j*np.sin(theta/2), np.cos(theta/2)]]).transpose([2, 0, 1])    

        choice = lambda: rx(np.random.normal(self.theta0, np.sqrt(self.sigma2), counts))
        operators = choice()

        for _ in range(1, self.n_qubits):
            operators = np.einsum("i...,ijk->i...jk", operators, choice())

        return operators
    
    @property
    def name(self):
        return "rxc"