import numpy as np
from numpy._typing import _ArrayLike


class Noisy_Xgate_Simulation:
    def __init__(self, 
                 theta0: float = 0, 
                 sigma2: float = 0.0001, 
                 rho: _ArrayLike = None, 
                 meas: _ArrayLike = None
                 ):
        """
        \theta \in N(\theta0, \sigma^2)
        self.rho: from distribution on Bloch sphere by default
        self.measurement_projector: |1><1| by default
        """
        self.theta0 = theta0
        self.sigma2 = sigma2
        self.rho = rho
        self.measurement_projector = meas

    @property
    def theta0(self):
        return self._theta0
    
    @theta0.setter
    def theta0(self, var):
        self._theta0 = var
        self.up_to_date = False

    @property
    def sigma2(self):
        return self._sigma2
    
    @sigma2.setter
    def sigma2(self, var):
        self._sigma2 = var
        self.up_to_date = False

    @property
    def rho(self):
        return self._rho
        
    @rho.setter
    def rho(self, var):
        if var is not None:
            self._rho = np.array(var)
        else:
            self._rho = self._rho_generation()

    def _rho_generation(self):
        Z = np.array([[1, 0], [0, -1]])
        Y = np.array([[0, -1j], [1j, 0]])
        X = np.array([[0, 1], [1, 0]])
        I = np.array([[1, 0], [0, 1]])
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, 1)
        return 1/2*I + 1/2*r*(X*np.sin(theta)*np.cos(phi) +
                            Y*np.sin(theta)*np.sin(phi) +
                            Z*np.cos(theta))
    
    @property
    def measurement_projector(self):
        return self._measurement_projector
    
    @measurement_projector.setter
    def measurement_projector(self, var):
        if (var is not None):
            self._measurement_projector = var
        else:
            self._measurement_projector = np.array([[1, 0],[0, 0]])

    @property
    def mean_G(self):
        if self.up_to_date:
            return self._mean_G
        else:
            self._mean_G = self.get_mean_G()
            self.up_to_date = True
            return self._mean_G

    def get_mean_G(self):
        mean = np.exp(((self.theta0 + 1j*self.sigma2)**2 - self.theta0**2)/2/self.sigma2)
        re = mean.real
        im = mean.imag
        G = np.array([[(1 + re)/2, 1j/2 * im, -1j/2 * im, (1 - re)/2], 
                      [1j/2 * im, (1 + re)/2, (1 - re)/2, -1j/2 * im],
                      [-1j/2 * im, (1 - re)/2, (1 + re)/2, 1j/2 * im],
                      [(1 - re)/2, -1j/2 * im, 1j/2 * im, (1 + re)/2]])
        return G
    
    def get_kraus(self):
        hi = self.mean_G.reshape([2, 2, 2, 2]).transpose([0, 2, 1, 3]).reshape((4, 4))
        eig = np.linalg.eig(hi)
        E = np.empty((4, 2, 2), dtype=np.complex128)
        probs = np.empty(4)

        for num, eigval in enumerate(eig.eigenvalues):
            Ek = eig.eigenvectors[:, num].reshape(2, 2)
            norm = abs(np.trace(Ek @ Ek.conj().transpose([1,0])))*2
            E[num] = Ek*np.sqrt(norm)
            probs[num] = np.round(abs(eigval)/norm, decimals=5)
        return E, probs

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
    
    def sample_measurements(self, rho_array):
        probs = np.round(np.abs(np.tensordot(rho_array, 
                                    self.measurement_projector, 
                                    axes=((2, 1), (0, 1))
                                    )), decimals=5)
        return np.random.binomial(1, probs)

    def discrete_sampling_sim(self, counts=100):
        E, probs = self.get_kraus()
        Operators = np.random.default_rng().choice(E, size=counts, p=probs).transpose(1, 2, 0)
        rho_evolve = np.diagonal(np.tensordot(np.tensordot(Operators, self.rho, axes=(1, 0)),
                                              Operators.conj(), 
                                              axes=((2, 1))),
                                  axis1=1,
                                  axis2=3).transpose([2, 0, 1])
        return self.sample_measurements(rho_evolve)
    
    def continious_sampling_sim(self, counts=100):
        theta = np.random.normal(self.theta0, np.sqrt(self.sigma2), counts)
        Xtheta = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                        [-1j*np.sin(theta/2), np.cos(theta/2)]])
        rho_evolve_array = np.diagonal(np.tensordot(np.tensordot(Xtheta, self.rho, axes=(1, 0)), 
                                              Xtheta.conj(), 
                                              axes=((2, 1))),
                                  axis1=1,
                                  axis2=3).transpose([2, 0, 1])
        return self.sample_measurements(rho_evolve_array)

    def mean_meas_probability(self):
        rho_evolve = (self.mean_G@self.rho.reshape([4])).reshape([2, 2])
        return self.meas_probability(rho_evolve)
    
    def noiseless_probaility(self):
        theta = self.theta0
        Xtheta = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                           [-1j*np.sin(theta/2), np.cos(theta/2)]])
        rho_evolve = Xtheta@self.rho@(Xtheta.conj())
        return self.meas_probability(rho_evolve)
    
    def meas_probability(self, rho):
        p = np.trace(rho@self.measurement_projector)
        if p > 1:
            p = 1
        elif p < 0:
            p = 0 
        return abs(p)