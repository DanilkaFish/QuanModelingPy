import numpy as np
import methods as mt


class eq:
    def __init__(self):
        self.L = 4
        self.T = 1
        self.N = 10000
        self.M = 1000
        self.MASS = 50
        
        self.x0 = 0.3*self.L
        self.sigma2 = 0.0005*self.L**2
        self.k0 = 0.7*self.L/self.T*self.MASS
        self.norm = (1/2/np.pi/self.sigma2)**(1/4)
        
        self.U = np.vectorize(self._U)

        self.kappa = np.pi/self.L
        self.normphi = np.sqrt(2/self.L)
        self.bar_height = self.k0/self.MASS/0.002/self.L
    
    def f0(self, x):
        return self.norm * np.exp(-(x - self.x0)**2/4/self.sigma2)*np.exp(1j*self.k0*x)

    def phi_n(self, x, n): 
        return self.normphi*np.sin(self.kappa*x*(n+1))

    def _U(self,x):
        if x<0.499*self.L:
            return 0
        elif x < 0.501*self.L:
            return self.bar_height
        else:
            return 0
        
if __name__ == "__main__":
    Kmin = 0
    Kmax = 400
    params = eq()
    mt.discret(params, name="discret_fin_barier")
    mt.galerkin_exp(params, Kmin, Kmax, name="galerkin_fin_bareier")
