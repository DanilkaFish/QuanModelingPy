import numpy as np
import scipy as sp
import plot as pr

def discret(data, sigma=0.5, name="discret"):
    """
    Shrodinger equation:
        ih df/dt = - h^2/2m d^2f/dx^2
        f(0,t) = f(data.L,t) = 0
        f(x,0) = data.f0(x) = C\exp{-\frac{(x-x0)^2}{2\sigma^2}}

    Discretization:
        (f_nm - f_nm-1)  = i{h dt}/{2m dx^2} (f_n+1m - 2f_nm + f_n-1m) - idt*data.U(x_n,t)(f_nm-1 +f_nm)/2
        (f_nm - f_nm-1)  = i{h dt}/{2m dx^2} (f_n+1m-1 - 2f_nm-1 + f_n-1m-1) - idt*data.U(x,t)

        (f_nm - f_nm-1) = sigma*Bpsi_m + (1-sigma)*B @ psi_{m-1} - idt*data.U(x_n, t_m - 0.5*dt)
        f_0m = f_data.Nm = 0
        f_n0 = data.f0(n*dx)
        
        psi_0 = data.f0(Xgrid)
        psi_m - psi_m-1 = sigma*B @ psi_m + (1-sigma)*B @ psi_m-1, 
        (I - sigma*B) @ psi_m = (I+(1-sigma)*B) @ psi_{m-1} - idt*data.U(x_n, t_m - 0.5*dt)
    """

    Xgrid, dx = np.linspace(0, data.L, data.N, endpoint=True, retstep=True)
    Tgrid, dt = np.linspace(0, data.T, data.M, endpoint=True, retstep=True)
    A = - 1/(2*data.MASS*dx*dx)

    Psi_vectors = np.ndarray((data.M, data.N), dtype=np.complex128)
    # init data
    Psi_vectors[0] = data.f0(Xgrid)
    
    # implicit layer
    abc = np.array([
                np.full(data.N, + 1j*sigma*dt*A),
                np.full(data.N, 1 - 1j*sigma*2*dt*A),
                np.full(data.N, + 1j*sigma*dt*A)
                ])
    abc[1,0], abc[1,-1], abc[0,0], abc[0,1], abc[2,-2], abc[2,-1] = 1, 1, 0, 0, 0, 0
    abc[1] += 1j*dt/2*data.U(Xgrid)

    # explicit layer 
    B_matrix = np.eye(data.N, dtype=np.complex128)
    B_matrix[0,0] += - 1j*dt*data.U(Xgrid[0])/2
    B_matrix[-1,-1] += - 1j*dt*data.U(Xgrid[-1])/2
    for i in range(1,data.N-1):
        B_matrix[i, i-1:i+2] += - 1j*dt*A*(1 - sigma) * np.array([1, -2, 1]) 
        B_matrix[i,i] += - 1j/2*dt*data.U(Xgrid[i])

    for i in range(1,data.M):
        Psi_vectors[i] = sp.linalg.solve_banded((1,1), abc, B_matrix @ Psi_vectors[i-1])

    # pr.plot(Xgrid, Tgrid, Psi_vectors, name )
    pr.gif(Xgrid, Tgrid, Psi_vectors,data.U,name=name)


def galerkin_exp(data, Kmin=0, Kmax=2, name="galerkin"):
    """
    Shrodinger equation:
        ih df/dt = - h^2/2m d^2f/dx^2
        f(0,t) = f(data.L,t) = 0
        f(x,0) = data.f0(x) = C\exp{-\frac{(x-x0)^2}{2\sigma^2}}

    Basis:
        psi_n(x)
        psi(x,t) = c_n(t)*psi_n(x)

    Galerkin:
        dc_n/dt = c_m<psi_n|H|psi_m> 
        C(t_k) = exp{iHdt} @ C(t_k-1))
    """
    # basis
    Xgrid, dx = np.linspace(0, data.L, data.N, endpoint=True, retstep=True)
    Tgrid, dt = np.linspace(0, data.T, data.M, endpoint=True, retstep=True)
    A = - 1/(2*data.MASS*dx*dx)
    K = Kmax - Kmin

    Coef_vectors = np.zeros((data.M, K), dtype=np.complex128)
    Psi_vectors = np.ndarray((data.M, data.N), dtype=np.complex128)
    Psi_vectors[0] = data.f0(Xgrid)

    # second derivative discretization with first order approx (laziness)
    Ham_matrix = np.eye(data.N, dtype=np.complex128)
    Ham_matrix[0, 0:3] = np.array([A + data.U(Xgrid[0]), -2*A, A])
    Ham_matrix[-1, -4:-1] = np.array([A, -2*A, A + data.U(Xgrid[-1])])
    for n in range(1,data.N-1):
        Ham_matrix[n, n-1:n+2] = np.array([A, -2*A + data.U(Xgrid[n]), A])

    Phi_vectors = [data.phi_n(Xgrid, k) for k in range(Kmin, Kmax)]

    # auxiliary matrix for effective projection
    HPhi_vectors = np.array([Ham_matrix @ Phi_vectors[k] for k in range(K)])
    Phi_conj_vectors = [Phi_vectors[k].conj() for k in range(K)]

    # Ham projection on basis space 
    Ham_basis_matrix = np.array([[dx*(HPhi_vectors[l] @ Phi_conj_vectors[k]) 
                                  for k in range(K)] for l in range(K)])
    # init conditions
    Coef_vectors[0] = np.array([dx*(Phi_conj_vectors[k] @ Psi_vectors[0]) for k in range(K)])
    # projection of init conditions (not necessary)
    Psi_vectors[0] = Coef_vectors[0,:] @ Phi_vectors

    data.U_basis_matrix = sp.linalg.expm(-1j*dt*Ham_basis_matrix)
    for m in range(1, data.M):
        Coef_vectors[m] = data.U_basis_matrix @ Coef_vectors[m-1]
        Psi_vectors[m] = Coef_vectors[m,:] @ Phi_vectors

    pr.gif(Xgrid, Tgrid, Psi_vectors, U=data.U, name=name)
