import numpy as np
from noisy_simulation import Noisy_Xgate_Simulation
import matplotlib.pyplot as plt


def error_plot(obj: Noisy_Xgate_Simulation, 
            N=1, 
            M=1, 
            num=1000,
            samples=[i*10 for i in range(1,5)],
            file_name="standart", 
            title_name="standart",
            meas_array=[np.array([[1, 0],[0, 0]])],
            cols=["Z"],
            rows=["Discrete", "Continious"]
            ):
    l = len(meas_array)
    N = 1
    M = 1
    THETA0 = np.linspace(0, 2*np.pi, N)
    SIGMA2 = np.linspace(0.0001, np.pi, M)


    DATA2, DATA1 = np.meshgrid(SIGMA2, THETA0)
    resc = np.empty((N, M))
    resd = np.empty((N, M))
    rest = np.empty((N, M))

    resth = np.empty((N, M))
    

    fig, axes = plt.subplots(1, l, figsize=(8*l, 15))
    fig.suptitle(title_name, size=40)

    deviation_d = np.empty(len(samples))
    deviation_c = np.empty(len(samples))
    deviation_t = np.empty(len(samples))

    for meas_index, meas in enumerate(meas_array):
        obj.measurement_projector = meas

        for n, theta0 in enumerate(THETA0):
            for m, sigma2 in enumerate(SIGMA2):
                obj.theta0 = theta0
                obj.sigma2 = sigma2
                resth[n, m] = obj.mean_meas_probability()
        for i, counts in enumerate(samples):    
            for n, theta0 in enumerate(THETA0):
                for m, sigma2 in enumerate(SIGMA2):
                    obj.theta0 = theta0
                    obj.sigma2 = sigma2
                    resc[n, m] = 0
                    resd[n, m] = 0
                    rest[n, m] = 0

                    for _ in range(num):
                        resc[n, m] += (np.sum(obj.continious_sampling_sim(counts))/counts - resth[n, m])**2
                        resd[n, m] += (np.sum(obj.discrete_sampling_sim(counts))/counts - resth[n, m])**2
                        rest[n, m] += (np.sum(np.random.binomial(1, resth[n, m], counts))/counts - resth[n, m])**2

            deviation_c[i] = np.sum(resc)/num/N/M
            deviation_d[i] = np.sum(resd)/num/N/M
            deviation_t[i] = np.sum(rest)/num/N/M

        def plot(res1, res2, rest, ax):
            ax.plot(samples, res1, label="c",color="black")
            ax.plot(samples, res2, label="d")
            ax.plot(samples, rest, label="t")
            ax.set_xlabel(r'$n$', fontsize=17)
            ax.set_ylabel(r'$\Delta p$', fontsize=17)

        plot(deviation_c, deviation_d, deviation_t, axes[meas_index])
        # plot( axes[1, meas_index])


    pad = 5 # in points
    for ax, col in zip(axes, cols):
        ax.annotate(col, xy=(0.5, 1.04), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size=25, ha='center', va='baseline')

    plt.savefig(file_name + ".png", dpi=400)


if __name__ == "__main__":
    psi1 = (np.array([1, -1])/np.sqrt(2)).reshape([2,1])
    psi2 = (np.array([1, 1])/np.sqrt(2)).reshape([2,1])

    measX = 1/2*np.array([[1, -1],[-1, 1]])
    measY = 1/2*np.array([[1, -1j],[1j, 1]])
    measZ = np.array([[1, 0], [0, 0]]) 
    cols = ['meas {}'.format(row) for row in ['X', 'Y', 'Z']]
    rows = ["Theory", "Discrete", "Continious"]

    PHI = [np.pi/6*i for i in range(0,6)]
    phi_label = [r"0", r"\pi/6", r"\pi/3", r"\pi/2", r"2\pi/3", r"5\pi/6"]
    names = [r"$\cos("+ phi + r")|+> + e^{i\pi/3}\sin(" + phi + r")|->$" for phi in phi_label]

    i = 10
    for phi, name in zip(PHI[3:4], names[3:4]):
        psi =  np.cos(phi/2)*psi1 + np.sin(phi/2)*psi2
        rho_init = np.kron(psi.conj().transpose(), psi)
        obj = Noisy_Xgate_Simulation(rho=rho_init)

        error_plot(obj=obj, 
                M=20,
                N=20,
                file_name=str(i) + f"error", 
                title_name=name,
                meas_array=[measX, measY, measZ],
                cols=cols,
                rows=rows)
        i += 1