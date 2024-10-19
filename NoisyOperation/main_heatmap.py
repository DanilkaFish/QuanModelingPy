import numpy as np
from noisy_simulation import Noisy_Xgate_Simulation
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable

# tgdm!!!!!!
def heatmap(obj: Noisy_Xgate_Simulation, 
            N=20, 
            M=20, 
            counts=100, 
            file_name="standart", 
            title_name="standart",
            meas_array=[np.array([[1, 0],[0, 0]])],
            cols=["Z"],
            rows=["Theory", "Discrete", "Continious"]
            ):
    THETA0 = np.linspace(0, 2*np.pi, N)
    SIGMA2 = np.linspace(0.01, np.pi, M)
    DATA2, DATA1 = np.meshgrid(SIGMA2, THETA0)

    resc = np.empty((N, M))
    resd = np.empty((N, M))
    resth = np.empty((N, M))
    l = len(meas_array)
    fig, axes = plt.subplots(3, l, figsize=(8*l, 15))
    norm = colors.Normalize(vmin=0, vmax=1)

    for index, meas in enumerate(meas_array):
        obj.measurement_projector = meas

        for n, theta0 in enumerate(THETA0):
            for m, sigma2 in enumerate(SIGMA2):
                obj.theta0 = theta0
                obj.sigma2 = sigma2
                resc[n, m] = np.sum(obj.continious_sampling_sim(counts))/counts
                resd[n, m] = np.sum(obj.discrete_sampling_sim(counts))/counts
                resth[n, m] = obj.mean_meas_probability()

        fig.suptitle(title_name, size=40)

        def plot(res, ax):
            c = ax.pcolormesh(DATA1, DATA2, res, )     
            ax.set_xlabel(r'$\theta_0$', fontsize=17)
            ax.set_ylabel(r'$\sigma^2$', fontsize=17)
            c.set_norm(norm)

        plot(resth, axes[0, index])
        plot(resd, axes[1, index])
        plot(resc, axes[2, index])


    pad = 5 # in points
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1.04), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size=25, ha='center', va='baseline')
    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 6), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=25, ha='right', va='center')

    fig.colorbar(ScalarMappable(norm=norm), 
                 ax=axes, 
                 orientation='vertical', 
                 fraction=.1)
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

    i = 6
    for phi, name in zip(PHI, names):
        psi =  np.cos(phi)*psi1 + np.exp(1j*np.pi/3)*np.sin(phi)*psi2
        rho_init = np.kron(psi.conj().transpose(), psi)
        obj = Noisy_Xgate_Simulation(rho=rho_init)

        counts = 100
        heatmap(obj=obj, 
                counts=counts, 
                file_name=str(i) + f"_{counts}", 
                title_name=name,
                meas_array=[measX, measY, measZ],
                cols=cols,
                rows=rows)
        i += 1