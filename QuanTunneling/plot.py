import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib import cm

def plot(X,T,F, name="png"):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    T, X = np.meshgrid(X, T)
 
    # Plot the surface.
    surf = ax.plot_surface(X, T, F.real**2 + F.imag**2, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(name + '.png', dpi=400)


def gif(X, T, F, U, name="gif"):
    t = T[-1]
    M = len(T)
    fig, ax = plt.subplots()
    G = F.real**2 + F.imag**2
    GR = F.real
    GI = F.imag
    line1, = ax.plot(X, G[0], label="Плотность")
    line2, = ax.plot(X, GR[0], label="Вещественная часть")
    line3, = ax.plot(X, GI[0],  label="Мнимая часть")
    line4, = ax.plot(X, U(X)/20)
    def animate(i):
        line1.set_ydata(G[i*10])
        line2.set_ydata(GR[i*10])
        line3.set_ydata(GI[i*10])  # update the data.
        return line1, line2, line3, line4

    ani = animation.FuncAnimation(
        fig, animate, interval=10, blit=True, save_count=M//10)

    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=15,
                                    metadata=dict(artist='Me'),
                                    bitrate=1800)
    ani.save(name + '.gif', writer=writer, dpi=100)

