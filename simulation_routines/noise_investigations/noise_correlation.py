import path_to_noise_qmeq as noise_qmeq

noise_qmeq.noise_qmeq_path()
import sys

print(sys.path)
import matplotlib.pyplot as plt
import numpy as np
import qmeq

import setup as set


def main():
    np.set_printoptions(precision=6)

    t_set = set.create_transport_setup()

    t_set.initialize_leads()
    t_set.initialize_box()
    t_set.connect_box()

    sys = t_set.build_qmeq_sys()

    fig, ax = plt.subplots(1, 1)
    phi_range = np.linspace(0, 1 * np.pi, 100)
    phi_idx = 0
    print(correlation_calc(sys, t_set))
    correlation_plot(sys, t_set, ax, phi_range, phi_idx)
    plt.show()
    return

    fig, axes = plt.subplots(1, 2)
    correlation_sweep_plot(fig, axes, sys, t_set, points=40)
    plt.show()


def correlation_sweep_plot(fig, axes, sys, t_set, points=100, logscale=False):
    X, Y, I = correlation_sweep(sys, t_set, points)
    if logscale:
        c1 = axes[0].contourf(X, Y, I[:, :, 0], locator=ticker.LogLocator())
        c2 = axes[1].contourf(X, Y, I[:, :, 1])
    else:
        c1 = axes[0].contourf(X, Y, I[:, :, 0])
        c2 = axes[1].contourf(X, Y, I[:, :, 1])

    cbar = fig.colorbar(c1, ax=axes[0])
    axes[0].locator_params(axis='both', nbins=5)
    cbar.ax.locator_params(axis='y', nbins=5)

    fs = 16
    axes[0].tick_params(labelsize=fs)
    cbar.ax.set_title('current', size=fs)
    cbar.ax.tick_params(labelsize=fs)

    cbar = fig.colorbar(c2, ax=axes[1])
    axes[1].locator_params(axis='both', nbins=5)
    cbar.ax.locator_params(axis='y', nbins=5)

    fs = 16
    axes[1].tick_params(labelsize=fs)
    cbar.ax.set_title('noise correlation', size=fs)
    cbar.ax.tick_params(labelsize=fs)

    return


def correlation_sweep(sys, t_set, points):
    x = np.linspace(0, 1 * np.pi, points)
    X, Y = np.meshgrid(x, x)

    current = np.zeros(X.shape + (2, ))
    for idx, dummy in np.ndenumerate(X):
        t_set.phi0 = X[idx]
        t_set.phi2 = Y[idx]
        current[idx] = correlation_calc(sys, t_set)
    return X, Y, current


def correlation_plot(sys, t_set, ax, phi_range, phi_idx):
    n_cor = []
    for phi in phi_range:
        n_cor.append(correlation_from_setup(sys, t_set, phi, phi_idx))

    n_cor = np.array(n_cor)
    ax.plot(phi_range, n_cor[:, 0])
    ax.set_ylabel('current')
    ax.grid(True)

    ax_t = ax.twinx()
    ax_t.plot(phi_range, n_cor[:, 1], c='r')
    ax_t.set_ylabel('Noise correlation', c='r')
    return n_cor


def correlation_from_setup(sys, t_set, phi, phi_idx):
    if phi_idx == 0:
        t_set.phi0 = phi
    elif phi_idx == 1:
        t_set.phi1 = phi
    elif phi_idx == 2:
        t_set.phi2 = phi
    elif phi_idx == 3:
        t_set.phi3 = phi
    return correlation_calc(sys, t_set)


def correlation_calc(sys, t_set):
    t_set.initialize_box()
    t_set.connect_box()
    sys.Tba = t_set.tunnel

    lead_comb = [[0], [1], [0, 1]]
    current = []
    for cl in lead_comb:
        t_set.counting_leads = cl
        sys = t_set.build_qmeq_sys()
        sys.solve(qdq=False, rotateq=False)

        current.append(sys.current_noise)

    current = np.array(current)
    noise_correlation = [
        +current[2, 0], (current[2, 1] - current[0, 1] - current[1, 1]) / 2
    ]
    return noise_correlation


if __name__ == '__main__':
    main()
