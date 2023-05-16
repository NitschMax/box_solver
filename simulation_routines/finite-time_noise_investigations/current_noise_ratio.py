import help_functions as hf
import matplotlib.pyplot as plt

import cyclic_blockade as cb
import setup as set
import time_evolution as te

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

import matplotlib.ticker as ticker
import numpy as np
import qmeq


def main():
    np.set_printoptions(precision=6)

    t_set = set.create_transport_setup()

    #t_set.adjust_to_z_blockade()

    t_set.initialize_leads()
    t_set.initialize_box()
    t_set.connect_box()

    t_set.maj_box.print_eigenstates()

    sys = t_set.build_qmeq_sys()
    sys.solve(qdq=False, rotateq=False)

    points = 81
    fig, axes = plt.subplots(1, 1, figsize=(3, 3 * 1.57))
    axes = [axes]
    #Call the nanonlund_annual function
    # nanolund_annual(sys, t_set, fig, axes, points)
    # plt.show()
    plt.close()

    points = 41
    fig, axes = plt.subplots(1, 2)
    #axes   = [axes]
    logsc = True

    tunnel_sweep_plot(sys, t_set, fig, [axes[0]], points, logscale=logsc)
    phase_sweep_plot(sys, t_set, fig, [axes[1]], points, logscale=logsc)
    plt.show()
    plt.close()
    return

    phi_idx = 0

    fig, ax = plt.subplots(1, 1)
    phi_range = np.linspace(0, np.pi, 11)
    phi_range = np.pi / 2 + np.linspace(-1e-2, 1e-2, 40)
    current_noise_plot_phi0(sys, t_set, ax, phi_range, phi_idx)
    plt.show()

    return


#write a function called nanolund_annual() that creates and saves the data for the nanolund annual report
#The function should be called in the main function
#The argument of the function should be sys, t_set, fig, axes, points
#The function should return X, Y, I
#The function should save the data in the folder /Users/ma0274ni/Documents/projects/majorana_box/plots/nanolund_annual/data


def nanolund_annual(sys, t_set, fig, axes, points):
    dir = '/Users/ma0274ni/Documents/projects/majorana_box/plots/nanolund_annual/'
    X = np.load(dir + 'data/X.npy')
    Y = np.load(dir + 'data/Y.npy')
    I = np.load(dir + 'data/I.npy')
    data = [X, Y, I]
    X, Y, I = phase_sweep_plot(sys,
                               t_set,
                               fig,
                               axes,
                               points,
                               logscale=True,
                               data=data,
                               rerun=False)
    data = [X, Y, I]
    np.save(dir + 'data/X', X)
    np.save(dir + 'data/Y', Y)
    np.save(dir + 'data/I', I)

    #Loop over the dpi numbers 300, 600, 900, 1200 and save the figure with each dpi number
    for dpi in [300, 600, 900, 1200]:
        plt.savefig(dir + 'Fano-factor_phase-sweep-81x81_dpi-' + str(dpi),
                    dpi=dpi)
    plt.show()

    return

    plt.savefig(dir + 'Fano-factor_phase-sweep-81x81_dpi-300', dpi=300)
    plt.savefig(dir + 'Fano-factor_phase-sweep-81x81_dpi-600', dpi=600)
    plt.savefig(dir + 'Fano-factor_phase-sweep-81x81_dpi-900', dpi=900)
    plt.savefig(dir + 'Fano-factor_phase-sweep-81x81_dpi-1200', dpi=1200)
    plt.show()
    return


def sweep_func(sys, t_set, t0, t2):
    t_set.gamma_00 = hf.gamma_from_tunnel(t0)
    t_set.gamma_02 = hf.gamma_from_tunnel(t2)
    t_set.block_via_phases(lead=0)

    t_set.connect_box()
    sys.Tba = t_set.tunnel

    sys.solve(qdq=False, rotateq=False)
    return np.array(sys.current_noise)


def tunnel_sweep_calc(sys, t_set, points):
    t1 = hf.tunnel_from_gamma(t_set.gamma_01)
    x = np.linspace(0, 2 * t1, points)
    y = x

    X, Y = np.meshgrid(x, y)

    I = np.array([
        sweep_func(sys, t_set, X[idx], Y[idx]) for idx in np.ndindex(X.shape)
    ])
    I = I.reshape(X.shape + (2, )).real

    return X / t1, Y / t1, I


def tunnel_sweep_plot(sys, t_set, fig, axes, points, logscale=False):
    X, Y, I = tunnel_sweep_calc(sys, t_set, points)
    fs = 18
    for ax in axes:
        ax.locator_params(axis='both', nbins=3)
        ax.set_xlabel(r'$|t_0|/t$', size=fs)
        ax.set_ylabel(r'$|t_2|/t$', size=fs)
    colorbar_plot(X, Y, I, fig, axes, logscale, fs)
    return


def sweep_phases(sys, t_set, phi_avg, phi_diff):
    #t_set.gamma_00 = 1
    #t_set.gamma_01 = 1
    #t_set.gamma_02 = 1
    t_set.phi0 = phi_avg + phi_diff
    t_set.phi2 = phi_avg - phi_diff
    t_set.block_via_rates(lead=0, tuneable_rates=[1, 1, 0, 0])

    t_set.connect_box()
    sys.Tba = t_set.tunnel

    sys.solve(qdq=False, rotateq=False)

    return np.array(sys.current_noise)


def phase_sweep_calc(sys, t_set, points):
    x = np.linspace(-np.pi / 2 - 1e-2, np.pi / 2 + 1e-2, points)
    y = x

    X, Y = np.meshgrid(x, y)

    I = np.array([
        sweep_phases(sys, t_set, X[idx], Y[idx]) for idx in np.ndindex(X.shape)
    ])
    I = I.reshape(X.shape + (2, )).real

    return X, Y, I


def phase_sweep_plot(sys,
                     t_set,
                     fig,
                     axes,
                     points,
                     logscale=False,
                     data=None,
                     rerun=False):
    if data is not None and rerun is False:
        X = data[0]
        Y = data[1]
        I = data[2]
    else:
        X, Y, I = phase_sweep_calc(sys, t_set, points)
    fs = 18
    for ax in axes:
        ax.set_xlabel(r'$\phi_\textrm{avg}$', size=fs)
        ax.set_ylabel(r'$\phi_\textrm{diff}$', size=fs, labelpad=-20)

        ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

        ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

    colorbar_plot(X, Y, I, fig, axes, logscale, fs)
    return X, Y, I


def colorbar_plot(X, Y, I, fig, axes, logscale, fs):
    if len(axes) == 1:
        second_plot = False
    else:
        second_plot = True

    if logscale:
        print(I[:, :, 0])
        c1 = axes[0].contourf(X, Y, I[:, :, 0], locator=ticker.LogLocator())
        if second_plot:
            c2 = axes[1].contourf(X, Y, I[:, :, 1] / I[:, :, 0],
                                  cmap='Blues')  #, cmap='RdYlGn'
    else:
        c1 = axes[0].contourf(X, Y, I[:, :, 1] / I[:, :, 0])
        if second_plot:
            c2 = axes[1].contourf(X, Y, I[:, :, 0])

    cbar = fig.colorbar(c1, ax=axes[0])
    cbar.ax.locator_params(axis='y', nbins=3)

    axes[0].tick_params(labelsize=fs)

    cbar.ax.set_title(r'$ I_\textrm{min} \, [\Gamma e]$', size=fs, pad=10)
    cbar.ax.tick_params(labelsize=fs)

    if second_plot:
        cbar = fig.colorbar(c2, ax=axes[1])
        #axes[1].locator_params(axis='both', nbins=5 )
        cbar.ax.locator_params(axis='y', nbins=5)

        axes[1].tick_params(labelsize=fs)
        cbar.ax.set_title('Fano factor', size=fs)
        cbar.ax.tick_params(labelsize=fs)

    fig.tight_layout()
    return


def current_noise_plot_phi0(sys, t_set, ax, phi_range, phi_idx):
    ratio = []
    current = []

    current_from_setup(sys, t_set, np.pi / 2, phi_idx)
    print(sys.phi0)

    for phi in phi_range:
        cur = current_from_setup(sys, t_set, phi, phi_idx)
        if cur[0] < 0:
            cur[0] = -cur[0]
        ratio.append(cur[1] / cur[0])
        current.append(cur[0])
    ax.plot(phi_range, ratio)
    #ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    #ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func) )
    ax.set_xlabel(r'$\phi_{}$'.format(phi_idx))
    ax.set_ylabel('noise/current')
    #ax.set_ylim( [ 0.5, 2] )
    ax.grid(True)

    ax_twin = ax.twinx()
    ax_twin.plot(phi_range, current, c='r')
    #ax_twin.set_ylim(bottom=0)
    ax_twin.set_ylabel(r'current [$e\Gamma$]')
    ax_twin.yaxis.get_label().set_color('r')


def current_from_setup(sys, t_set, phi, phi_idx):
    if phi_idx == 0:
        t_set.phi0 = phi
    elif phi_idx == 1:
        t_set.phi1 = phi
    elif phi_idx == 2:
        t_set.phi2 = phi
    elif phi_idx == 3:
        t_set.phi3 = phi

    t_set.initialize_box()
    t_set.connect_box()
    sys.Tba = t_set.tunnel

    sys.solve(qdq=False, rotateq=False)
    return sys.current_noise


def format_func(value, tick_number):
    # find number of multiples of pi/2
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N == -1:
        return r"$-\pi/2$"
    elif N == -2:
        return r"$-\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)


if __name__ == '__main__':
    main()
