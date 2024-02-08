import path_to_noise_qmeq as noise_qmeq

noise_qmeq.noise_qmeq_path()

import data_directory as dd
import help_functions as hf
import matplotlib.pyplot as plt

import setup as set

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

from multiprocessing import Pool

import matplotlib.ticker as ticker
import numpy as np
import qmeq
from matplotlib import colors


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

    points = 81
    fig, axes = plt.subplots(1, 2)
    #axes   = [axes]
    logsc = True

    current_condition = True
    t_set.blockade_condition_via_current = current_condition

    params = param_dict()
    params = fill_param_dict(params, t_set, points)
    recalculate = False
    print('Starting tunnel sweep')
    tunnel_sweep_plot(sys,
                      t_set,
                      fig, [axes[0]],
                      points,
                      params,
                      logscale=logsc,
                      recalculate=recalculate)
    recalculate = False
    print('Starting phase sweep')
    phase_sweep_plot(sys,
                     t_set,
                     fig, [axes[1]],
                     points,
                     params,
                     logscale=logsc,
                     recalculate=recalculate)

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
                               recalculate=False)

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


def sweep_func(t_set, t0, t2, idx, guess=None):
    print(idx)
    t_set.gamma_00 = hf.gamma_from_tunnel(t0)
    t_set.gamma_02 = hf.gamma_from_tunnel(t2)
    minimum = t_set.block_via_phases(lead=0, phase_angle_guess=guess)

    return [minimum.fun, minimum.fun]


def tunnel_sweep_calc(sys, t_set, points, params, recalculate=False):
    t1 = hf.tunnel_from_gamma(t_set.gamma_01)
    x = np.linspace(+1e-4 * t1, 2 * t1, points)
    y = x

    X, Y = np.meshgrid(x, y)

    I = None
    if not recalculate:
        I = dd.load_data(params, '0_blockade-tunnel-sweep')
    if I is None:
        # Set up loop over sweep_func with parallel processing
        # 1. Create a list of arguments to sweep_func

        args = [(t_set.copy(), X[idx], Y[idx], idx)
                for idx in np.ndindex(X.shape)]

        # 2. Create a function that takes the arguments and calls sweep_func
        # 3. Use multiprocessing to call the function with the arguments

        # Use Pool.map to call the function with the arguments
        # Can you make pool return feedback about the progress?
        # A:

        I = np.array(Pool().map(sweep_func_wrapper, args))
        I = I.reshape(X.shape + (2, )).real
        dd.save_data(I, params, '0_blockade-tunnel-sweep')

    return X / t1, Y / t1, I


def sweep_func_wrapper(args):
    return sweep_func(*args)


def tunnel_sweep_plot(sys,
                      t_set,
                      fig,
                      axes,
                      points,
                      params,
                      logscale=False,
                      lead=0,
                      recalculate=False):

    X, Y, I = tunnel_sweep_calc(sys,
                                t_set,
                                points,
                                params,
                                recalculate=recalculate)

    fs = 18
    for ax in axes:
        ax.locator_params(axis='both', nbins=3)
        ax.set_xlabel(r'$|t_0|/t$', size=fs)
        ax.set_ylabel(r'$|t_2|/t$', size=fs)
    cbar = colorbar_plot(X, Y, I, fig, axes, logscale, fs)
    # How include a 1-d line in the plot?
    if t_set.model == 1:
        fine_X = np.linspace(0, 2, 10000)
        inner_circle_X = fine_X[fine_X <= 1]
        outer_circle_X = fine_X[fine_X >= 1]
        lw = 2
        # How to get an orange dashed line?
        # A: Use plot with linestyle='--'
        # And the color orange is 'r'
        # No that's red
        # A:
        axes[0].plot(inner_circle_X,
                     np.sqrt(1 - inner_circle_X**2),
                     '--',
                     color='orange',
                     linewidth=lw)
        axes[0].plot(outer_circle_X,
                     np.sqrt(outer_circle_X**2 - 1),
                     '--',
                     color='orange',
                     linewidth=lw)
        axes[0].plot(fine_X,
                     np.sqrt(fine_X**2 + 1),
                     '--',
                     color='orange',
                     linewidth=lw)
        axes[0].set_xlim(0, 2)
        axes[0].set_ylim(0, 2)
    return


def sweep_phases(t_set, phi_avg, phi_diff, idx):
    print(idx)
    #t_set.gamma_00 = 1
    #t_set.gamma_01 = 1
    #t_set.gamma_02 = 1
    t_set.phi0 = phi_avg + phi_diff
    t_set.phi2 = phi_avg - phi_diff
    minimum = t_set.block_via_rates(lead=0, tuneable_rates=[1, 1, 0, 0])

    return [minimum.fun, minimum.fun]

    t_set.connect_box()
    sys.Tba = t_set.tunnel

    sys.solve(qdq=False, rotateq=False)

    return np.array(sys.current_noise)


def phase_sweep_calc(sys, t_set, points, params, recalculate=False):
    x = np.linspace(-np.pi / 2 - 1e-2, np.pi / 2 + 1e-2, points)
    y = x

    X, Y = np.meshgrid(x, y)

    I = None
    if not recalculate:
        I = dd.load_data(params, '0_blockade-phase-sweep')
    if I is None:
        args = [(t_set.copy(), X[idx], Y[idx], idx)
                for idx in np.ndindex(X.shape)]
        I = np.array(Pool().map(sweep_phases_wrapper, args))
        # Got the error: BrokenPipeError: [Errno 32] Broken pipe
        # when running this code. What does it mean?
        # A: It means that the process that you are trying to write to
        # has been closed. This can happen if you try to write to a
        # process that has already finished. In this case, it means
        # that the process has finished before you have finished
        # writing to it. This can happen if you are writing too much

        I = I.reshape(X.shape + (2, )).real
        dd.save_data(I, params, '0_blockade-phase-sweep')

    return X, Y, I


def sweep_phases_wrapper(args):
    return sweep_phases(*args)


def phase_sweep_plot(sys,
                     t_set,
                     fig,
                     axes,
                     points,
                     params,
                     logscale=False,
                     lead=0,
                     recalculate=False):

    X, Y, I = phase_sweep_calc(sys, t_set, points, params, recalculate)
    fs = 18
    for ax in axes:
        ax.set_xlabel(r'$\phi_\textrm{avg}$', size=fs)
        ax.set_ylabel(r'$\phi_\textrm{diff}$', size=fs, labelpad=-20)

        ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

        ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

    colorbar_plot(X, Y, I, fig, axes, logscale, fs)

    if t_set.model == 1:
        fine_X = np.linspace(-np.pi / 2, np.pi / 2, 1000)
        # Choose linewidth for the lines
        lw = 2
        axes[0].plot(fine_X,
                     np.pi / 4 * np.ones_like(fine_X),
                     '--',
                     color='orange',
                     linewidth=lw)
        axes[0].plot(fine_X,
                     -np.pi / 4 * np.ones_like(fine_X),
                     '--',
                     color='orange',
                     linewidth=lw)
        axes[0].plot(fine_X,
                     np.pi / 2 - fine_X,
                     '--',
                     color='orange',
                     linewidth=lw)
        axes[0].plot(fine_X,
                     -np.pi / 2 - fine_X,
                     '--',
                     color='orange',
                     linewidth=lw)
        axes[0].plot(fine_X,
                     np.pi / 2 + fine_X,
                     '--',
                     color='orange',
                     linewidth=lw)
        axes[0].plot(fine_X,
                     -np.pi / 2 + fine_X,
                     '--',
                     color='orange',
                     linewidth=lw)
        axes[0].set_xlim(-np.pi / 2, np.pi / 2)
        axes[0].set_ylim(-np.pi / 2, np.pi / 2)

    return X, Y, I


def colorbar_plot(X, Y, I, fig, axes, logscale, fs):
    if len(axes) == 1:
        second_plot = False
    else:
        second_plot = True

    # Make the colorplot black and white
    # All values above 1e-3 are white, all values below are black

    #I[:, :, 0] = np.where(I[:, :, 0] > 1e-3, 0, 1)
    # Define minimum value below which data is cut off
    cmap = 'viridis'
    if logscale:
        # Cast I to float to avoid overflow
        # c1 = axes[0].contourf(X.astype(float),
        #                       Y.astype(float),
        #                       I[:, :, 0].astype(float),
        #                       locator=ticker.LogLocator(),
        #                       cmap=cmap)
        c1 = axes[0].pcolormesh(X.astype(float),
                                Y.astype(float),
                                I[:, :, 0].astype(float),
                                cmap=cmap,
                                norm=colors.LogNorm(vmin=1.4e-7, vmax=5e-1))
        if second_plot:
            c2 = axes[1].contourf(X, Y, I[:, :, 1] / I[:, :, 0], cmap=cmap)
    else:
        c1 = axes[0].contourf(X, Y, I[:, :, 0], cmap=cmap)
        if second_plot:
            c2 = axes[1].contourf(X, Y, I[:, :, 1] / I[:, :, 0], cmap=cmap)

    # Print minimum and maximum values of I
    print('I_min = ', I[:, :, 0].min())
    print('I_max = ', I[:, :, 0].max())
    cbar = fig.colorbar(c1, ax=axes[0])
    cbar.locator = ticker.LogLocator(numticks=4)
    cbar.update_ticks()
    # Disable minor ticks for logscale
    cbar.ax.minorticks_off()

    cbar.ax.set_title(r'$ I_\textrm{min} \, [\Gamma e]$', size=fs, pad=10)
    cbar.ax.tick_params(labelsize=fs)

    axes[0].tick_params(labelsize=fs)

    if second_plot:
        cbar = fig.colorbar(c2, ax=axes[1])
        #axes[1].locator_params(axis='both', nbins=5 )
        cbar.ax.locator_params(axis='y', nbins=5)

        axes[1].tick_params(labelsize=fs)
        cbar.ax.set_title('Fano factor', size=fs)
        cbar.ax.tick_params(labelsize=fs)

    fig.tight_layout()
    return cbar


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


def param_dict():
    params = {
        "theta0": 0,
        "theta1": 0,
        "theta2": 0,
        "theta3": 0,
        "model": 1,
        "gridpoints": 0,
        "routine": "1vN",
        "current_condition": False,
    }
    return params


def fill_param_dict(params, t_set, points):
    params['model'] = t_set.model
    if params['model'] == 1:
        # Set all the thetas to 0
        params['theta0'] = 0
        params['theta1'] = 0
        params['theta2'] = 0
        params['theta3'] = 0
    else:
        params['theta0'] = t_set.th0
        params['theta1'] = t_set.th1
        params['theta2'] = t_set.th2
        params['theta3'] = t_set.th3

    params['routine'] = t_set.method
    params['gridpoints'] = points
    params['current_condition'] = t_set.blockade_condition_via_current
    return params


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
