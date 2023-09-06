from copy import copy

import help_functions as hf
import matplotlib.pyplot as plt
import numpy as np
import qmeq

import quasi_stationary_investigation as qsi
import setup as set
import time_evolution as te


def main():
    t_set_z = set.create_transport_setup()
    t_set_z.initialize_leads()
    t_set_z.adjust_to_z_blockade()
    t_set_z.initialize_box()
    t_set_z.connect_box()

    t_set_x = t_set_z.copy()

    t_set_x.adjust_to_x_blockade()
    t_set_x.initialize_leads()
    t_set_x.initialize_box()
    t_set_x.connect_box()

    sys_z = t_set_z.build_qmeq_sys()
    sys_x = t_set_x.build_qmeq_sys()

    sys_z.solve(qdq=False, rotateq=False)
    sys_x.solve(qdq=False, rotateq=False)
    print(sys_z.current, sys_x.current)

    waiting_time = 1e2
    n = 1
    pre_run = 1e3
    qs_desc = False
    include_Irem = True

    initialization = 2  ### Initialization routines 0: stationary state of system;
    ### 1: 1 at first occupation, 0 otherwise; 2: maximally mixed state
    if t_set_z.model == 2:
        timescale = 1 / (t_set_z.eps_abs_0 / t_set_z.gamma_01)
    elif t_set_z.model == 1:
        timescale = 1
    timescale = 1

    qsi.analytical_prediction(t_set_z.th2)

    rho0, sys_z, sys_x, current_fct_z, current_fct_x, time_evo_rho_z, time_evo_rho_x = initialize_cycle_fcts(
        t_set_z, t_set_x, qs_desc, initialization, lead=0)

    ### Screw what I wrote about initialization of rho0 before ##
    #rho0   = qsi.initialize_rho(sys_x, t_set_x, rho0)

    #average_charge_of_cycle, overlap   = charge_transmission_cycle(current_fct_z, current_fct_x, time_evo_rho_z, time_evo_rho_x, rho0, n, waiting_time, sys, pre_run)
    #print('The choosen cycle setup transmits per switch: ', average_charge_of_cycle )

    fig, ax = plt.subplots(1, 1)
    angles = np.linspace(0, np.pi / 2 - 1e-2, 4)
    #charge_angle_plot(ax, angles, t_set_z, t_set_x, n, pre_run, waiting_time, timescale=1)
    #plt.show()

    waiting_times = np.power(10, np.linspace(+1, 7, 4))
    charge_waiting_plot(ax,
                        waiting_times,
                        current_fct_z,
                        current_fct_x,
                        time_evo_rho_z,
                        time_evo_rho_x,
                        rho0,
                        n,
                        pre_run,
                        sys_z,
                        timescale=timescale,
                        include_Irem=include_Irem)
    plt.show()
    return

    den_mat_cycle_plot(ax,
                       time_evo_rho_z,
                       time_evo_rho_x,
                       rho0,
                       waiting_time,
                       pre_run=1e3)
    plt.show()
    return


def charge_angle_plot(ax,
                      angles,
                      t_set_z,
                      t_set_x,
                      n,
                      pre_run,
                      waiting_time,
                      timescale=1,
                      include_I_rem=False):
    charge = []
    for angle in angles:
        t_set_z.th2 = angle
        t_set_x.th2 = angle

        t_set_z.initialize_leads()
        t_set_z.adjust_to_z_blockade()
        t_set_z.initialize_box()
        t_set_z.connect_box()

        t_set_x.initialize_leads()
        t_set_x.adjust_to_x_blockade()
        t_set_x.initialize_box()
        t_set_x.connect_box()

        rho0, sys_z, sys_x, current_fct_z, current_fct_x, time_evo_rho_z, time_evo_rho_x = initialize_cycle_fcts(
            t_set_z, t_set_x, qs_desc=False, initialization=2, lead=0)
        charge.append(
            charge_transmission_cycle(current_fct_z,
                                      current_fct_x,
                                      time_evo_rho_z,
                                      time_evo_rho_x,
                                      rho0,
                                      n,
                                      waiting_time,
                                      sys_z,
                                      pre_run=pre_run,
                                      include_Irem=include_Irem))

    charge = np.array(charge)
    ax.scatter(angles, charge, marker='x', label='Numverical results')
    angles = np.linspace(0, np.pi / 2 - 1e-2, 100)
    analytic_result = np.array(
        [2 * qsi.analytical_prediction(angle) for angle in angles])
    ax.plot(angles, analytic_result, label='Analytical prediction')
    ax.set_xlabel(r'$\Theta_2$')
    ax.set_ylabel('Charge transmission per cycle')
    ax.legend()
    ax.grid(True)


def den_mat_cycle_plot(ax,
                       time_evo_rho_z,
                       time_evo_rho_x,
                       rho0,
                       waiting_time,
                       pre_run=1e3):
    diff_z, diff_x = den_mat_cycle(time_evo_rho_z,
                                   time_evo_rho_x,
                                   rho0,
                                   waiting_time,
                                   pre_run=pre_run)
    ax.plot(diff_z, label='Matrix-diff of z-blockade to x')
    ax.plot(diff_x, label='Matrix-diff of x-blockade to x')
    ax.legend()
    ax.grid(True)
    return


def charge_waiting_plot(ax,
                        waiting_times,
                        current_fct_z,
                        current_fct_x,
                        time_evo_rho_z,
                        time_evo_rho_x,
                        rho0,
                        n,
                        pre_run,
                        sys,
                        timescale=1,
                        include_Irem=False):
    #print('Neglected charge of the order', timescale**(-2)/2*waiting_times[-1])
    charge = np.array([
        charge_transmission_cycle(current_fct_z,
                                  current_fct_x,
                                  time_evo_rho_z,
                                  time_evo_rho_x,
                                  rho0,
                                  n,
                                  waiting_time,
                                  sys,
                                  pre_run=pre_run,
                                  include_Irem=include_Irem)
        for waiting_time in waiting_times
    ])
    labels = ['Integrated charge']
    ax.scatter(waiting_times / timescale, charge)
    #ax.plot(waiting_times/timescale, timescale**(-2)*waiting_times, label='Charge neglected by integration')
    ax.set_xlabel(r'Blockade time $[ \epsilon^{-1} ]$')
    ax.set_ylabel('Charge per cycle [e]')
    ax.set_xscale('log')
    #ax.set_ylim([0, 1.2])
    ax.legend(labels=labels)
    ax.grid(True)
    return


def initialize_cycle_fcts(t_set_z, t_set_x, qs_desc, initialization=0, lead=0):
    sys_z = t_set_z.build_qmeq_sys()
    sys_z.solve(qdq=False, rotateq=False)
    time_evo_rho_z = te.finite_time_evolution(sys_z, qs_desc)
    current_fct_z = te.current(sys_z)

    rho0 = state_preparation(sys_z, initialization)

    sys_x = t_set_x.build_qmeq_sys()
    sys_x.solve(qdq=False, rotateq=False)
    time_evo_rho_x = te.finite_time_evolution(sys_x, qs_desc)
    current_fct_x = te.current(sys_x)

    return rho0, sys_z, sys_x, current_fct_z, current_fct_x, time_evo_rho_z, time_evo_rho_x


def den_mat_cycle(time_evo_1, time_evo_2, rho0, T, pre_run=1e3):
    rho = rho0
    n = int(pre_run)
    difference_1 = np.zeros(n)
    difference_2 = np.zeros(n)

    print('Pre-running system {} cycles'.format(n))
    for k in range(n):
        rho = time_evo_1(rho, T)
        rho = time_evo_2(rho, T)

    rho_lim = rho
    rho = rho0

    print('Measuring density matrix convergence {} cycles'.format(n))
    for k in range(n):
        rho = time_evo_1(rho, T)
        difference_1[k] = hf.measure_of_vector(rho, rho_lim)

        rho = time_evo_2(rho, T)
        difference_2[k] = hf.measure_of_vector(rho, rho_lim)

    return difference_1, difference_2


def charge_transmission_cycle(current_fct_1,
                              current_fct_2,
                              time_evo_1,
                              time_evo_2,
                              rho0,
                              n,
                              T,
                              sys,
                              pre_run=1e3,
                              include_Irem=False):
    rho = rho0
    charge_transmitted = 0

    if T > 10:
        integration_range = 10
    else:
        integration_range = T

    # print('Pre-running system {} cycles'.format(pre_run))
    for k in range(int(pre_run)):
        rho = time_evo_1(rho, T)
        rho = time_evo_2(rho, T)
    epsrel = 1e-1

    for i in range(n):
        # print('Cycle {} of {}'.format(i + 1, n))
        additional_charge = te.charge_transmission(current_fct_1,
                                                   time_evo_1,
                                                   rho,
                                                   tzero=0,
                                                   tau=integration_range)[0]
        if include_Irem and T > integration_range:
            additional_charge += te.charge_transmission(
                current_fct_1,
                time_evo_1,
                rho,
                tzero=integration_range,
                tau=T,
                epsrel=epsrel)[0]
        charge_transmitted += additional_charge
        rho = time_evo_1(rho, T)

        additional_charge = te.charge_transmission(current_fct_2,
                                                   time_evo_2,
                                                   rho,
                                                   tzero=0,
                                                   tau=integration_range)[0]
        if include_Irem and T > integration_range:
            additional_charge += te.charge_transmission(
                current_fct_2,
                time_evo_2,
                rho,
                tzero=integration_range,
                tau=T,
                epsrel=epsrel)[0]
        charge_transmitted += additional_charge
        rho = time_evo_2(rho, T)

    return charge_transmitted / (1 * n)


def state_preparation(sys, initialization=0):
    rho0 = sys.phi0.copy()
    if initialization == 1:
        rho0[0] = 1
        rho0[1:] = 0
    elif initialization == 2:
        num_occ, dof = te.model_spec_dof(sys.phi0)
        rho0[:num_occ] = 1
        rho0[num_occ:] = 0
        rho0 = te.normed_occupations(rho0)
    return rho0


if __name__ == '__main__':
    main()
