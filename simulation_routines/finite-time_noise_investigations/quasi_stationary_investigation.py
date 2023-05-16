import cyclic_blockade as cb
import setup as set
import time_evolution as te

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

import qmeq

def main():
    np.set_printoptions(precision=2)
    pre_run = False
    pre_run = True
    
    x_blockade  = False
    z_blockade  = True
    sys, t_set, rho0    = previous_run(x_blockade=x_blockade, z_blockade=z_blockade)
    sys_2, t_set_2, rho0_2  = previous_run(x_blockade=not x_blockade, z_blockade=not z_blockade)
    sys_3, t_set_3, rho0_3  = previous_run(x_blockade=False, z_blockade=False, disconnect_source=True)

    rhoTildes, basis_indices    = qss_basis(sys, t_set, rho0)

    logx    = False

    time_order  = 1
    points      = int(1e3 )

    fig, axes   = plt.subplots(2,1)
    waiting_times   = [6, 2]
    xlabel      = False
    for idx, T in enumerate(waiting_times):
        print(idx)
        if idx == 1:
            print(xlabel)
            xlabel  = True
        ax  = axes[idx]
        cyclic_overlap_plot(fig, ax, T, T/4, points, sys, sys_2, sys_3, rho0, rhoTildes, basis_indices, pre_run, logx=logx)
        majorana_overlap_paper_plot(ax, T, legend=False, xlabel=xlabel)
        fig.tight_layout()
    plt.show()
    return


    #t_set.maj_box.print_eigenstates()

    #rho0   = initialize_rho_odd(sys, t_set, rho0)

    if t_set.model == 2:
        for idx in range(16):
            print(idx, t_set.maj_box.states.get_state(idx) )

    rho0    = rotate_rhoVec_into_eigenbasis(sys, t_set, rho0 )

    if logx:
        time    = 10**np.linspace(-2, time_order, points)
    else:
        time    = np.linspace(0, T, points)

    time_int    = np.linspace(0, 1, 500)
    analytical_prediction(-t_set.th2 )
    fig, ax = plt.subplots(1,1)
    rho0    = overlap_plot(fig, ax, time, sys, t_set, rho0_2, rhoTildes, basis_indices, logx=logx)
    majorana_overlap_paper_plot(ax)
    fig.tight_layout()
    plt.show()
    return

    for k in range(4):
        print(k)
        fig, axes   = plt.subplots(2,2)
        rho0_2  = overlap_plot(fig, axes[0,0], time, sys, t_set, rho0, rhoTildes, basis_indices, logx=logx)
        #te.finite_time_plot(axes[0,1], sys, rho0, time_int, lead=[0], logx=False, logy=False, plot_charge=True, i_n=False, qs_desc=False )

        rho0    = overlap_plot(fig, axes[1,0], time, sys_2, t_set_2, rho0_2, rhoTildes, basis_indices, logx=logx)
        #te.finite_time_plot(axes[1,1], sys_2, rho0_2, time_int, lead=[0], logx=False, logy=False, plot_charge=True, i_n=False, qs_desc=False )
        plt.show()
        plt.close()

def cyclic_overlap_plot(fig, ax, block_time, change_time, N, sys, sys_2, sys_3, rho0, rhoTildes, basis_indices, pre_run, logx=False):
    time        = np.linspace(-change_time, 3*block_time+2*change_time, N)

    time_evo    = te.finite_time_evolution(sys, qs_desc=False)
    time_evo_2  = te.finite_time_evolution(sys_2, qs_desc=False)
    time_evo_3  = te.finite_time_evolution(sys_3, qs_desc=False)

    rho0_2      = rho0.copy()
    rho0_prime  = rho0.copy()
    rho0_2_prime    = rho0.copy()

    if pre_run:
        for k in range(int(1e3) ):
            rho0_prime  = time_evo(rho0, block_time)
            rho0_2      = time_evo_3(rho0_prime, change_time)
            rho0_2_prime    = time_evo_2(rho0_2, block_time)
            rho0        = time_evo_3(rho0_2_prime, change_time)

    overlaps    = []
    for t in time:
        if t <= 0:
            rho_at_t    = time_evo_3(rho0_2_prime, t+change_time)
        elif t <= block_time:
            rho_at_t    = time_evo(rho0, t)
        elif t <= block_time+change_time:
            rho_at_t    = time_evo_3(rho0_prime, t-block_time)
        elif t <= 2*block_time+change_time:
            rho_at_t    = time_evo_2(rho0_2, t-block_time-change_time)
        elif t <= 2*block_time+2*change_time:
            rho_at_t    = time_evo_3(rho0_2_prime, t-2*block_time-change_time)
        elif t <= 3*block_time+2*change_time:
            rho_at_t    = time_evo(rho0, t-2*block_time-2*change_time)
        
        overlaps.append([overlap_routine(sys, rhoT, rho_at_t ) for rhoT in rhoTildes] )

    overlaps    = np.array(overlaps )
    overlaps[:,2]   += overlaps[:,3]
    overlaps    = overlaps[:,:3]

    lw = 2
    colors  = ['blue', 'green', 'orange']
    for idx in range(3):
        ax.plot(time/block_time, overlaps[:,idx], '-', lw=lw, c=colors[idx] ) 
    ax.set_ylim([0,1])
    ax.set_xlim(np.array([-0.4*change_time, 2.2*block_time+2.0*change_time])/block_time )
    if logx:
        ax.set_xscale('log')
    ax.locator_params(axis='y', nbins=3)
    ax.axes.xaxis.set_ticklabels([])
    return np.array(time_evo(rho0, time[-1] ) )

def majorana_overlap_paper_plot(ax, block_time, legend=True, xlabel=True):
    fs  = 20
    if xlabel:
        ax.set_xlabel(r'time', size=fs)
    ax.set_ylabel('Occupation', size=fs)
    if legend:
        ax.legend(['00', '11', '10', '01'], prop={'size':fs})
    ax.tick_params(labelsize=fs)
    ax.set_title(r'$\tau={}/\Gamma$'.format(block_time), size=fs, x=0.70, y=+0.70)
    

def initialize_rho(sys, t_set, rho0):
    if t_set.model == 1:
        return rho0
    e0, e1, o0, o1  = rinotation_parameters(-t_set.th2)
    idx0        = 0
    idx1        = 4
    rho0[:] = 0
    rho0[ idx0] = np.abs(e0)**2
    rho0[ idx1] = np.abs(e1)**2
    rho0Mat     = te.map_vec_to_den_mat(sys, rho0)
    rho0Mat[idx0,idx1]  = e1*np.conj(e0 )
    rho0Mat[idx1,idx0]  = np.conj(e1)*e0
    rho0        = te.map_den_mat_to_vec(sys, rho0Mat)
    return rho0

def initialize_rho_odd(sys, t_set, rho0):
    if t_set.model == 1:
        return rho0
    e0, e1, o0, o1  = rotation_parameters(-t_set.th2)
    idx0        = 5
    idx1        = 6
    rho0[:] = 0
    rho0[ idx0] = np.abs(o0)**2
    rho0[ idx1] = np.abs(o1)**2
    rho0Mat     = te.map_vec_to_den_mat(sys, rho0)
    rho0Mat[idx0,idx1]  = -o1*np.conj(o0 )
    rho0Mat[idx1,idx0]  = -np.conj(o1)*o0

    rho0        = te.map_den_mat_to_vec(sys, rho0Mat)
    return rho0

def analytical_prediction(theta):
    e0, e1, o0, o1  = rotation_parameters(theta)
    ePlus       = np.abs(e0+1j*e1)
    eMinus      = np.abs(e0-1j*e1)
    prediction  = 2*ePlus**2/4*eMinus**2/2+2*eMinus**2/4*ePlus**2/2
    print('ePlus squared: ', ePlus**2 )
    print('eMinus squared: ', eMinus**2 )
    print('Analytically predicted charge: ', prediction )
    return prediction

def rotation_parameters(theta):
    renormPlus  = np.sqrt(2*(1+1/np.cos(theta) ) )
    e0  = (1/np.sqrt(np.cos(theta))*np.exp(-1j*theta/2)+1 )/renormPlus
    e1  = (1/np.sqrt(np.cos(theta))*np.exp(+1j*theta/2)-1 )/renormPlus
    
    renormMinus = np.sqrt(2*(1+1/np.cos(theta)+2/np.sqrt(np.cos(theta) )*np.cos(theta/2) ) )
    o0  = (1/np.sqrt(np.cos(theta))*np.exp(+1j*theta/2)+1 )/renormMinus
    o1  = (1/np.sqrt(np.cos(theta))*np.exp(-1j*theta/2)+1 )/renormMinus

    return e0, e1, o0, o1

def overlap_plot(fig, ax, time, sys, t_set, rho0, rhoTildes, basis_indices, logx=False):
    overlaps    = []
    time_evo    = te.finite_time_evolution(sys, qs_desc=False)
    for t in time:
        rho_at_t    = np.array(time_evo(rho0, t) )
        overlaps.append([overlap_routine(sys, rhoT, rho_at_t ) for rhoT in rhoTildes] )

    overlaps    = np.array(overlaps )
    ax.plot(time, overlaps, '-' ) 
    if logx:
        ax.set_xscale('log')
    if len(basis_indices) < 20:
        ax.legend(basis_indices, fontsize=12)
    ax.grid(True)
    return np.array(time_evo(rho0, time[-1] ) )

def qss_basis(sys, t_set, rhoEx):
    rho = rhoEx.copy()
    rho[:]  = 0

    result  = []
    indices = [1,2]
    indices = list(range(16) )
    if t_set.model == 2:
        indices = list(range(0, 16) )
    else:
        indices = list(range(4) )
    print('Basis indices included ', indices)

    for idx in indices:
        rhoT        = rho.copy()
        rhoT[idx]   = 1
        rhoTmat = te.map_vec_to_den_mat(sys, rhoT)
        rhoTmat = rotate_into_eigenbasis(t_set, rhoTmat)
        result.append(rhoTmat )

    result  = np.array(result)
    return result, indices

def overlap_routine(sys, rhoMat, rhoVec):
    return np.trace(np.dot(rhoMat, te.map_vec_to_den_mat(sys, rhoVec) ) ).real

def rotate_rhoVec_into_eigenbasis(sys, t_set, rhoVec):
    result  = te.map_vec_to_den_mat(sys, rhoVec)
    return te.map_den_mat_to_vec(sys, rotate_into_eigenbasis(t_set, result) )

def rotate_into_eigenbasis(t_set, rhoMat):
    return np.dot(t_set.maj_box.U.getH(), np.dot(rhoMat, t_set.maj_box.U) )

def previous_run(x_blockade=False, z_blockade=False, disconnect_source=False):
    t_set   = set.create_transport_setup()
    t_set.initialize_leads()
    t_set.initialize_box()

    t_set.connect_box()
    #t_set.block_via_phases()
    if x_blockade:
        print('Block to x')
        t_set.adjust_to_x_blockade()
    if z_blockade:
        print('Block to z')
        t_set.adjust_to_z_blockade()
    if disconnect_source:
        print('Block completely')
        t_set.disconnect_from_source()
    t_set.connect_box()

    print(t_set.gamma_01)
    sys = t_set.build_qmeq_sys()

    sys.solve(qdq=False, rotateq=False)
    rho0        = sys.phi0.copy()
    print('Stationary current: ', sys.current )

    return sys, t_set, rho0
    

if __name__=='__main__':
    main()


