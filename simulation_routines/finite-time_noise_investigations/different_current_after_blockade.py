import setup as set

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm

def main():
    thetas  = np.linspace(0, np.pi, 4)

    t_set   = set.create_transport_setup()
    t_set.initialize_leads()
    t_set.initialize_box()

    fig, axes   = plt.subplots(2,1)
    #theta0_variation_plot(fig, axes[0], t_set, thetas)

    points      = 5
    theta01_sweep_plot(fig, axes, t_set, points)
    fig.tight_layout()
    plt.show()

def theta01_sweep_plot(fig, axes, t_set, points):
    # create some sample data
    dphi    = 1e-2
    X, Y = np.meshgrid(np.linspace(-dphi, np.pi+dphi, points), np.linspace(-dphi, np.pi+dphi, points))

    factors = [1.0, 0.5]


    for idx, ax in enumerate(axes):
        t_set.factor0   = factors[idx]
        t_set.factor1   = factors[idx]

        Z   = np.abs(np.array([current_calculation(t_set, X[idx], Y[idx]) for idx, dummy in np.ndenumerate(X) ] ).reshape(X.shape+(2,) ) )
        data    = np.abs(Z[:,:,0]-Z[:,:,1] )/Z[:,:,1]
        c = ax.contourf(X, Y, data, norm=LogNorm(), locator=ticker.LogLocator())
        
        # Set axis labels and title
        ax.set_xlabel(r"$\theta_0$")
        ax.set_ylabel(r"$\theta_1$")
        
        # Create colorbar
        cb = fig.colorbar(c)
        cb.ax.set_title(r"$|\Delta I|$")    

def current_calculation(t_set, x, y):
    t_set.th0   = x
    t_set.th1   = y

    t_set.adjust_to_z_blockade()
    gamma_0     = t_set.gamma_00
    gamma_1     = t_set.gamma_01

    t_set.gamma_01  = 0.0
    t_set.gamma_00  = gamma_0
    t_set.initialize_box()
    t_set.connect_box()
    sys = t_set.build_qmeq_sys()
    sys.solve(qdq=False, rotateq=False)
    current     = sys.current[0]
    
    t_set.gamma_01  = gamma_1
    t_set.gamma_00  = 0.0
    t_set.initialize_box()
    t_set.connect_box()
    sys = t_set.build_qmeq_sys()
    sys.solve(qdq=False, rotateq=False)
    return [current, sys.current[0]]

def theta0_variation_plot(fig, ax, t_set, thetas):
    current_0   = []
    current_1   = []
    for theta in thetas:
        t_set.th0   = theta
        t_set.adjust_to_z_blockade()
        gamma_0     = t_set.gamma_00
        gamma_1     = t_set.gamma_01

        t_set.gamma_01  = 0
        t_set.initialize_box()
        t_set.connect_box()
        sys = t_set.build_qmeq_sys()
        sys.solve(qdq=False, rotateq=False)
        current_0.append(sys.current[0] )
    
        t_set.gamma_01  = gamma_1
        t_set.gamma_00  = 0.0
        t_set.initialize_box()
        t_set.connect_box()
        sys = t_set.build_qmeq_sys()
        sys.solve(qdq=False, rotateq=False)
        current_1.append(sys.current[0] )


    current = np.transpose(np.array([current_0, current_1]) )

    ax.scatter(thetas, current[:,0], marker='x', label='site 0' )
    ax.scatter(thetas, current[:,1], marker='x', label='site 1' )
    thetas  = np.linspace(0, np.pi, 1000)

    ax.plot(thetas, np.transpose(func(thetas)[2] ), label='analytics' )
    ax.legend()

def func(theta):
    gamma_plus  = 2*(1+np.sin(theta))/np.abs(np.cos(theta) )
    gamma_minus = 2*(1-np.sin(theta))/np.abs(np.cos(theta) )
    gamma_drain = 4
    gamma_eff   = np.abs(np.cos(theta) )
    gamma_eff   = 1/(1/gamma_plus+1/gamma_minus+2/gamma_drain)
    return np.array([gamma_plus, gamma_minus, 2*gamma_eff] )

if __name__=='__main__':
    main()
