import setup as set
import time_evolution as te
import cyclic_blockade as cb
import help_functions as hf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import qmeq

def main():
	np.set_printoptions(precision=6)

	t_set	= set.create_transport_setup()

	#t_set.adjust_to_z_blockade()

	t_set.initialize_leads()
	t_set.initialize_box()
	t_set.connect_box()

	t_set.maj_box.print_eigenstates()
	

	sys	= t_set.build_qmeq_sys()
	sys.solve(qdq=False, rotateq=False)

	phi_idx	= 0

	fig, ax		= plt.subplots(1,1)
	phi_range	= np.linspace(0, np.pi, 100)
	phi_range	= np.pi/2+np.linspace(-1e-2, 1e-2, 40)
	current_noise_plot_phi0(sys, t_set, ax, phi_range, phi_idx)
	plt.show()

	return

	points	= 10
	fig, axes		= plt.subplots(1,2)
	logsc	= True

	phase_sweep_plot(sys, t_set, fig, axes, points, logscale=logsc)
	plt.show()
	return

	tunnel_sweep_plot(sys, t_set, fig, axes, points, logscale=logsc)
	plt.show()
	return

def sweep_func(sys, t_set, t0, t2):
	t_set.gamma_00	= hf.gamma_from_tunnel(t0)
	t_set.gamma_02	= hf.gamma_from_tunnel(t2)
	t_set.block_via_phases(lead=0)

	t_set.connect_box()
	sys.Tba	= t_set.tunnel

	sys.solve(qdq=False, rotateq=False)
	return np.array(sys.current_noise )

def tunnel_sweep_calc(sys, t_set, points):
	t1	= hf.tunnel_from_gamma(t_set.gamma_01 )
	x	= np.linspace(0, 2*t1, points)
	y	= x

	X, Y	= np.meshgrid(x, y)

	I	= np.array([sweep_func(sys, t_set, X[idx], Y[idx]) for idx in np.ndindex(X.shape ) ] )
	I	= I.reshape(X.shape + (2,) ).real

	return X/t1, Y/t1, I

def tunnel_sweep_plot(sys, t_set, fig, axes, points, logscale=False):
	X,Y,I	= tunnel_sweep_calc(sys, t_set, points)
	colorbar_plot(X, Y, I, fig, axes, logscale)
	return

def sweep_phases(sys, t_set, phi_avg, phi_diff):
	t_set.phi0	= phi_avg+phi_diff
	t_set.phi2	= phi_avg-phi_diff
	t_set.block_via_rates(lead=0)

	t_set.connect_box()
	sys.Tba	= t_set.tunnel

	sys.solve(qdq=False, rotateq=False)
	return np.array(sys.current_noise )

def phase_sweep_calc(sys, t_set, points):
	x	= np.linspace(-np.pi/2, np.pi/2, points)
	y	= x

	X, Y	= np.meshgrid(x, y)

	I	= np.array([sweep_phases(sys, t_set, X[idx], Y[idx]) for idx in np.ndindex(X.shape ) ] )
	I	= I.reshape(X.shape + (2,) ).real

	return X, Y, I

def phase_sweep_plot(sys, t_set, fig, axes, points, logscale=False):
	X,Y,I	= phase_sweep_calc(sys, t_set, points)
	colorbar_plot(X, Y, I, fig, axes, logscale)
	return

def colorbar_plot(X, Y, I, fig, axes, logscale):
	if logscale:
		c1	= axes[0].contourf(X, Y, I[:,:,1]/I[:,:,0] )
		c2	= axes[1].contourf(X, Y, I[:,:,0], locator=ticker.LogLocator() )
	else:
		c1	= axes[0].contourf(X, Y, I[:,:,1]/I[:,:,0] )
		c2	= axes[1].contourf(X, Y, I[:,:,0] )

	cbar	= fig.colorbar(c1, ax=axes[0])
	axes[0].locator_params(axis='both', nbins=5 )
	cbar.ax.locator_params(axis='y', nbins=5 )

	fs	= 16
	axes[0].tick_params(labelsize=fs)
	cbar.ax.set_title('Fano factor', size=fs)
	cbar.ax.tick_params(labelsize=fs)

	cbar	= fig.colorbar(c2, ax=axes[1])
	axes[1].locator_params(axis='both', nbins=5 )
	cbar.ax.locator_params(axis='y', nbins=5 )

	fs	= 16
	axes[1].tick_params(labelsize=fs)
	cbar.ax.set_title('mc', size=fs)
	cbar.ax.tick_params(labelsize=fs)

	return


def current_noise_plot_phi0(sys, t_set, ax, phi_range, phi_idx):
	ratio	= []
	current	= []
	
	current_from_setup(sys, t_set, np.pi/2, phi_idx)
	print(sys.phi0)

	for phi in phi_range:
		cur	= current_from_setup(sys, t_set, phi, phi_idx)
		if cur[0]<0:
			cur[0] = -cur[0]
		ratio.append(cur[1]/cur[0] )
		current.append(cur[0] )
	ax.plot(phi_range, ratio)
	#ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
	#ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func) )
	ax.set_xlabel(r'$\phi_{}$'.format(phi_idx))
	ax.set_ylabel('noise/current')
	#ax.set_ylim( [ 0.5, 2] )
	ax.grid(True)

	ax_twin	= ax.twinx()
	ax_twin.plot(phi_range, current, c='r')
	#ax_twin.set_ylim(bottom=0)
	ax_twin.set_ylabel(r'current [$e\Gamma$]')
	ax_twin.yaxis.get_label().set_color('r')

def current_from_setup(sys, t_set, phi, phi_idx):
	if phi_idx==0:
		t_set.phi0	= phi
	elif phi_idx==1:
		t_set.phi1	= phi
	elif phi_idx==2:
		t_set.phi2	= phi
	elif phi_idx==3:
		t_set.phi3	= phi

	t_set.initialize_box()
	t_set.connect_box()
	sys.Tba	= t_set.tunnel

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

	

if __name__=='__main__':
	main()
