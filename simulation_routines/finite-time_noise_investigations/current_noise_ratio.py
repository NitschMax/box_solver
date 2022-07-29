import setup as set
import time_evolution as te
import cyclic_blockade as cb

import matplotlib.pyplot as plt
import numpy as np
import qmeq

def main():
	np.set_printoptions(precision=6)

	t_set	= set.create_transport_setup()

	#t_set.adjust_to_z_blockade()

	t_set.initialize_leads()
	t_set.initialize_box()
	t_set.connect_box()
	sys	= t_set.build_qmeq_sys()
	
	phi_range	= np.pi/2+np.linspace(-1e-5, 1e-5, 100)
	phi_range	= np.linspace(0, np.pi, 400)
	fig, ax		= plt.subplots(1,1)
	current_noise_plot_phi0(sys, t_set, ax, phi_range)
	plt.show()

	return

def current_noise_plot_phi0(sys, t_set, ax, phi_range):
	ratio	= []
	current	= []
	for phi in phi_range:
		t_set.phi0	= phi
		t_set.initialize_box()
		t_set.connect_box()
		sys.Tba	= t_set.tunnel

		sys.solve(qdq=False, rotateq=False)
		ratio.append(sys.current_noise[0]/sys.current_noise[1] )
		current.append(sys.current_noise[0] )
	ax.plot(phi_range, ratio)
	ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
	ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func) )
	ax.set_xlabel(r'$\phi_0$')
	ax.set_ylabel('current/noise')
	ax.set_ylim( [ 0.5, 2] )
	ax.grid(True)

	ax_twin	= ax.twinx()
	ax_twin.plot(phi_range, current, c='g')
	ax_twin.set_ylim(bottom=0)
	ax_twin.set_ylabel(r'current [$e\Gamma$]')

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
