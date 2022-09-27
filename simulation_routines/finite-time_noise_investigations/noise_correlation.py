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

	fig, ax	= plt.subplots(1,1)
	phi_range	= np.linspace(0, 1*np.pi, 100)
	phi_idx		= 2
	correlation_plot(sys, t_set, ax, phi_range, phi_idx)
	plt.show()


def correlation_plot(sys, t_set, ax, phi_range, phi_idx):
	n_cor	= []
	for phi in phi_range:
		n_cor.append(correlation_from_setup(sys, t_set, phi, phi_idx) )

	n_cor	= np.array(n_cor)
	ax.plot(phi_range, n_cor[:,1])
	ax.grid(True)

	ax_t	= ax.twinx()
	ax_t.plot(phi_range, n_cor[:,0], c='r')
	ax_t.set_ylabel('current')
	return n_cor

def correlation_from_setup(sys, t_set, phi, phi_idx):
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

	lead_comb	= [[0], [1], [2] ]
	current		= []
	for cl in lead_comb:
		t_set.counting_leads	= cl
		sys	= t_set.build_qmeq_sys()
		sys.solve(qdq=False, rotateq=False)

		current.append(sys.current_noise )
	
	current	= np.array(current)
	noise_correlation	= [-current[2,0], (current[2,1] - current[0,1] - current[1,1] )/2 ]
	return noise_correlation
	

if __name__=='__main__':
	main()
