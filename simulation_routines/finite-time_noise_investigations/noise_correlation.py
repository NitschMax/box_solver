import setup as set
import time_evolution as te

import fock_basis_rotation as fbr
import fock_class as fc


import matplotlib.pyplot as plt
import numpy as np
import qmeq

def main():
	np.set_printoptions(precision=6)

	t_set	= set.create_transport_setup()
	t_set.assign_box(set.create_box() )
	t_set.connect_box()

	sys	= t_set.build_qmeq_sys()
	#sys	= qmeq.Builder_many_body(Ea=t_set.Ea, Na=t_set.par, Tba=t_set.tunnel, dband=t_set.dband, mulst=t_set.mu_lst, tlst=t_set.T_lst, kerntype=t_set.method, itype=t_set.itype, countingleads=t_set.leads )

	sys.solve(qdq=False, rotateq=False)
	print(sys.current)
	fig, ax	= plt.subplots(1,1)
	logx	= True
	logy	= True
	plot_charge	= False
	i_n	= True
	rho0	= np.array([1, 1, 1, 1, 0, 0, 0, 0] )
	t	= 10**np.linspace(-1, 2, 100)
	te.finite_time_plot(ax, sys, rho0, t, logx=logx, logy=logy, plot_charge=plot_charge, i_n=i_n)
	plt.show()


def majorana_noise_box(tc, energies):
	overlaps	= np.diag(energies, k=1)
	maj_op		= [fc.maj_operator(index=0, lead=[0], coupling=[tc[0]]), fc.maj_operator(index=1, lead=[0,1], coupling=[tc[1],tc[2]]), \
					fc.maj_operator(index=2, lead=[1], coupling=[tc[3]]), fc.maj_operator(index=3, lead=[2], coupling=[tc[4]]) ]
	par		= np.array([0,0,1,1])
	return maj_op, overlaps, par

def abs_noise_box(tb10, tb11, tb20, tb21, tm20, tm21, tm30, tm31, tt40, tt41, eps=0):
	overlaps	= fbr.default_overlaps(8, overlaps)

	maj_op		=  [fc.maj_operator(index=0, lead=[0], coupling=[tb10]), fc.maj_operator(index=1, lead=[0], coupling=[tb11]), \
				fc.maj_operator(index=2, lead=[0,1], coupling=[tb20,tm20]), fc.maj_operator(index=3, lead=[0,1], coupling=[tb21,tm21]), \
				fc.maj_operator(index=4, lead=[1], coupling=[tm30]), fc.maj_operator(index=5, lead=[1], coupling=[tm31]), \
				fc.maj_operator(index=6, lead=[2], coupling=[tt40]), fc.maj_operator(index=7, lead=[2], coupling=[tt41]) ]
	par		= np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] )
	return maj_op, overlaps, par

if __name__=='__main__':
	main()
