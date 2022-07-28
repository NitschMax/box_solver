import setup as set
import time_evolution as te

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

	sys.solve(qdq=False, rotateq=False)


	#sys	= qmeq.Builder_many_body(Ea=t_set.Ea, Na=t_set.par, Tba=t_set.tunnel, dband=t_set.dband, mulst=t_set.mu_lst, tlst=t_set.T_lst, kerntype=t_set.method, itype=t_set.itype, countingleads=t_set.leads )
	#sys.solve(qdq=False, rotateq=False)

	print(sys.current)
	fig, ax	= plt.subplots(1,1)
	logx	= True
	logy	= True
	plot_charge	= False
	i_n	= True
	#rho0	= np.array([1, 1, 1, 1, 0, 0, 0, 0] )
	rho0	= sys.phi0
	t	= 10**np.linspace(-1, 2, 100)
	te.finite_time_plot(ax, sys, rho0, t, logx=logx, logy=logy, plot_charge=plot_charge, i_n=i_n)
	plt.show()


if __name__=='__main__':
	main()
