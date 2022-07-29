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

	sys.solve(qdq=False, rotateq=False)

	print(sys.current_noise)
	print(sys.current_noise[0]/sys.current_noise[1])

	return

	fig, ax	= plt.subplots(1,1)
	logx	= True
	logy	= False
	plot_charge	= False
	i_n	= t_set.i_n
	initialization	= 1
	rho0	= cb.state_preparation(sys, initialization)
	t	= 10**np.linspace(-1, 2, 100)
	te.finite_time_plot(ax, sys, rho0, t, logx=logx, logy=logy, plot_charge=plot_charge, i_n=i_n)
	plt.show()


if __name__=='__main__':
	main()
