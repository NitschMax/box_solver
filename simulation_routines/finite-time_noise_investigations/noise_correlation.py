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

	current	= []
	for k in range(3):
		t_set.counting_leads	= [k]
		sys	= t_set.build_qmeq_sys()
		sys.solve(qdq=False, rotateq=False)

		current.append(sys.current_noise )
	
	current	= np.array(current)
	print(current)
	noise_correlation	= current[2,1] - current[0,1] - current[1,1]
	print(noise_correlation)

	return

if __name__=='__main__':
	main()
