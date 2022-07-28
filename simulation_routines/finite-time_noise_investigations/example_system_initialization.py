from sys import path
path.append('../../classes')
path.append('../../qmeq_noise_version')

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

