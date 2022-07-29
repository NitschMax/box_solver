import numpy as np
from sys import path
path.append('../../classes')

import help_functions as hf

import edge_class as edger
import fock_class as fc
import box_class as bc
import transport_setup_class as tsc


def create_transport_setup():
	model		= 3		# 1: Majorana box, 	2: Box with ABSs on every connection	3: Two ABSs and two Majoranas
	box_symmetry	= 2		# 1: Simple Box,	2: Asymmetric Box,			3: Asymmetric Box with three leads
	
	counting_leads		= [0]
	i_n			= False		### Flag to determine wether to include noise calculations; i_n means include_noise
	
	### Overlaps between Majorans with numbers 0, 1, 2, 3
	eps01 	= 1e-3
	eps12 	= 0e-3
	eps23	= 2e-3
	
	### Overlaps between ABSs
	eps	= 0e-6
	
	dphi	= 1e-6
	
	### Rates for the connections between leads 0,1,e and edges 0,1,2,3
	gamma_00	= 1e-0
	gamma_01	= 1e-0
	gamma_02	= 0e-0		## Only relevant for asymetric Box
	
	gamma_e2	= 1e-0		## Only relevant for simple box
	gamma_e3	= 1e-0
	
	gamma_11	= 0e-3		## Only reelvant for asym Box with three leads
	gamma_12	= 0e-3		## Only reelvant for asym Box with three leads
	
	### Phases of the edges 0, 1, 2, 3
	phi0	= +1/2*np.pi-dphi
	phi1	= 0
	phi2	= +0/2*np.pi-dphi
	phi3	= 0
	
	### Wavefct factors for second Majoranas at each edge; only relevant for ABSs
	factor0	= 0
	factor1	= 0
	### Irrelevant for model 3
	factor2	= 1
	factor3	= 1
	
	### Relative wavefct phase-angles for second Majoranas at each edge; only relevant for ABSs
	th0	= 0.30*np.pi
	th1	= 0.10*np.pi
	### Irrelevant for model 3
	th2	= 0.00*np.pi
	th3	= 0.00*np.pi
	
	T_0	= 1e2
	T_1 	= T_0
	T_e 	= T_0
	
	v_bias	= 2e3
	mu_0	= v_bias/2
	mu_1	= mu_0
	mu_e	= -v_bias/2
	
	Vg	= 0
	dband	= 1e5
	
	method	= 'pyRedfield'
	method	= 'pyPauli'
	method	= 'pyLindblad'
	method	= 'py1vN'
	itype	= 1

	t_set	= tsc.transport_setup(dband, method, itype, counting_leads, i_n, model, box_symmetry, eps01, eps12, eps23, eps, dphi, \
			gamma_00, gamma_01, gamma_02, gamma_e2, gamma_e3, gamma_11, gamma_12, phi0, phi1, phi2, phi3, factor0, factor1, factor2, factor3, th0, th1, th2, th3, T_0, T_1, T_e, v_bias, Vg )
	return t_set





