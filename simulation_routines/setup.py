import numpy as np
import help_functions_setup as hf
import edge_class as edger
import fock_class as fc
import box_class as bc


model		= 1		# 1: Majorana box, 	2: Box with ABSs on every connection
box_symmetry	= 2		# 1: Simple Box,	2: Asymmetric Box,			3: Asymmetric Box with three leads

leads		= [1]

### Overlaps between Majorans with numbers 0, 1, 2, 3
eps01 	= 0e-5
eps12 	= 1e-3
eps23	= 0e-5

### Overlaps between ABSs
eps	= 1e-3

dphi	= 1e-6

### Rates for the connections between edges and leads 0, 1, e
gamma_0		= 1e-0
gamma_1		= 1e-3
gamma_e		= 1e-0

### Phases of the edges 0, 1, 2, 3
phi0	= +1/2*np.pi-dphi
phi1	= 0
phi2	= +0/2*np.pi-dphi
phi3	= 0

### Wavefct factors for second Majoranas at each edge; only relevant for ABSs
factor0	= 1
factor1	= 1
factor2	= 1
factor3	= 1

### Relative wavefct phase-angles for second Majoranas at each edge; only relevant for ABSs
th0	= 0.00
th1	= 0.00
th2	= 0.30
th3	= 0.00

T_0	= 1e2
T_1 	= T_0
T_e 	= T_0

v_bias	= 2e3
mu_0	= v_bias/2
mu_1	= mu_0
mu_e	= -v_bias/2

dband	= 1e5

if box_symmetry == 3:
	T_lst 	= { 0:T_0, 1:T_1, 2:T_e}
	mu_lst 	= { 0:mu_0, 1:mu_1, 2:mu_e}
elif box_symmetry == 1 or box_symmetry == 2:
	T_lst 	= { 0:T_0, 1:T_e}
	mu_lst 	= { 0:mu_0, 1:mu_e}

method	= 'pyRedfield'
method	= 'pyPauli'
method	= 'pyLindblad'
method	= 'py1vN'
itype	= 1

def do_setup():
	if box_symmetry == 1:
		edgy0 	= edger.edge(phi0, [0], [gamma_0])
		edgy1 	= edger.edge(phi1, [0], [gamma_0])
		edgy2 	= edger.edge(phi2, [1], [gamma_e])
		edgy3 	= edger.edge(phi3, [1], [gamma_e])
	elif box_symmetry == 2:
		edgy0 	= edger.edge(phi0, [0], [gamma_0])
		edgy1 	= edger.edge(phi1, [0], [gamma_0])
		edgy2 	= edger.edge(phi2, [0], [gamma_0])
		edgy3 	= edger.edge(phi3, [1], [gamma_e])
	elif box_symmetry == 3:
		edgy0 	= edger.edge(phi0, [0], [gamma_0])
		edgy1 	= edger.edge(phi1, [0, 1], [gamma_0, gamma1])
		edgy2 	= edger.edge(phi2, [1], [gamma_1])
		edgy3 	= edger.edge(phi3, [2], [gamma_e])
	
	edgy0.create_majorana(0)
	edgy1.create_majorana(1, overlaps={0: eps01})
	edgy2.create_majorana(2, overlaps={1: eps12})
	edgy3.create_majorana(3, overlaps={2: eps23})
	
	if model == 2:
		edgy0.create_majorana(4, wf_factor=factor0, wf_phase_angle=th0, overlaps={0: eps})
		edgy1.create_majorana(5, wf_factor=factor1, wf_phase_angle=th1, overlaps={1: eps})
		edgy2.create_majorana(6, wf_factor=factor2, wf_phase_angle=th2, overlaps={2: eps})
		edgy3.create_majorana(7, wf_factor=factor3, wf_phase_angle=th3, overlaps={3: eps})
	elif model == 3:
		edgy0.create_majorana(4, wf_factor=factor0, wf_phase_angle=th0, overlaps={0: eps})
		edgy1.create_majorana(5, wf_factor=factor1, wf_phase_angle=th1, overlaps={1: eps})
	
	edge_lst	= np.array([edgy0, edgy1, edgy2, edgy3], dtype='object')
	Vg		= +0e1
	maj_box		= bc.majorana_box.box_via_edges(edge_lst, Vg)
	maj_box.diagonalize()
	#maj_box.print_eigenstates()
	
	Ea		= maj_box.elec_en
	par		= maj_box.par
	tunnel		= maj_box.constr_tunnel()
	return Ea, par, tunnel

Ea, par, tunnel	= do_setup()





