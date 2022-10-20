import fock_class as fc
import edge_class as ec
import box_class as bc
from qmeq import Builder_many_body

import numpy as np
from scipy.optimize import minimize
from copy import copy

class transport_setup:
	def __init__(self, dband, method, itype, counting_leads, i_n, model, box_symmetry, eps01, eps12, eps23, eps_abs_0, eps_abs_1, eps_abs_2, eps_abs_3, dphi, \
			gamma_00, gamma_01, gamma_02, gamma_e2, gamma_e3, gamma_11, gamma_12, phi0, phi1, phi2, phi3, factor0, factor1, factor2, factor3, th0, th1, th2, th3, T_0, T_1, T_e, v_bias, Vg ):
		self.dband	= dband
		self.method	= method
		self.itype	= itype
		self.counting_leads	= counting_leads
		self.i_n		= i_n
		self.model		= model			# 1: Majorana box, 	2: Box with ABSs on every connection
		self.box_symmetry	= box_symmetry		# 1: Simple Box,	2: Asymmetric Box,			3: Asymmetric Box with three leads

		### Overlaps between Majorans with numbers 0, 1, 2, 3
		self.eps01 	= eps01
		self.eps12 	= eps12
		self.eps23	= eps23

		### Overlaps between ABSs
		self.eps_abs_0	= eps_abs_0
		self.eps_abs_1	= eps_abs_1
		self.eps_abs_2	= eps_abs_2
		self.eps_abs_3	= eps_abs_3

		self.dphi	= dphi

		### Rates for the connections between leads 0,1,e and edges 0,1,2,3
		self.gamma_00	= gamma_00
		self.gamma_01	= gamma_01
		self.gamma_02	= gamma_02		## Only relevant for asymetric Box
		self.gamma_e2	= gamma_e2		## Only relevant for simple box
		self.gamma_e3	= gamma_e3
		self.gamma_11	= gamma_11		## Only reelvant for asym Box with three leads
		self.gamma_12	= gamma_12		## Only reelvant for asym Box with three leads

		### Phases of the edges 0, 1, 2, 3
		self.phi0	= phi0
		self.phi1	= phi1
		self.phi2	= phi2
		self.phi3	= phi3

		### Wavefct factors for second Majoranas at each edge; only relevant for ABSs
		self.factor0	= factor0
		self.factor1	= factor1
		self.factor2	= factor2
		self.factor3	= factor3

		### Relative wavefct phase-angles for second Majoranas at each edge; only relevant for ABSs
		self.th0	= th0
		self.th1	= th1
		self.th2	= th2
		self.th3	= th3

		### Parameters for the construction of leads
		self.T_0	= T_0
		self.T_1 	= T_1
		self.T_e 	= T_e
		self.v_bias	= v_bias
		self.Vg		= Vg
		self.dband	= dband

		### Method specification
		self.method	= method
		self.itype	= itype

		### Parameters for the qmeq Many body builder that are calculated via class routines
		self.Ea		= None
		self.par	= None
		self.tunnel	= None
		self.edges	= None
		self.maj_box	= None
		self.T_lst	= None
		self.mu_lst	= None
		self.leads_initialized	= False
		self.edges_assigned	= False
		self.box_assigned	= False
		self.box_connected	= False

	def build_qmeq_sys(self):
		if not self.box_assigned or not self.box_connected or not self.leads_initialized:
			print('Can not use builder of qmeq as no Majorana box assigned or connected')
			return None
		else:
			if self.i_n:
				return Builder_many_body(Ea=self.Ea, Na=self.par, Tba=self.tunnel, dband=self.dband, mulst=self.mu_lst, tlst=self.T_lst, kerntype=self.method, itype=self.itype, countingleads=self.counting_leads )
			else:
				return Builder_many_body(Ea=self.Ea, Na=self.par, Tba=self.tunnel, dband=self.dband, mulst=self.mu_lst, tlst=self.T_lst, kerntype=self.method, itype=self.itype)

	def adjust_to_z_blockade(self, gamma=1.0):
		if self.box_symmetry == 1:
			self.gamma_02	= 0.0
			self.gamma_01	= gamma
			self.phi1	= 0.0
			opt_func	= lambda x: self.tune_couplings_to_z_blockade(x)
			guess		= [1.0, 1/2*np.pi]					## Blockade-condition for z-blockade in a pure majorana box
			result		= minimize(opt_func, guess).x
		elif self.box_symmetry == 3:
			self.gamma_00	= gamma
			self.gamma_01	= gamma
			self.gamma_11	= 0.0
			self.gamma_12	= 0.0
		elif self.box_symmetry == 2:
			self.gamma_02	= 0.0
			self.gamma_01	= gamma
			self.phi1	= 0.0
			opt_func	= lambda x: self.tune_couplings_to_z_blockade(x)
			guess		= [1.0, 1/2*np.pi]					## Blockade-condition for z-blockade in a pure majorana box
			result		= minimize(opt_func, guess).x
			#self.tune_couplings_to_z_blockade(result )

	def tune_couplings_to_z_blockade(self, couplings):
		self.gamma_00	= couplings[0]
		self.phi0	= couplings[1]
		self.initialize_box()
		return self.maj_box.calculate_blockade_cond(lead=0)	
	
	def adjust_to_x_blockade(self, gamma=1.0):
		if self.box_symmetry == 1:
			self.adjust_to_z_blockade(gamma)
		elif self.box_symmetry == 3:
			self.gamma_00	= 0.0
			self.gamma_01	= 0.0
			self.gamma_11	= gamma
			self.gamma_12	= gamma
		elif self.box_symmetry == 2:
			self.gamma_00	= 0.0
			self.gamma_01	= gamma
			self.phi1	= 0.0
			opt_func	= lambda x: self.tune_couplings_to_x_blockade(x)
			guess		= [1.0, 1/2*np.pi]					## Blockade-condition for x-blockade in a pure majorana box
			result		= minimize(opt_func, guess).x
			#self.tune_couplings_to_x_blockade(result )

	def tune_couplings_to_x_blockade(self, couplings):
		self.gamma_02	= couplings[0]
		self.phi2	= couplings[1]
		self.initialize_box()
		return self.maj_box.calculate_blockade_cond(lead=0)	
			
	def initialize_leads(self):
		mu_0	= self.v_bias/2
		mu_1	= mu_0
		mu_e	= -self.v_bias/2

		if self.box_symmetry == 3:
			self.T_lst 	= { 0:self.T_0, 1:self.T_1, 2:self.T_e}
			self.mu_lst 	= { 0:mu_0, 1:mu_1, 2:mu_e}
		elif self.box_symmetry == 1 or self.box_symmetry == 2:
			self.T_lst 	= { 0:self.T_0, 1:self.T_e}
			self.mu_lst 	= { 0:mu_0, 1:mu_e}
		self.leads_initialized	= True

	def classify_edges(self, lead):
		potential_edges	= 1*np.array([lead in edgy.cl for edgy in self.edges], dtype='bool')		## Find out which edges are connected to lead
		return potential_edges

	def tune_phases(self, angles, lead ):
		#edge_classif	= self.classify_edges(lead)
		edge_classif	= np.array([1,0,1,0] )
		edge_phase_angles	= np.array([self.phi0, self.phi1, self.phi2, self.phi3] )
		angles			= edge_classif*angles + (1-edge_classif)*edge_phase_angles		## Make sure that only the phaseangles of edges attached to lead are changed
		self.phi0, self.phi1, self.phi2, self.phi3	= angles
		self.initialize_box()
		return self.maj_box.calculate_blockade_cond(lead)

	def block_via_phases(self, lead=0):
		print('Adjusting phases through lead {} to establish blockade.'.format(lead) )
		angles_zero	= np.array([np.pi/4, 0, np.pi/3, 0] )
		opt_func	= lambda x: self.tune_phases(x, lead)
		bnds		= ((0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi) )
		min_res		= minimize(opt_func, angles_zero, tol=self.dphi**2, options={'maxiter':1000}, bounds=bnds )
		return

	def tune_rates(self, rates, lead ):
		edge_classif	= np.array([1,0,1,0] )
		edge_rates	= np.array([self.gamma_00, self.gamma_01, self.gamma_02, self.gamma_e3] )
		rates			= edge_classif*rates + (1-edge_classif)*edge_rates		## Make sure that only the rates of edges attached to lead are changed
		self.gamma_00, self.gamma_01, self.gamma_02, self.gamma_e3	= rates
		self.initialize_box()
		return self.maj_box.calculate_blockade_cond(lead)

	def block_via_rates(self, lead=0):
		rates_zero	= np.array([1,1,1,1] )
		opt_func	= lambda x: self.tune_rates(x, lead)
		bnds		= ((0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf) )
		min_res		= minimize(opt_func, rates_zero, tol=self.dphi**2, options={'maxiter':10000}, bounds=bnds )
		return


	def initialize_box(self):
		if self.box_symmetry == 1:
			edgy0 	= ec.edge(self.phi0, [0], [self.gamma_00])
			edgy1 	= ec.edge(self.phi1, [0], [self.gamma_01])
			edgy2 	= ec.edge(self.phi2, [1], [self.gamma_e2])
			edgy3 	= ec.edge(self.phi3, [1], [self.gamma_e3])
		elif self.box_symmetry == 2:
			edgy0 	= ec.edge(self.phi0, [0], [self.gamma_00])
			edgy1 	= ec.edge(self.phi1, [0], [self.gamma_01])
			edgy2 	= ec.edge(self.phi2, [0], [self.gamma_02])
			edgy3 	= ec.edge(self.phi3, [1], [self.gamma_e3])
		elif self.box_symmetry == 3:
			edgy0 	= ec.edge(self.phi0, [0], [self.gamma_00])
			edgy1 	= ec.edge(self.phi1, [0, 1], [self.gamma_01, self.gamma_11])
			edgy2 	= ec.edge(self.phi2, [1], [self.gamma_12])
			edgy3 	= ec.edge(self.phi3, [2], [self.gamma_e3])
		
		edgy0.create_majorana(0)
		edgy1.create_majorana(1, overlaps={0: self.eps01})
		edgy2.create_majorana(2, overlaps={1: self.eps12})
		edgy3.create_majorana(3, overlaps={2: self.eps23})

		if self.model == 2:
			edgy0.create_majorana(4, wf_factor=self.factor0, wf_phase_angle=self.th0, overlaps={0: self.eps_abs_0})
			edgy1.create_majorana(5, wf_factor=self.factor1, wf_phase_angle=self.th1, overlaps={1: self.eps_abs_1})
			edgy2.create_majorana(6, wf_factor=self.factor2, wf_phase_angle=self.th2, overlaps={2: self.eps_abs_2})
			edgy3.create_majorana(7, wf_factor=self.factor3, wf_phase_angle=self.th3, overlaps={3: self.eps_abs_3})
		elif self.model == 3:
			edgy0.create_majorana(4, wf_factor=self.factor0, wf_phase_angle=self.th0, overlaps={0: self.eps_abs_0, 2:self.eps_abs_2})
			edgy1.create_majorana(5, wf_factor=self.factor1, wf_phase_angle=self.th1, overlaps={1: self.eps_abs_1, 3:self.eps_abs_3})
		
		edge_lst	= np.array([edgy0, edgy1, edgy2, edgy3], dtype='object')
		self.edges	= edge_lst
		maj_box		= bc.majorana_box.box_via_edges(edge_lst, self.Vg)
		self.assign_box(maj_box)
		return maj_box

	def assign_box(self, maj_box):
		self.maj_box	= maj_box
		self.box_assigned	= True

	def connect_box(self):
		if not self.box_assigned:
			print('No box to connect in the transport setup')
			return
		else:
			self.maj_box.diagonalize()
	
			self.Ea		= self.maj_box.elec_en
			self.par	= self.maj_box.par
			self.tunnel	= self.maj_box.constr_tunnel()

			self.box_connected	= True

	def copy(self):
		return copy(self )
