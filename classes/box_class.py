import numpy as np
import fock_class as fc
import fock_basis_rotation as fbr
import fock_tunnel_mat as ftm

class majorana_box:
	def __init__(self, majoranas, overlaps, Vg=0, name='Unidentified'):
		self.majoranas	= majoranas
		self.overlaps	= overlaps
		self.Vg		= Vg
		self.tunnel	= 0

		self.maj_num	= len(majoranas)
		if np.mod(self.maj_num, 2) != 0:
			print('Number of Majoranas not allowed!')
		self.elec_num	= int(self.maj_num/2 )

		self.diagonal	= False
		self.U		= np.matrix(np.zeros((self.elec_num, self.elec_num) ) )
		self.energies	= np.zeros(2**self.elec_num)
		self.par	= np.concatenate((np.zeros(2**(self.elec_num-1) ), np.ones(2**(self.elec_num-1) ) ) ).astype(int)
		self.elec_en	= np.zeros(2**self.elec_num)
		self.name	= name
		self.adj_charging(self.Vg)
		self.states	= fc.set_of_fock_states(self.elec_num)
		self.blockade_cond	= 1
		
	def calculate_blockade_cond(self, lead=0):
		self.blockade_cond	= 0
		for majorana in self.majoranas:
			for idx, connection in np.ndenumerate(majorana.lead):
				if connection == lead:
					self.blockade_cond	+= majorana.coupling[idx]**2
		self.blockade_cond	= np.abs(self.blockade_cond)
		return self.blockade_cond
	
	def diagonalize(self):
		self.energies, self.U	= fbr.rotated_system(self.elec_num, self.overlaps)
		self.adj_charging(self.Vg)
		self.diagonal	= True

	@classmethod
	def box_via_edges(cls, list_of_edges, Vg=0):
		majoranas	= np.concatenate([np.array(edge.majoranas) for edge in list_of_edges] )
		number_majoranas	= np.max([maj.index for maj in majoranas] )+1

		if number_majoranas != majoranas.size:
			print('Indices of Majoranas invalid! Can not build a box with that input.')
			return

		energy_overlaps	= np.zeros( (number_majoranas, number_majoranas) )
		for maj in majoranas:
			for partner_ind in maj.overlaps:
				energy_overlaps[partner_ind, maj.index]	+= maj.overlaps[partner_ind]
		return majorana_box(majoranas, energy_overlaps, Vg)

	def adj_charging(self, Vg):
		half_number		= int(2**self.elec_num/2 )
		self.Vg			= Vg
		self.elec_en			= self.energies.copy()
		self.elec_en[half_number:]	= self.energies[half_number:] - Vg
		return self.elec_en

	def print_eigenstates(self):
		if self.diagonal:
			for k, col in enumerate(self.U.transpose() ):
				column	= np.ravel(col )
				superpos	= str(k) + '. '

				for l, elem in np.ndenumerate(column):
					if np.abs(elem) > 0.01:
						superpos	+= str(self.states.get_state(l, fac=np.round(elem, 3) ) ) +  ' + '
				superpos	= superpos[:-2]
				print(superpos)
		else:
			print('Unable to plot eigenbasis. Diagonalize system first')
		return

	def constr_tunnel(self):
		tunnel		= ftm.constr_tunnel_mat(self.elec_num, self.majoranas)
		if self.diagonal:
			self.tunnel	= np.array([np.dot(self.U.getH(), np.dot(t, self.U) ) for t in tunnel] )
		else:
			print('System not yet diagonalized. Tunnelmatrix in default basis returned!')
			self.tunnel	= tunnel
		return self.tunnel
		
	def change(self, majoranas=[], overlaps=[]):
		if len(overlaps) > 0:
			self.overlaps	= overlaps
			self.U		= np.matrix(np.zeros((self.elec_num, self.elec_num) ) )
			self.diagonal	= False

		if len(majoranas) > 0:
			self.majoranas	= majoranas
		
