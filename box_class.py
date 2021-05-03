import numpy as np
import fock_class as fc
import fock_basis_rotation as fbr
import fock_tunnel_mat as ftm

class majorana_box:
	def __init__(self, majoranas, overlaps, Vg=0):
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
		self.elec_en	= np.zeros(2**self.elec_num)
		self.adj_charging(self.Vg)
		
	def diagonalize(self):
		self.energies, self.U	= fbr.rotated_system(self.elec_num, self.overlaps)
		self.adj_charging(self.Vg)
		self.diagonal	= True

	def adj_charging(self, Vg):
		half_number		= int(2**self.elec_num/2 )
		self.Vg			= Vg
		self.elec_en[half_number:]	= self.energies[half_number] - Vg
		return self.elec_en

	def constr_tunnel(self):
		tunnel		= ftm.constr_tunnel_mat(self.elec_num, self.majoranas)
		if self.diagonal:
			self.tunnel	= np.array([np.dot(self.U.getH(), np.dot(t, self.U) ) for t in tunnel] )
		else:
			print('System not yet diagonalized. Tunnelmatrix in default basis returned!')
		return self.tunnel
		
	def change(self, majoranas=[], overlaps=[]):
		if len(overlaps) > 0:
			self.overlaps	= overlaps
			self.U		= np.matrix(np.zeros((self.elec_num, self.elec_num) ) )
			self.diagonal	= False

		if len(majoranas) > 0:
			self.majoranas	= majoranas
		
