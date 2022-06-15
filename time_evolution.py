import numpy as np
import qmeq
import matplotlib.pyplot as plt
import os
import fock_class as fc
import fock_tunnel_mat as ftm
import fock_basis_rotation as fbr
import box_class as bc
import bias_scan as bias_sc
import tunnel_scan
import asym_box as abox

from time import perf_counter
import scipy.optimize as opt
from scipy.linalg import eig


def main():
	eps12 	= 1e-6
	eps23	= 0e-6
	eps34 	= 2e-6

	dphi	= 1e-6
	
	gamma 	= 1.0
	t 	= np.sqrt(gamma/(2*np.pi))+0.j

	factors	= [1.00, 1, 0.00, 1]

	phases	= np.array([+1/2*np.pi-dphi, 0, +1/2*np.pi, 0] )
	phases	= np.exp(1j*phases )

	tb1	= t*phases[0]*factors[0]
	tb2     = t*phases[1]*factors[1]
	tb3     = t*phases[2]*factors[2]
	tt4	= t*phases[3]*factors[3]

	T1	= 1e1
	T2 	= T1

	v_bias	= 2e2
	mu1	= v_bias/2
	mu2	= -v_bias/2

	dband	= 1e5
	Vg	= +0e1
	
	T_lst 	= { 0:T1 , 1:T1}
	mu_lst 	= { 0:mu1 , 1:mu2}
	method	= 'Redfield'
	method	= 'Pauli'
	method	= 'Lindblad'
	method	= '1vN'
	itype	= 1
	
	test_run	= True

	maj_op, overlaps, par	= abox.majorana_leads(tb1, tb2, tb3, tt4, eps12, eps23, eps34)

	maj_box		= bc.majorana_box(maj_op, overlaps, Vg, 'asymmetric_box')
	maj_box.diagonalize()
	Ea		= maj_box.elec_en
	tunnel		= maj_box.constr_tunnel()

	sys	= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=itype)

	sys.solve(qdq=False, rotateq=False)
	rho0		= np.array([1, 0, 0, 0, 0, 0, 0, 0])

	stationary_state_limit(sys, rho0)


def stationary_state_limit(sys, rho0):
	kernel		= np.matrix(sys.kern )
	eigenval, U_l, U_r	= get_eigensystem_from_kernel(kernel)

	zero_ind	= np.argmin(np.abs(eigenval ) )
	zero_mat	= np.dot(U_r[:,zero_ind], U_l.getH()[zero_ind] )
	lim_solution	= np.dot(zero_mat, rho0)

	print('Solution via kernel: ', lim_solution)

	print('Eigenenergies:', sys.Ea)
	print('Density matrix:', sys.phi0 )
	print('Current:', sys.current )
	return 

def finite_time_evolution(sys, rho0):
	kernel		= np.matrix(sys.kern )
	eigenval, U_l, U_r	= get_eigensystem_from_kernel(kernel)

	return 0

def get_eigensystem_from_kernel(kernel):
	eigensystem	= eig(kernel, right=True, left=True)

	eigenval	= eigensystem[0]
	print(eigenval)
	U_l		= np.matrix(eigensystem[1] )
	U_r		= np.matrix(eigensystem[2] )

	inverse_norm	= np.diag(1/np.diag(np.dot(U_l.getH(), U_r) ) )
	U_r		= np.dot(U_r, inverse_norm)
	return eigenval, U_l, U_r


if __name__=='__main__':
	main()
