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
from scipy.special import digamma
from scipy.integrate import quad


def main():
	np.set_printoptions(precision=6)
	eps12 	= 1e-4
	eps23	= 0e-4
	eps34 	= 2e-4

	dphi	= 1e-7
	
	gamma 	= 1.0
	t 	= np.sqrt(gamma/(2*np.pi))+0.j

	factors	= [1.00, 1, 0.00, 1]

	phases	= np.array([+1/2*np.pi-dphi, 0, +1/2*np.pi, 0] )
	phases	= np.exp(1j*phases )

	tb1	= t*phases[0]*factors[0]
	tb2     = t*phases[1]*factors[1]
	tb3     = t*phases[2]*factors[2]
	tt4	= t*phases[3]*factors[3]

	T1	= 1e2
	T2 	= T1

	v_bias	= 2e3
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

	print('Eigenenergies:', sys.Ea)
	print('Density matrix:', sys.phi0 )
	print('Current:', sys.current )
	print()

	rho0		= np.array([1, 1, 0, 0, 0, 0, 0, 0])/2

	stationary_sol	= stationary_state_limit(sys, rho0)
	kernel_cur	= current(sys, stationary_sol)
	print('Current via kernel: ', kernel_cur)

	time_evo_rho	= finite_time_evolution(sys, rho0)
	finite_sol	= time_evo_rho(100 )
	finite_cur	= current(sys, finite_sol)

	print('Finite time solution via kernel: ', finite_sol)
	print('Finite time current via kernel: ', finite_cur)

	fig,ax	= plt.subplots()
	T	= 6
	t	= np.linspace(0, T, 1000)
	finite_time_plot(ax, sys, rho0, t)
	transm_charge_left	= quad(lambda x: current(sys, time_evo_rho(x) ), 0, np.inf)
	print('Charge transmitted through the left lead: ', transm_charge_left )
	plt.show()

	return

def finite_time_plot(ax, sys, rho0, t):
	dt		= t[1]-t[0]
	time_evo_rho	= finite_time_evolution(sys, rho0)
	
	finite_cur	= np.array([current(sys, time_evo_rho(time) ) for time in t])
	ax.plot(t, finite_cur)
	ax.set_xlabel(r'$t \, [1/\Gamma]$')
	ax.set_ylabel(r'$I_{trans} \, [e\Gamma]$')

	ax_twin		= ax.twinx()
	charge		= np.cumsum(finite_cur)*dt
	color		= 'r'
	ax_twin.plot(t, charge, c=color)
	ax_twin.set_ylabel('Charge transmitted through left lead [e]', c=color)
	ax_twin.tick_params(axis='y', labelcolor=color)
	

def current(sys, rho):
	Tba	= sys.Tba[0]
	zeros	= np.zeros((2,2) )
	ones	= np.ones((2,2) )
	I_matrix_plus	= np.block([[ones, get_I_matrix(sys, 1)*ones], [ones, ones]] )
	I_matrix_minus	= np.block([[ones, get_I_matrix(sys, -1)*ones], [ones, ones]] )
	TbaRight	= Tba*I_matrix_plus
	TbaLeft		= Tba*I_matrix_minus

	cur	= -2*2*np.pi*np.trace(np.imag(np.dot(TbaLeft, np.dot(map_vec_to_den_mat(rho), TbaRight) ) ) )
	return cur

def get_I_matrix(sys, sign=1):
	digamma	= princ_int(sys)
	x_L	= get_x_from_sys(sys)
	return sign*(-1j/2*fermi_func(sign*x_L) + digamma/(2*np.pi) )

def princ_int(sys):
	T	= sys.tlst[0]
	mu	= sys.mulst[0]
	D	= sys.dband
	x_L	= get_x_from_sys(sys)
	return np.real(digamma(0.5+1j*x_L/(2*np.pi) )) - np.log(D/(2*np.pi*T) )

def get_x_from_sys(sys):
	matrix_of_energydiffs	= np.ones((2,2) )
	energies		= sys.Ea
	for indices, value in np.ndenumerate(matrix_of_energydiffs):
		matrix_of_energydiffs[indices]	= energies[indices[0] ] - energies[indices[1]+2]
	x_L			= (-matrix_of_energydiffs - sys.mulst[0])/sys.tlst[0]		# Minus before energies because of indices cb for I compared to indices bc for Tba
	return x_L

def fermi_func(x):
	return 1/(1+np.exp(x) )

def map_vec_to_den_mat(rho):
	dim		= rho.size
	half_dim	= int(dim/2)
	den_mat		= np.zeros((half_dim, half_dim), dtype=np.complex )

	den_mat		+= np.diag(rho[:half_dim] )
	den_mat[0,1]	+= rho[half_dim] + 1j*rho[half_dim+2]
	den_mat[1,0]	+= np.conjugate(den_mat[0,1] )

	den_mat[2,3]	+= rho[half_dim+1] + 1j*rho[half_dim+3]
	den_mat[3,2]	+= np.conjugate(den_mat[2,3] )

	return den_mat

def stationary_state_limit(sys, rho0):
	kernel		= np.matrix(sys.kern )
	eigenval, U_l, U_r	= get_eigensystem_from_kernel(kernel)

	zero_ind	= np.argmin(np.abs(eigenval ) )
	zero_mat	= np.dot(U_r[:,zero_ind], U_l.getH()[zero_ind] )
	lim_solution	= np.array(np.dot(zero_mat, rho0)).reshape(-1)

	print('Solution via kernel: ', lim_solution)

	return lim_solution

def finite_time_evolution(sys, rho0):
	kernel		= np.matrix(sys.kern )
	eigenval, U_l, U_r	= get_eigensystem_from_kernel(kernel)
	dimensions	= U_l[0].size
	time_evol_mats	= np.array([np.dot(U_r[:,index], U_l.getH()[index] ) for index in range(dimensions) ] )
	
	time_evol	= lambda t: normed_occupations(np.sum(np.array([np.exp(eigenval[index]*t)*np.dot(time_evol_mats[index], rho0) for index in range(dimensions)]), axis=0) )

	return time_evol

def normed_occupations(vector):
	half_length	= int(vector.size/2)
	return vector/np.sum(vector[:half_length])

def get_eigensystem_from_kernel(kernel):
	eigensystem	= eig(kernel, right=True, left=True)

	eigenval	= eigensystem[0]
	U_l		= np.matrix(eigensystem[1] )
	U_r		= np.matrix(eigensystem[2] )

	inverse_norm	= np.diag(1/np.diag(np.dot(U_l.getH(), U_r) ) )
	#eigenval[np.argmin(np.abs(eigenval) ) ]	= 0
	
	U_r		= np.dot(U_r, inverse_norm)
	return eigenval, U_l, U_r


if __name__=='__main__':
	main()
