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

	eps	= 1e-5

	dphi	= 1e-6
	
	gamma 	= 1e+0
	gamma_u	= 1e+0
	t 	= np.sqrt(gamma/(2*np.pi))+0.j
	t_u	= np.sqrt(gamma_u/(2*np.pi))+0.j

	factors	= [1.00, 1, 0.00, 1]

	phases	= np.array([+1/2*np.pi-dphi, 0, +1/2*np.pi+dphi, 0] )
	phases	= np.exp(1j*phases )

	th	= [0.50, 0.50, 0.50, 0.50]
	th	= [0.20, 0.20, 0.20, 0.20]
	th	= [0.00, 0.00, 0.00, 0.00]
	th	= [0.10, 0.10, 0.10, 0.10]

	thetas	= np.array(th )*np.pi + np.array([1, 2, 3, 4] )*dphi

	theta_phases	= np.exp( 1j*thetas)

	tunnel_mult	= [0, 1, 1, 1]
	tunnel_mult	= [0.5, 0.6, 0.7, 0.8]
	tunnel_mult	= [0.5, 1.0, 1.0, 1]
	tunnel_mult	= [0.5, 0.5, 0.5, 0.5]
	tunnel_mult	= [1, 1, 1, 1]

	model	= 2

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

	lead	= 0

	pre_run	= False
	pre_run	= True
	
	if pre_run:
		factors_pre	= [0.00, 1, 1.00, 1]

		phases_pre	= np.array([+1/2*np.pi-dphi, 0, +1/2*np.pi+dphi, 0] )
		phases_pre	= np.exp(1j*phases_pre )
		tb1, tb2, tb3, tt4, tb11, tb21, tb31, tt41	= tunnel_coupl(t, t_u, phases_pre, factors_pre, theta_phases, tunnel_mult)
		maj_op, overlaps, par	= box_definition(model, tb1, tb2, tb3, tt4, tb11, tb21, tb31, tt41, eps12, eps23, eps34, eps)

		maj_box		= bc.majorana_box(maj_op, overlaps, Vg, 'asymmetric_box')
		maj_box.diagonalize()
		Ea		= maj_box.elec_en
		tunnel		= maj_box.constr_tunnel()

		sys	= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=itype)

		sys.solve(qdq=False, rotateq=False)
		rho0	= sys.phi0
		initial_cur	= sys.current

		time_evo_rho	= finite_time_evolution(sys )
		rho0		= time_evo_rho(rho0, 1e9 )
		initial_cur	= current(sys, lead=lead)(rho0)

		#print('Initial state of the system: ', rho0)
		print('Current at initialization:', initial_cur)
		print()

	tb1, tb2, tb3, tt4, tb11, tb21, tb31, tt41	= tunnel_coupl(t, t_u, phases, factors, theta_phases, tunnel_mult)
	maj_op, overlaps, par	= box_definition(model, tb1, tb2, tb3, tt4, tb11, tb21, tb31, tt41, eps12, eps23, eps34, eps)

	maj_box		= bc.majorana_box(maj_op, overlaps, Vg, 'asymmetric_box')
	maj_box.diagonalize()
	Ea		= maj_box.elec_en
	tunnel		= maj_box.constr_tunnel()

	sys	= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=itype)

	sys.solve(qdq=False, rotateq=False)

	print('Eigenenergies:', sys.Ea)
	#print('Density matrix:', sys.phi0 )
	print('Current:', sys.current )
	print()

	if not pre_run:
		if model == 1:
			rho0	= np.array([1, 1, 0, 0, 1, 0, 0, 0])
			rho0	= normed_occupations(rho0 )
		elif model == 2:
			rho0	= abs_block(model )

	current_fct	= current(sys, lead=lead)
	stationary_sol	= stationary_state_limit(sys, rho0)
	#print('Solution via kernel: ', stationary_sol)
	kernel_cur	= current_fct(stationary_sol)
	print('Current via kernel: ', kernel_cur)

	time_evo_rho	= finite_time_evolution(sys)
	finite_sol	= time_evo_rho(rho0, 1e9 )
	finite_cur	= current_fct(finite_sol)

	#print('Finite time solution via kernel: ', finite_sol)
	print('Finite time current left lead via kernel: ', finite_cur)

	fig,ax	= plt.subplots()
	T	= 6
	t	= np.linspace(0, T, 1000)
	finite_time_plot(ax, sys, rho0, t, lead=lead)
	transm_charge	= charge_transmission(sys, current_fct, time_evo_rho, rho0, tau=5)
	print('Charge transmitted through the left lead: ', transm_charge )
	plt.show()

	return

def charge_transmission(sys, current_fct, time_evo_rho, rho0, tau=np.inf):
	return quad(lambda x: current_fct(time_evo_rho(rho0, x) ), 0, tau)

def abs_block(model):
	num_occ		= 16
	dof		= 128
	rho0		= np.zeros(dof )
	rho0[0]		= 1
	return rho0

def model_spec_dof(rho):
	dof	= rho.size
	num_occ	= int(np.sqrt(2*dof) )
	return num_occ, dof

	if model == 1:
		return 4, 8
	if model == 2:
		return 16, 128

def finite_time_plot(ax, sys, rho0, t, lead=0):
	dt		= t[1]-t[0]
	time_evo_rho	= finite_time_evolution(sys)
	current_fct	= current(sys, lead=lead)
	
	finite_cur	= np.array([current_fct(time_evo_rho(rho0, time) ) for time in t])
	ax.plot(t, finite_cur)
	ax.set_xlabel(r'$t \, [1/\Gamma]$')
	ax.set_ylabel(r'$I_{trans} \, [e\Gamma]$')

	ax_twin		= ax.twinx()
	charge		= np.cumsum(finite_cur)*dt
	color		= 'r'
	ax_twin.plot(t, charge, c=color)
	if lead==0:
		lead_string	= 'left'
	else:
		lead_string	= 'right'

	ax.grid(True)
	ax_twin.set_ylabel('Charge transmitted through ' + lead_string + ' lead [e]', c=color)
	ax_twin.tick_params(axis='y', labelcolor=color)
	
def current(sys, lead=0):
	Tba	= sys.Tba[lead]
	num_occ, dof	= model_spec_dof(sys.phi0)
	ones	= np.ones((int(num_occ/2), int(num_occ/2) ) )
	I_matrix_plus	= np.block([[ones, get_I_matrix(sys, 1, lead=lead)*ones], [ones, ones]] )
	I_matrix_minus	= np.block([[ones, get_I_matrix(sys, -1, lead=lead)*ones], [ones, ones]] )
	TbaRight	= Tba*I_matrix_plus
	TbaLeft		= Tba*I_matrix_minus

	current_rho	= lambda rho: -2*2*np.pi*np.trace(np.imag(np.dot(TbaLeft, np.dot(map_vec_to_den_mat(sys, rho), TbaRight) ) ) )
	return current_rho

def get_I_matrix(sys, sign=1, lead=0):
	digamma	= princ_int(sys, lead=lead)
	x_L	= get_x_from_sys(sys, lead=lead)
	return sign*(-1j/2*fermi_func(sign*x_L) + digamma/(2*np.pi) )

def princ_int(sys, lead=0):
	T	= sys.tlst[lead]
	mu	= sys.mulst[lead]
	D	= sys.dband
	x_L	= get_x_from_sys(sys, lead)
	return np.real(digamma(0.5+1j*x_L/(2*np.pi) )) - np.log(D/(2*np.pi*T) )

def get_x_from_sys(sys, lead=0):
	energies		= sys.Ea
	num_occ			= energies.size
	par_occ			= int(num_occ/2)
	matrix_of_energydiffs	= np.zeros((par_occ, par_occ) )
	for indices, value in np.ndenumerate(matrix_of_energydiffs):
		matrix_of_energydiffs[indices]	= energies[indices[0] ] - energies[indices[1]+par_occ]
	x_L			= (-matrix_of_energydiffs - sys.mulst[lead])/sys.tlst[lead]		# Minus before energies because of indices cb for I compared to indices bc for Tba
	return x_L

def fermi_func(x):
	return 1/(1+np.exp(x) )

def map_vec_to_den_mat(sys, rho):
	num_occ, dof	= model_spec_dof(rho)
	ofd_dof		= dof - num_occ
	half_ofd_dof	= int(ofd_dof/2 )

	par_occ		= int(num_occ/2)
	zeros		= np.zeros((par_occ, par_occ), dtype=np.complex )
	even_par	= zeros.copy()
	odd_par		= zeros.copy()
	for indices, value in np.ndenumerate(even_par):
		even_par[indices]	= rho[sys.si.get_ind_dm0(indices[0], indices[1], 0) ]
		if indices[0] < indices[1]:
			even_par[indices]	+= 1j*rho[sys.si.get_ind_dm0(indices[0], indices[1], 0)+half_ofd_dof ]
		elif indices[0] > indices[1]:
			even_par[indices]	-= 1j*rho[sys.si.get_ind_dm0(indices[0], indices[1], 0)+half_ofd_dof ]

	for indices, value in np.ndenumerate(odd_par):
		odd_par[indices]	= rho[sys.si.get_ind_dm0(indices[0], indices[1], 1) ]
		if indices[0] < indices[1]:
			odd_par[indices]	+= 1j*rho[sys.si.get_ind_dm0(indices[0], indices[1], 1)+half_ofd_dof ]
		elif indices[0] > indices[1]:
			odd_par[indices]	-= 1j*rho[sys.si.get_ind_dm0(indices[0], indices[1], 1)+half_ofd_dof ]

	den_mat	= np.block([[even_par, zeros], [zeros, odd_par] ] )
			
	return den_mat

def stationary_state_limit(sys, rho0):
	kernel		= np.matrix(sys.kern )
	eigenval, U_l, U_r	= get_eigensystem_from_kernel(kernel)

	zero_ind	= np.argmin(np.abs(eigenval ) )
	smallest_time	= sorted(np.abs(eigenval) )[1]
	print(smallest_time )

	zero_mat	= np.dot(U_r[:,zero_ind], U_l.getH()[zero_ind] )
	
	lim_solution	= np.array(np.dot(zero_mat, rho0)).reshape(-1)


	return lim_solution

def finite_time_evolution(sys):
	kernel		= np.matrix(sys.kern )
	eigenval, U_l, U_r	= get_eigensystem_from_kernel(kernel)
	dimensions	= U_l[0].size
	time_evol_mats	= np.array([np.dot(U_r[:,index], U_l.getH()[index] ) for index in range(dimensions) ] )
	
	time_evol	= lambda rho0, t: normed_occupations(np.sum(np.array([np.exp(eigenval[index]*t)*np.dot(time_evol_mats[index], rho0) for index in range(dimensions)]), axis=0) )

	return time_evol

def normed_occupations(vector):
	num_occ, dof	= model_spec_dof(vector)
	return vector/np.sum(vector[:num_occ])

def get_eigensystem_from_kernel(kernel):
	eigensystem	= eig(kernel, right=True, left=True)

	eigenval	= eigensystem[0]
	U_l		= np.matrix(eigensystem[1] )
	U_r		= np.matrix(eigensystem[2] )

	inverse_norm	= np.diag(1/np.diag(np.dot(U_l.getH(), U_r) ) )
	#eigenval[np.argmin(np.abs(eigenval) ) ]	= 0
	
	U_r		= np.dot(U_r, inverse_norm)
	return eigenval, U_l, U_r

def box_definition(model, tb1, tb2, tb3, tt4, tb11, tb21, tb31, tt41, eps12, eps23, eps34, eps):
	if model == 1:
		maj_op, overlaps, par	= abox.majorana_leads(tb1, tb2, tb3, tt4, eps12, eps23, eps34)
	elif model == 2:
		maj_op, overlaps, par	= abox.abs_leads(tb1, tb11, tb2, tb21, tb3, tb31, tt4, tt41, eps)
	
	return maj_op, overlaps, par

def tunnel_coupl(t, t_u, phases, factors, theta_phases, tunnel_mult):
	tb1	= t*phases[0]*factors[0]
	tb2     = t*phases[1]*factors[1]
	tb3     = t*phases[2]*factors[2]
	tt4	= t_u*phases[3]*factors[3]

	tb11	= tb1*theta_phases[0]*tunnel_mult[0]
	tb21	= tb2*theta_phases[1]*tunnel_mult[1]
	tb31	= tb3*theta_phases[2]*tunnel_mult[2]
	tt41	= tt4*theta_phases[3]*tunnel_mult[3]

	return tb1, tb2, tb3, tt4, tb11, tb21, tb31, tt41

if __name__=='__main__':
	main()
