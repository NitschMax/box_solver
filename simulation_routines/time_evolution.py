import cyclic_blockade as cb
import bias_scan as bias_sc
import tunnel_scan
import asym_box as abox

import fock_class as fc
import fock_tunnel_mat as ftm
import fock_basis_rotation as fbr
import box_class as bc

import numpy as np
import matplotlib.pyplot as plt
import os

import scipy.optimize as opt
from scipy.linalg import eig
from scipy.special import digamma
from scipy.integrate import quad

import qmeq


def main():
	np.set_printoptions(precision=6)
	eps12 	= 1e-3
	eps23	= 0e-3
	eps34 	= 2e-3

	eps	= 1e-3

	dphi	= 1e-6
	
	gamma 	= 1e+0
	gamma_u	= 1e+0
	t 	= np.sqrt(gamma/(2*np.pi))+0.j
	t_u	= np.sqrt(gamma_u/(2*np.pi))+0.j

	factors	= [1.00, 1, 0.00, 1]

	phases	= np.array([+1/2*np.pi-dphi, 0, +1/2*np.pi+dphi, 0] )
	phases	= np.exp(1j*phases )

	th	= [0.50, 0.50, 0.50, 0.50]
	th	= [0.00, 0.00, 0.00, 0.00]
	th	= [0.00, 0.00, 0.00, 0.00]
	th	= [0.300, 0.00, 0.00, 0.00]

	thetas	= np.array(th )*np.pi + np.array([1, 2, 3, 4] )*dphi

	theta_phases	= np.exp( 1j*thetas)

	tunnel_mult	= [0.5, 0.6, 0.7, 0.8]
	tunnel_mult	= [0.5, 1.0, 1.0, 1]
	tunnel_mult	= [0, 0, 0, 0]
	tunnel_mult	= [1, 1, 1, 1]

	model	= 1

	T1	= 1e2
	T2 	= T1

	v_bias	= 2e3
	mu1	= v_bias/2
	mu2	= -v_bias/2

	dband	= 1e5
	Vg	= +0e1
	
	T_lst 	= { 0:T1 , 1:T1}
	mu_lst 	= { 0:mu1 , 1:mu2}
	method	= 'pyRedfield'
	method	= 'pyPauli'
	method	= 'pyLindblad'
	method	= 'py1vN'
	itype	= 1

	lead	= 1

	pre_run	= False
	pre_run	= True
	
	if pre_run:
		guess_x	= np.array([np.pi/2, np.pi/2, 0, 1] )
		phases_pre, factors_pre	= cb.find_blockade(model, t, t_u, theta_phases, tunnel_mult, dphi, guess_x )

		tb1, tb2, tb3, tt4, tb11, tb21, tb31, tt41	= tunnel_coupl(t, t_u, phases_pre, factors_pre, theta_phases, tunnel_mult)
		maj_op, overlaps, par	= box_definition(model, tb1, tb2, tb3, tt4, tb11, tb21, tb31, tt41, eps12, eps23, eps34, eps)

		maj_box		= bc.majorana_box(maj_op, overlaps, Vg, 'asymmetric_box')
		maj_box.diagonalize()
		Ea		= maj_box.elec_en
		tunnel		= maj_box.constr_tunnel()

		sys	= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=itype, countingleads=[lead])

		sys.solve(qdq=False, rotateq=False)
		rho0	= sys.phi0
		initial_cur	= sys.current
		maj_box.print_eigenstates()
		print('Initial state of the system: ', sys.phi0)
		print('Current at initialization:', initial_cur)
		print()

	guess_z	= np.array([np.pi/2, np.pi/2, 1, 0] )
	phases, factors	= cb.find_blockade(model, t, t_u, theta_phases, tunnel_mult, dphi, guess_z )

	tb1, tb2, tb3, tt4, tb11, tb21, tb31, tt41	= tunnel_coupl(t, t_u, phases, factors, theta_phases, tunnel_mult)
	maj_op, overlaps, par	= box_definition(model, tb1, tb2, tb3, tt4, tb11, tb21, tb31, tt41, eps12, eps23, eps34, eps)

	maj_box		= bc.majorana_box(maj_op, overlaps, Vg, 'asymmetric_box')
	maj_box.diagonalize()
	Ea		= maj_box.elec_en
	tunnel		= maj_box.constr_tunnel()

	sys	= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=itype, countingleads=[lead])

	sys.solve(qdq=False, rotateq=False)
	if not pre_run:
		phi0	= sys.phi0

	print('Eigenenergies:', sys.Ea)
	#print('Density matrix:', sys.phi0 )
	#print('Current:', sys.current )
	print()


	current_fct	= current(sys, lead=lead, i_n=True)
	#stationary_sol	= stationary_state_limit(sys, rho0)
	#print('Solution via kernel: ', stationary_sol)
	#kernel_cur	= current_fct(stationary_sol)
	#print('Current via kernel: ', kernel_cur)

	time_evo_rho	= finite_time_evolution(sys)
	finite_sol	= time_evo_rho(rho0, 0e9 )
	finite_cur	= current_fct(finite_sol)
	print('Current:', finite_cur )

	#print('Finite time solution via kernel: ', finite_sol)
	print('Finite time current left lead via kernel: ', finite_cur)

	fig,ax	= plt.subplots()

	logx	= False
	logy	= False
	qs_desc	= False
	i_n	= True
	T	= 9
	t	= 10**np.linspace(0, T, 1000)
	t	= np.linspace(0e1, 1e1, 1000)
	finite_time_plot(ax, sys, rho0, t, lead=lead, logx=logx, logy=logy, plot_charge=False, i_n=i_n, qs_desc=qs_desc )
	#transm_charge	= charge_transmission(current_fct, time_evo_rho, rho0, tau=5)
	#print('Charge transmitted through the left lead: ', transm_charge )
	plt.show()
	return

	energies	= np.linspace(0, 0.1, 20)
	eigenvalue_plot(ax, model, lead, Vg, dband, mu_lst, T_lst, method, itype, tb1, tb2, tb3, tt4, tb11, tb21, tb31, tt41, eps12, eps23, eps34, energies)
	plt.show()
	return

def eigenvalue_plot(ax, model, lead, Vg, dband, mu_lst, T_lst, method, itype, tb1, tb2, tb3, tt4, tb11, tb21, tb31, tt41, eps12, eps23, eps34, energies):
	eigenvalues	= []
	for eps in energies:
		maj_op, overlaps, par	= box_definition(model, tb1, tb2, tb3, tt4, tb11, tb21, tb31, tt41, eps12, eps23, eps34, eps)

		maj_box		= bc.majorana_box(maj_op, overlaps, Vg, 'asymmetric_box')
		maj_box.diagonalize()
		Ea		= maj_box.elec_en
		tunnel		= maj_box.constr_tunnel()

		sys	= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=itype, countingleads=[lead])

		sys.solve(qdq=False, rotateq=False)
		eigenvalues.append(quasi_stationary_states(sys, real=True, imag=True) )
	ax.set_xlabel(r'ABS overlap [$\Gamma$]')
	ax.set_ylabel('Eigenvalues')
	ax.grid(True)
	eigenvalues	= np.array(eigenvalues)
	labels	= np.array(range(eigenvalues[0,0].size) )[::-1]
	ax_twin		= ax.twinx()
	ax.plot(energies, eigenvalues[:,0].real )
	ax_twin.plot(energies, eigenvalues[:,0].imag )
	ax.legend(labels)

def filter_smallest_eigen(eigenval, real=True, imag=False, order=True):
	rates		= np.column_stack( (eigenval, range(eigenval.size) ) )
	rates		= rates[eigenval.real > -0.1]
	if order:
		rates		= rates[np.argsort(rates[:,0].real ) ]
	result		= rates[:,0]
	indices		= rates[:,1]
	if real == True:
		result	= np.real(rates[:,0] ).astype(np.complex )
	if imag == True:
		result	+= 1j*np.imag(rates[:,0] )
	return result, indices

def filter_largest_eigen(eigenval, real=True, imag=False, order=True):
	rates		= np.column_stack( (eigenval, range(eigenval.size) ) )
	rates		= rates[eigenval.real < -0.1]
	if order:
		rates		= rates[np.argsort(rates[:,0].real ) ]
	result		= rates[:,0]
	indices		= rates[:,1]
	if real == True:
		result	= np.real(rates[:,0] ).astype(np.complex )
	if imag == True:
		result	+= 1j*np.imag(rates[:,0] )
	return result, indices

def quasi_stationary_states(sys, real=True, imag=False):
	kernel		= np.matrix(sys.kern )
	eigenval, U_l, U_r	= get_eigensystem_from_kernel(kernel)

	zero_ind	= np.argmin(np.abs(eigenval ) )
	smallest_rate	= filter_smallest_eigen(eigenval, real=real, imag=imag)

	return smallest_rate

def charge_transmission(current_fct, time_evo_rho, rho0, tzero=0, tau=np.inf):
	return quad(lambda x: current_fct(time_evo_rho(rho0, x) ), tzero, tau)

def abs_block():
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
	if model == 3:
		return 8, 32

def finite_time_plot(ax, sys, rho0, t, lead=0, logx=False, logy=False, plot_charge=True, i_n=True, qs_desc=False):
	dt		= t[1]-t[0]
	time_evo_rho	= finite_time_evolution(sys, qs_desc=qs_desc)
	current_fct	= current(sys, lead=lead, i_n=i_n)

	finite_cur	= np.array([current_fct(time_evo_rho(rho0, time) ) for time in t])
	
	if i_n:
		labels		= ['Current', 'Noise']
	else:
		labels		= ['Current']
	if np.less(finite_cur[:,0], 0).all() and logy:
		ax.plot(t, -finite_cur[:,0])
		labels		= ['Negative Current', 'Noise']
	else:
		ax.plot(t, finite_cur[:,0])
	ax.plot(t, finite_cur[:,1])
	ax.set_xlabel(r'$t \, [1/\Gamma]$')
	ax.set_ylabel(r'$I_{trans} \, [e\Gamma]$')

	if lead==0:
		lead_string	= 'left'
	else:
		lead_string	= 'right'

	if plot_charge:
		ax_twin		= ax.twinx()
		charge		= np.cumsum(finite_cur)*dt
		color		= 'r'
		ax_twin.plot(t, charge, c=color)
		ax_twin.set_ylabel('Charge transmitted through ' + lead_string + ' lead [e]', c=color)
		ax_twin.tick_params(axis='y', labelcolor=color)

	if logx:
		ax.set_xscale('log')
	if logy:
		ax.set_yscale('log')

	ax.grid(True)
	ax.legend(labels=labels)
	return
	
def partial_current(sys, lead=0, i_n=False):
	Tba	= sys.Tba[lead]
	num_occ, dof	= model_spec_dof(sys.phi0)
	ones	= np.ones((int(num_occ/2), int(num_occ/2) ) )
	I_matrix_plus	= np.block([[ones, get_I_matrix(sys, 1, lead=lead)*ones], [ones, ones]] )
	I_matrix_minus	= np.block([[ones, get_I_matrix(sys, -1, lead=lead)*ones], [ones, ones]] )
	TbaRight	= Tba*I_matrix_plus
	TbaLeft		= Tba*I_matrix_minus

	eigenval, U_l, U_r	= get_eigensystem_from_kernel(sys.kern)
	time_evol_mats		= np.array([np.dot(U_r[:,index], U_l.getH()[index] ) for index in range(U_l[0].size) ] )

	current_p	= lambda ind: -2*2*np.pi*np.trace(np.imag(np.dot(TbaLeft, np.dot(map_vec_to_den_mat(sys, U_r[:,ind]), TbaRight) ) ) )
	return current_p

def current_self_implemented(sys, lead=0, i_n=False):
	Tba	= sys.Tba[lead]
	num_occ, dof	= model_spec_dof(sys.phi0)
	ones	= np.ones((int(num_occ/2), int(num_occ/2) ) )
	I_matrix_plus	= np.block([[ones, get_I_matrix(sys, 1, lead=lead)*ones], [ones, ones]] )
	I_matrix_minus	= np.block([[ones, get_I_matrix(sys, -1, lead=lead)*ones], [ones, ones]] )
	TbaRight	= Tba*I_matrix_plus
	TbaLeft		= Tba*I_matrix_minus
	current_rho	= lambda rho: -2*2*np.pi*np.trace(np.imag(np.dot(TbaLeft, np.dot(map_vec_to_den_mat(sys, rho), TbaRight) ) ) )
	return current_rho

def current(sys, lead=0, i_n=False):
	current_rho	= lambda rho: current_via_sys(sys, rho, lead, i_n=i_n)
	return current_rho

def current_via_sys(sys, rho, lead, i_n=False):
	sys.appr.kernel_handler.set_phi0(rho)
	sys.phi0	= rho
	sys.current.fill(0.0)

	if i_n:
		sys.current_noise.fill(0.0)
		sys.appr.generate_current_noise()
		return np.array(sys.current_noise )
	
	sys.appr.generate_current()
	return sys.current[lead]

def get_I_matrix(sys, sign=1, lead=0):
	digamma	= princ_int(sys, lead=lead)
	x_L	= get_x_from_sys(sys, lead=lead)
	return sign*(-1j/2*fermi_func(sign*x_L) + digamma/(2*np.pi) )

def princ_int(sys, lead=0, include_eigenenergies=1):
	T	= sys.tlst[lead]
	mu	= sys.mulst[lead]
	D	= sys.dband
	x_L	= get_x_from_sys(sys, lead, include_eigenenergies=include_eigenenergies)
	return np.real(digamma(0.5+1j*x_L/(2*np.pi) )) - np.log(D/(2*np.pi*T) )

def get_x_from_sys(sys, lead=0, include_eigenenergies=1):
	energies		= sys.Ea
	num_occ			= energies.size
	par_occ			= int(num_occ/2)
	matrix_of_energydiffs	= np.zeros((par_occ, par_occ) )
	for indices, value in np.ndenumerate(matrix_of_energydiffs):
		matrix_of_energydiffs[indices]	= energies[indices[0] ] - energies[indices[1]+par_occ]
	matrix_of_energydiffs	*= include_eigenenergies
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
	smallest_rate, indices	= filter_smallest_eigen(eigenval, real=True, imag=False)
	print('Slowest decay rates', smallest_rate )

	zero_mat	= np.dot(U_r[:,zero_ind], U_l.getH()[zero_ind] )
	
	lim_solution	= np.array(np.dot(zero_mat, rho0)).reshape(-1)

	return lim_solution

def finite_time_evolution(sys, qs_desc=False):
	kernel		= np.matrix(sys.kern )
	eigenval, U_l, U_r	= get_eigensystem_from_kernel(kernel)
	dimensions	= U_l[0].size
	smallest_rate, small_indices	= filter_smallest_eigen(eigenval, real=True, imag=False)
	time_evol_mats	= np.array([np.dot(U_r[:,index], U_l.getH()[index] ) for index in range(dimensions) ] )

	indices	= range(dimensions)
	if qs_desc:
		indices	= np.abs(small_indices).astype(np.int)

	time_evol_mats	= time_evol_mats[indices]
	eigenval	= eigenval[indices]
	
	#time_evol	= lambda rho0, t: normed_occupations(np.sum(np.array([np.exp(eigenval[index]*t)*np.dot(time_evol_mats[index], rho0) for index in indices]), axis=0) )
	time_evol	= lambda rho0, t: normed_occupations(np.matmul(np.transpose(np.tensordot(time_evol_mats, rho0, axes=1) ), np.exp(eigenval*t) ) )

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
	elif model == 3:
		maj_op, overlaps, par	= abox.six_maj(tb1, tb2, tb3, tt4, eps12, eps23, eps34, eps, tb11=tb11, tb21=tb21)
	
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