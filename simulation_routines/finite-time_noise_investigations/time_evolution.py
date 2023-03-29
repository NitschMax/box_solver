import cyclic_blockade as cb
import setup as set

import fock_class as fc
import fock_tunnel_mat as ftm
import fock_basis_rotation as fbr
import box_class as bc

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

import matplotlib.ticker as ticker
import os

import scipy.optimize as opt
from scipy.linalg import eig
from scipy.special import digamma
from scipy.integrate import quad,nquad

import qmeq


def main():
	np.set_printoptions(precision=6)
	pre_run	= False
	pre_run	= True
	
	if pre_run:
		t_set_x	= set.create_transport_setup()
		t_set_x.initialize_leads()
		t_set_x.initialize_box()
	
		t_set_x.adjust_to_x_blockade()
		t_set_x.connect_box()
	
		sys_x	= t_set_x.build_qmeq_sys()

		sys_x.solve(qdq=False, rotateq=False)
		rho0		= sys_x.phi0
		initial_cur	= sys_x.current
		t_set_x.maj_box.print_eigenstates()
		print('Initial state of the system: ', sys_x.phi0)
		print('Current at initialization:', initial_cur)
		print()

		t_set_z	= t_set_x.copy()

		t_set_z.adjust_to_z_blockade()

		t_set_z.initialize_leads()
		t_set_z.initialize_box()
		t_set_z.connect_box()

		sys_z	= t_set_z.build_qmeq_sys()

		sys_z.solve(qdq=False, rotateq=False)
		rho0_2	= sys_z.phi0

		lead	= [0,1]
		i_n	= t_set_z.i_n

		current_fct_z	= current(sys_z, i_n=i_n)
		#stationary_sol	= stationary_state_limit(sys_z, rho0)
		#print('Solution via kernel: ', stationary_sol)
		#kernel_cur	= current_fct(stationary_sol)
		#print('Current via kernel: ', kernel_cur)

		time_evo_rho	= finite_time_evolution(sys_z)
		finite_sol	= time_evo_rho(rho0, 0e9 )
		finite_cur	= current_fct_z(finite_sol)
		print('Current:', finite_cur )

		#print('Finite time solution via kernel: ', finite_sol)
		print('Finite time current left lead via kernel: ', finite_cur)


	t_set	= t_set_z

	t_set.initialize_leads()
	t_set.initialize_box()
	t_set.connect_box()
	t_set.maj_box.print_eigenstates()

	sys	= t_set.build_qmeq_sys()
	sys.solve(qdq=False, rotateq=False)
	fig, ax		= plt.subplots(1,1)

	logx	= True
	logy	= False
	qs_desc	= False

	T	= 4
	points	= 1000
	t	= np.linspace(0e1, 10**T, points )
	t	= 10**np.linspace(-4.5, 4.2, points )
	lead	= [0]
	finite_time_plot(ax, sys_z, rho0, t, lead=lead, logx=logx, logy=logy, plot_charge=True, i_n=i_n, qs_desc=qs_desc, sys_2=sys_x )
	#transm_charge	= charge_transmission(current_fct, time_evo_rho, rho0, tau=5)
	#print('Charge transmitted through the left lead: ', transm_charge )
	plt.tight_layout()
	plt.show()
	return

	print(sys.phi0 )
	print(sys.current)
	check_validity_of_EV(sys)
	return

	x	= 10**np.linspace(-6, 0, 16)
	print(x)
	X, Y	= np.meshgrid(x, x)
	the_code_does_work_plot(fig, ax, X, Y, t_set)
	plt.show()
	return

	fig, axes	= plt.subplots(2,2)
	tunnel_matrix_plot(axes, t_set)
	plt.show()
	return

	energies	= np.linspace(0, 0.1, 20)
	eigenvalue_plot(ax, energies, t_set, number=1, real=True, imag=False )
	plt.show()
	return

	
def check_validity_of_EV(sys):
	largest_ev	= quasi_stationary_states(sys, real=True, imag=True, number=1)[0].real
	print(largest_ev)
	if np.abs(largest_ev) > 1e-14:
		print('Positive eigenvalue detected! Check spectrum of Liovillion.')
	else:
		print('Eigenvalue spectrum of Liovillion appears physicsal.')
	return

def the_code_does_work_plot(fig, ax, X, Y, t_set):
	st_state_eigen	= np.zeros(X.shape)

	for idx, dummy in np.ndenumerate(X):
		t_set.eps_abs_0	= +1.0*X[idx]
		t_set.eps_abs_1	= +0.0*X[idx]
		t_set.eps_abs_2	= +1.0*Y[idx]
		t_set.eps_abs_3	= +0.0*Y[idx]
		t_set.initialize_leads()
		t_set.initialize_box()
		t_set.connect_box()

		sys	= t_set.build_qmeq_sys()
		sys.solve(qdq=False, rotateq=False)
		st_state_eigen[idx]	= quasi_stationary_states(sys, real=True, imag=True, number=1)[0].real

	c	= ax.contourf(X, Y, np.abs(st_state_eigen ), locator=ticker.LogLocator() )
	cbar	= fig.colorbar(c, ax=ax)

	fs	= 16
	ax.tick_params(labelsize=fs)
	ax.set_xlabel(r'$\epsilon_0$', size=fs)
	ax.set_ylabel(r'$\epsilon_2$', size=fs)
	ax.set_xscale('log')
	ax.set_yscale('log')
	#ax.locator_params(axis='both', nbins=5)

	cbar.ax.tick_params(labelsize=fs)
	cbar.ax.set_title(label='largest EV', size=fs)
	cbar.ax.locator_params(axis='y', nbins=5)

	
def tunnel_matrix_plot(axes, t_set):
	t_set.initialize_leads()
	t_set.initialize_box()
	t_set.connect_box()

	sys	= t_set.build_qmeq_sys()
	sys.solve(qdq=False, rotateq=False)

	axes[0,0].imshow(sys.Tba[0].real )
	axes[0,1].imshow(sys.Tba[1].real )
	axes[1,0].imshow(sys.Tba[0].imag )
	axes[1,1].imshow(sys.Tba[1].imag )
	print('Largest eigenvalue of the Liouvillion', quasi_stationary_states(sys, real=True, imag=True, number=1)[0].real )

def eigenvalue_plot(ax, energies, t_set, number=0, real=True, imag=True):
	eigenvalues	= []
	for eps in energies:
		t_set.eps_abs_0	= 1.0*eps
		t_set.eps_abs_1	= 0.0*eps
		t_set.eps_abs_2	= 0.0*eps
		t_set.eps_abs_3	= 0.0*eps

		t_set.initialize_leads()
		t_set.initialize_box()
		t_set.connect_box()

		sys	= t_set.build_qmeq_sys()
		sys.solve(qdq=False, rotateq=False)

		eigenvalues.append(quasi_stationary_states(sys, real=real, imag=imag, number=number) )

	ax.set_xlabel(r'ABS overlap [$\Gamma$]')
	ax.set_ylabel('Eigenvalues')
	ax.grid(True)
	eigenvalues	= np.array(eigenvalues)
	labels	= np.array(range(eigenvalues[0,0].size) )[::-1]
	ax.plot(energies, eigenvalues[:,0].real )
	ax.legend(labels)
	if imag:
		ax_twin		= ax.twinx()
		ax_twin.plot(energies, eigenvalues[:,0].imag )

def filter_smallest_eigen(eigenval, real=True, imag=False, order=True, number=0):
	rates		= np.column_stack( (eigenval, range(eigenval.size) ) )
	rates		= rates[eigenval.real > -0.01]
	if order:
		rates		= rates[np.argsort(rates[:,0].real )[::-1] ]
	if number != 0:
		rates	= rates[:number]

	result		= rates[:,0]
	indices		= rates[:,1]
	if real == True:
		result	= np.real(rates[:,0] ).astype(complex )
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
		result	= np.real(rates[:,0] ).astype(complex )
	if imag == True:
		result	+= 1j*np.imag(rates[:,0] )
	return result, indices

def quasi_stationary_states(sys, real=True, imag=False, number=0):
	kernel		= np.matrix(sys.kern )
	eigenval, U_l, U_r	= get_eigensystem_from_kernel(kernel)

	zero_ind	= np.argmin(np.abs(eigenval ) )
	smallest_rate	= filter_smallest_eigen(eigenval, real=real, imag=imag, number=number)

	return smallest_rate

def charge_transmission(current_fct, time_evo_rho, rho0, tzero=0, tau=np.inf, epsrel=1.5e-8, use_nquad=False):
	if not use_nquad:
		result	= quad(lambda x: current_fct(time_evo_rho(rho0, x) )[0], tzero, tau, epsrel=epsrel)
	else:
		opts = {'epsrel': epsrel}
		result	= nquad(lambda x: current_fct(time_evo_rho(rho0, x) )[0], [(tzero, tau)], opts=opts )
	return result

def model_spec_dof(rho):
	dof	= rho.size
	num_occ	= int(np.sqrt(2*dof) )
	return num_occ, dof

def model_spec_dof_mat(rhoMat):
	num_occ	= rhoMat.shape[0]
	dof	= int(num_occ**2/2)	
	return num_occ, dof


def finite_time_plot(ax, sys, rho0, t, lead=[0], logx=False, logy=False, plot_charge=True, i_n=True, qs_desc=False, sys_2=None):
	dt		= t[1]-t[0]
	time_evo_rho	= finite_time_evolution(sys, qs_desc=qs_desc)
	current_fct	= current(sys, i_n=i_n)

	include_sys_2	= sys_2 is not None

	if include_sys_2:
		rho0_2	= rho0.copy()
		time_evo_rho_2	= finite_time_evolution(sys_2, qs_desc=qs_desc)
		current_fct_2	= current(sys_2, i_n=i_n)

	if include_sys_2:
		finite_cur	= []
		charge		= []
		for time in t:
			integration_cut	= np.minimum(time, 10)
			for k in range(8):
				rho0	= time_evo_rho_2(rho0_2, time)
				rho0_2	= time_evo_rho(rho0, time)
			if plot_charge:
				#taus	= 10**np.linspace(-4, np.log10(time), 100)
				#integrand	= np.array([current_fct(time_evo_rho(rho0, tau) )[0] for tau in taus ])
				#integrand	+= np.array([current_fct_2(time_evo_rho_2(rho0_2, tau) )[0] for tau in taus ])
				charge_flow	= charge_transmission(current_fct, time_evo_rho, rho0, tzero=0, tau=integration_cut, epsrel=1.5e-8, use_nquad=False)[0]
				#charge_flow	+= charge_transmission(current_fct_2, time_evo_rho_2, rho0_2, tzero=0, tau=integration_cut, epsrel=1.5e-8, use_nquad=False)[0]
				if integration_cut < time:
					charge_flow	+= charge_transmission(current_fct, time_evo_rho, rho0, tzero=integration_cut, tau=time, epsrel=1.5e-8, use_nquad=False)[0]
					#charge_flow	+= charge_transmission(current_fct_2, time_evo_rho_2, rho0_2, tzero=integration_cut, tau=time, epsrel=1.5e-8, use_nquad=False)[0]
				charge.append(charge_flow )

			cur	= (current_fct(time_evo_rho(rho0, time) ) + current_fct_2(time_evo_rho_2(rho0_2, time) ) )/2
			finite_cur.append(current_fct(time_evo_rho(rho0, time) ) )
		finite_cur	= np.array(finite_cur )
	else:
		finite_cur	= np.array([current_fct(time_evo_rho(rho0, time) ) for time in t])
	fs		= 16
	current_color	= 'b'
	
	if i_n:
		labels		= ['Current', 'Noise']
		ax.plot(t, finite_cur )
	else:
		labels		= []
		for k in lead:
			if np.less(finite_cur[:,k], 0).all() and logy:
				ax.plot(t, -finite_cur[:,k], c=current_color)
			else:
				ax.plot(t, finite_cur[:,k], c=current_color)

	ax.set_xlabel(r'$\tau \, [1/\Gamma]$', fontsize=fs)
	ax.set_ylabel(r'$I_{trans} \, [e\Gamma]$', fontsize=fs)
	ax.tick_params(axis='x', labelsize=fs)
	ax.locator_params(axis='y', nbins=3)
	ax.set_xlim([0.8e-4, 1.2e4])

	if plot_charge:
		ax_twin		= ax.twinx()
		color		= 'r'
		ax_twin.set_ylabel('Transmitted charge [e]', c=color, fontsize=fs)
		ax_twin.tick_params(axis='y', labelcolor=color, labelsize=fs)
		ax_twin.locator_params(axis='y', nbins=3)

		ax.set_ylabel(r'$I_{trans} \, [e\Gamma]$', c=current_color)
		ax.tick_params(axis='y', labelcolor=current_color, labelsize=fs)

		if not include_sys_2:
			charge		= np.cumsum(finite_cur[:,lead]*np.transpose([np.gradient(t)]) )
		ax_twin.plot(t, charge, c=color, label='Charge through lead {}'.format(lead))

	if logx:
		ax.set_xscale('log')
	if logy:
		ax.set_yscale('log')
	#ax.set_xticks([1e-3, 1e0, 1e3, 1e6] )
	ax.set_xticks([1e-4, 1e-2, 1e0, 1e2, 1e4] )

	#ax.grid(True)
	#ax.legend(labels=labels, loc=7)
	
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

def current(sys, i_n=False):
	current_rho	= lambda rho: current_via_sys(sys, rho, i_n=i_n)
	return current_rho

def current_via_sys(sys, rho, i_n=False):
	sys.phi0	= rho
	sys.appr.kernel_handler.set_phi0(rho)
	sys.current.fill(0.0)

	if i_n:
		sys.current_noise.fill(0.0)
		sys.appr.generate_current_noise()
		return np.array(sys.current_noise )
	
	sys.appr.generate_current()
	return np.array(sys.current )

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
	zeros		= np.zeros((par_occ, par_occ), dtype=complex )
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

def map_den_mat_to_vec(sys, rhoMat):
	num_occ, dof	= model_spec_dof_mat(rhoMat)
	ofd_dof		= dof - num_occ
	half_ofd_dof	= int(ofd_dof/2 )

	par_occ		= int(num_occ/2)

	rho		= np.zeros(dof)
	#rho[:num_occ]	+= np.diag(rhoMat ).real

	iteration_help	= np.zeros((par_occ, par_occ) )

	for indices, values in np.ndenumerate(iteration_help):
		rho[sys.si.get_ind_dm0(indices[0], indices[1], 0) ]	= rhoMat[indices].real
		rho[sys.si.get_ind_dm0(indices[0], indices[1], 1) ]	= rhoMat[indices[0]+par_occ, indices[1]+par_occ ].real
		if indices[0] < indices[1]:
			rho[sys.si.get_ind_dm0(indices[0], indices[1], 0)+half_ofd_dof ]	= rhoMat[indices].imag
			rho[sys.si.get_ind_dm0(indices[0], indices[1], 1)+half_ofd_dof ]	= rhoMat[indices[0]+par_occ, indices[1]+par_occ ].imag
	
	return rho

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
	time_evol	= lambda rho0, t: normed_occupations(np.matmul(np.transpose(np.tensordot(time_evol_mats, rho0.copy(), axes=1) ), np.exp(eigenval*t) ) )

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


if __name__=='__main__':
	main()
