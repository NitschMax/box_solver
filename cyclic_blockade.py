import numpy as np
import qmeq
import matplotlib.pyplot as plt
import os

from scipy.optimize import minimize

import time_evolution as te
import box_class as bc


def main():
	np.set_printoptions(precision=3)
	eps12 	= 1e-4
	eps23	= 2e-4
	eps34 	= 3e-4

	eps	= 1e-2

	dphi	= 1e-6
	
	gamma 	= 1e+0
	gamma_u	= 1e+0
	t 	= np.sqrt(gamma/(2*np.pi))+0.j
	t_u	= np.sqrt(gamma_u/(2*np.pi))+0.j

	phases	= np.array([+1/2*np.pi-dphi, 0, +1/2*np.pi+dphi, 0] )
	phases	= np.exp(1j*phases )

	th	= [0.00, 0.00, 0.00, 0.00]
	th	= [0.30, 0.30, 0.30, 0.30]
	th	= [0.10, 0.40, 0.30, 0.20]
	th	= [0.00, 0.00, 0.00, 0.00]

	thetas	= np.array(th )*np.pi + np.array([1, 2, 3, 4] )*dphi

	theta_phases	= np.exp( 1j*thetas)

	tunnel_mult	= [0, 1, 1, 1]
	tunnel_mult	= [0.3, 0.3, 0.3, 0.3]
	tunnel_mult	= [0.4, 0.3, 0.3, 0.4]
	tunnel_mult	= [1, 1, 1, 1]

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

	model	= 2
	initialization	= 1			# Determines how the system is initialized before the cycle, 0: trivial initialization, 1: stationary state of z-blockade, 2: finite time evolution in z-blockade

	lead	= 0
	
	run_optimization	= False
	run_optimization	= True
	if run_optimization:
		guess_z	= np.array([np.pi/2, np.pi/2, 1, 0] )
		guess_x	= np.array([np.pi/2, np.pi/2, 0, 1] )
		phases_z, factors_z	= find_blockade(model, t, t_u, theta_phases, tunnel_mult, dphi, guess_z )
		phases_x, factors_x	= find_blockade(model, t, t_u, theta_phases, tunnel_mult, dphi, guess_x )
	else:
		factors_z	= [1.00, 1, 0.00, 1]
		factors_x	= [0.00, 1, 1.00, 1]
		phases_z	= phases
		phases_x	= phases

	waiting_time	= 1e3*1/gamma
	n		= 10

	print(factors_z, phases_z)
	print(factors_x, phases_x)

	Ea, tunnel_z, par	= box_preparation(t, t_u, phases_z, factors_z, theta_phases, tunnel_mult, eps12, eps23, eps34, eps, Vg, model)
	Ea, tunnel_x, par	= box_preparation(t, t_u, phases_x, factors_x, theta_phases, tunnel_mult, eps12, eps23, eps34, eps, Vg, model)

	sys		= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel_z, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=itype)
	sys.solve(qdq=False, rotateq=False)
	time_evo_rho_z	= te.finite_time_evolution(sys)
	current_fct_z	= te.current(sys, lead=lead)

	sys		= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel_x, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=itype)
	sys.solve(qdq=False, rotateq=False)
	time_evo_rho_x	= te.finite_time_evolution(sys)
	current_fct_x	= te.current(sys, lead=lead)
	
	rho0	= state_preparation(sys, model, initialization)

	average_charge_of_cycle	= charge_transmission_cycle(sys, current_fct_z, current_fct_x, time_evo_rho_z, time_evo_rho_x, rho0, n, waiting_time)
	print('The choosen cycle setup transmits per switch: ', average_charge_of_cycle )

	return

def find_blockade(model, t, t_u, theta_phases, tunnel_mult, dphi, guess ):
	func	= lambda x: check_for_blockade(lead_connections(model, t, t_u, [np.exp(1j*x[0]), 1, np.exp(1j*x[1]), 1], [x[2], 1, x[3], 1], theta_phases, tunnel_mult ) )
	result	= minimize(func, guess, tol=dphi).x
	phases	= np.exp(1j*np.array([result[0], 0, result[1], 0] ) + 1j*np.array([-dphi, 0, dphi, 0]) )
	factors = np.array([result[2], 1, result[3], 1] )
	return phases, factors

def charge_transmission_cycle(sys, current_fct_1, current_fct_2, time_evo_1, time_evo_2, rho0, n, T):
	rho	= rho0
	charge_transmitted	= 0
	integration_range	= 10
	pre_run	= 1e3
	print('Pre-running system {} cycles'.format(pre_run) )
	for k in range(int(pre_run) ):
		rho	= time_evo_1(rho, T)
		rho	= time_evo_2(rho, T)

	for i in range(n):
		print('Cycle {} of {}'.format(i+1, n) )
		charge_transmitted	+= te.charge_transmission(sys, current_fct_1, time_evo_1, rho, tau=integration_range)[0]
		rho	= time_evo_1(rho, T)

		charge_transmitted	+= te.charge_transmission(sys, current_fct_2, time_evo_2, rho, tau=integration_range)[0]
		rho	= time_evo_2(rho, T)

	return charge_transmitted/(2*n )
		
def state_preparation(sys, model, initialization=0):
	if initialization==0:
		rho0		= te.abs_block(model )
	elif initialization==1:
		rho0	= sys.phi0
	elif initialization==2:
		rho0		= te.abs_block(model )
	return rho0

def lead_connections(model, t, t_u, phases, factors, theta_phases, tunnel_mult, lead=0):
	tb1, tb2, tb3, tt4, tb11, tb21, tb31, tt41	= te.tunnel_coupl(t, t_u, phases, factors, theta_phases, tunnel_mult)
	if model == 1 or model == 3:
		if lead	== 0:
			tunnel_amplitudes	= [tb1, tb2, tb3]
		elif lead == 1:
			tunnel_amplitudes	= [tt4]
	elif model == 2:
		if lead	== 0:
			tunnel_amplitudes	= [tb1, tb2, tb3, tb11, tb21, tb31]
		elif lead == 1:
			tunnel_amplitudes	= [tt4, tt41]
	return np.array(tunnel_amplitudes )

def check_for_blockade(tunnel_amplitudes ):
	blockade_cond	= np.abs(np.sum(tunnel_amplitudes**2 ) )
	return blockade_cond
	if blockade_cond < 1e-3:
		return True
	else:
		return False

def box_preparation(t, t_u, phases, factors, theta_phases, tunnel_mult, eps12, eps23, eps34, eps, Vg, model):
	tb1, tb2, tb3, tt4, tb11, tb21, tb31, tt41	= te.tunnel_coupl(t, t_u, phases, factors, theta_phases, tunnel_mult)
	maj_op, overlaps, par	= te.box_definition(model, tb1, tb2, tb3, tt4, tb11, tb21, tb31, tt41, eps12, eps23, eps34, eps)

	maj_box		= bc.majorana_box(maj_op, overlaps, Vg, 'asymmetric_box')
	maj_box.diagonalize()
	Ea		= maj_box.elec_en
	tunnel		= maj_box.constr_tunnel()

	return Ea, tunnel, par


if __name__=='__main__':
	main()

