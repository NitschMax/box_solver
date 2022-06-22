import numpy as np
import qmeq
import matplotlib.pyplot as plt
import os
import fock_class as fc
import fock_tunnel_mat as ftm
import fock_basis_rotation as fbr
import box_class as bc
import data_directory as dd
import tunnel_scan as ts
import asym_box as box

import multiprocessing
from joblib import Parallel, delayed
from time import perf_counter
import scipy.optimize as opt

def main():
	eps	= 1e-7
	eps12 	= 1*eps
	eps34 	= 2*eps

	eps23	= 0*eps

	dphi	= 1e-6
	
	gamma 	= 1
	t 	= np.sqrt(gamma/(2*np.pi))+0.j

	phase1	= np.exp( +1j/2*np.pi + 1j*dphi )
	phase3	= np.exp( +0j/2*np.pi + 1j*dphi )

	tb1	= t*phase1
	tb2     = t
	tb3     = t*phase3

	tt4	= t

	theta_1	= 0.30*np.pi + 1*dphi
	theta_2	= 0.15*np.pi + 2*dphi
	theta_3	= 0.60*np.pi + 3*dphi
	theta_4	= 0.75*np.pi + 4*dphi

	thetas	= np.array([theta_1, theta_2, theta_3, theta_4])

	th	= [0.30, 0.00, 0.00, 0.00]
	th	= [0.30, 0.15, 0.60, 0.75]

	thetas	= np.array(th )*np.pi + np.array([1, 2, 3, 4] )*dphi

	theta_phases	= np.exp( 1j*thetas)

	tunnel_mult	= [0, 1, 1, 1]
	tunnel_mult	= [0.5, 0.6, 0.7, 0.8]
	tunnel_mult	= [0.5, 1.0, 1.0, 1]
	tunnel_mult	= [0.3, 0.4, 0.2, 0.5]
	tunnel_mult	= [1, 1, 1, 1]


	tb11	= tb1*theta_phases[0]*tunnel_mult[0]
	tb21	= tb2*theta_phases[1]*tunnel_mult[1]
	tb31	= tb3*theta_phases[2]*tunnel_mult[2]
	tt41	= tt4*theta_phases[3]*tunnel_mult[3]

	model	= 1

	T1	= 1e1
	T2 	= T1

	bias	= 2e2
	mu1	= bias/2
	mu2	= -bias/2

	dband	= 1e5
	Vg	= +0e1
	
	T_lst 	= { 0:T1 , 1:T1}
	mu_lst 	= { 0:mu1 , 1:mu2}
	method	= 'Pauli'
	method	= '1vN'
	method	= 'Lindblad'

	if model == 1:
		maj_op, overlaps, par	= box.majorana_leads(tb1, tb2, tb3, tt4, eps12, eps23, eps34)
	else:
		maj_op, overlaps, par	= box.abs_leads(tb1, tb11, tb2, tb21, tb3, tb31, tt4, tt41, eps)

	maj_box		= bc.majorana_box(maj_op, overlaps, Vg, name='asymmetric_box')
	maj_box.diagonalize()
	Ea		= maj_box.elec_en
	tunnel		= maj_box.constr_tunnel()

	sys	= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=1)

	sys.solve(qdq=False, rotateq=False)

	print('Eigenenergies:', sys.Ea)
	print('Density matrix:', sys.phi0 )
	print('Current:', sys.current )

	#fig, (ax1,ax2)	= plt.subplots(1,2)
	fig, ax2	= plt.subplots(1,1)

	recalculate	= True
	recalculate	= False

	save_result	= False
	save_result	= True

	logscale	= False
	logscale	= True

	plot_state_ov	= False
	plot_state_ov	= True

	block_state	= np.array([1, 0, 0, 0, 0, 0, 0, 0])
	block_state	= np.array([1, 1, 0, 0, +0, 0, +1, 0])*1/2
	block_state	= np.array([1, 1, 0, 0, +1, 0, +0, 0])*1/2
	block_state	= np.array([0, 1, 0, 0, 0, 0, 0, 0])		# [00, 11, 01, 10, Re(00/11), Re(01,10), Im(00,11), Im(01,10)]

	phi_1		= 1*np.pi/4
	phi_3		= 1*np.pi/2

	phi_avg		= (phi_1+phi_3)/2
	phi_diff	= (phi_1-phi_3)/2

	nu 		= 1*np.pi/8
	phi_avg		= +1*np.pi/4 + nu
	phi_diff	= +1*np.pi/4 - nu

	rotated_phases	= [phi_avg, 0, phi_diff, 0] 

	points	= 100
	points	= 50
	num_cores	= 6

	x	= np.linspace(1e-5, 2, points )
	y	= x
	
	X,Y	= np.meshgrid(x, y)

	#X,Y,I	= ts.abs_zero_scan_and_plot(fig, ax2, X, Y, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult, recalculate, num_cores, save_result, logscale)
	#I,roots	= ts.abs_scan_and_plot(fig, ax2, X, Y, rotated_phases, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult, recalculate, num_cores, logscale, plot_state_ov, block_state)

	points	= 100
	points	= 50

	x	= np.linspace(-np.pi/2-dphi, np.pi/2+dphi, points)
	X, Y	= np.meshgrid(x, x)
	X	+= dphi
	Y	-= dphi

	X,Y,I2,den_mat	= ts.phase_zero_scan_and_plot(fig, ax2, X, Y, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult, recalculate, num_cores, save_result, logscale)

	plt.tight_layout()
	plt.show()
	return

	#### Implementation of 1-d cuts ###
	points	= 400
	x	= np.linspace(1e-5, 2, points )
	delta	= x[1]-x[0]
	print(delta)
	Y	= np.ones((1, x.size) )*0.7
	X	= np.array([x] )
	X,Y,I	= ts.abs_zero_scan(X, Y, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult, recalculate, num_cores, save_result)
	ax1.plot(X[0,1:-1], np.diff(I[0], n=2 )/delta**2 )
	#ax1.plot(X[0,1:], np.diff(I[0], n=1 )/delta**1 )
	ax1.grid(True)
	#ax1.set_yscale('log')
	ax1.set_xlabel(r'$\frac{t_1}{t}$')
	ax1.set_ylabel(r'$\frac{d^2}{dt_1^2} \, I_{min}$')

	x	= np.linspace(-np.pi/2-dphi, np.pi/2+dphi, points)
	delta	= x[1]-x[0]
	Y	= np.ones((1, x.size) )*np.pi/3
	X	= np.array([x] )
	X,Y,I	= ts.phase_zero_scan(X, Y, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult, recalculate, num_cores, save_result)
	ax2.plot(X[0,1:-1], np.diff(I[0], n=2 )/delta**2 )
	#ax2.plot(X[0,1:], np.diff(I[0], n=1 )/delta**1 )
	ax2.grid(True)
	#ax2.set_yscale('log')
	ax2.set_xlabel(r'$\Phi_{avg}$')
	ax2.set_ylabel(r'$\frac{d^2}{d \phi_{avg}^2 } \, I_{min}$')

	plt.tight_layout()
	plt.show()



def current(phase, maj_box, t, Ea, dband, mu_lst, T_lst, method):
	phi_1	= phase
	phi_3	= -phase
	tb1	= np.exp(1j*phi_1 )*t
	tb2	= t
	tb3	= np.exp(1j*phi_3 )*t
	tt4	= t

	maj_op, overlaps, par	= majorana_leads(tb1, tb2, tb3, tt4)

	maj_box.change(majoranas = maj_op)
	tunnel		= maj_box.constr_tunnel()

	sys		= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=1)
	sys.solve(qdq=False, rotateq=False)

	return sys.current[0]

def current_2d(phases, maj_box, t, Ea, dband, mu_lst, T_lst, method, factors=[1.0, 1.0]):
	phi_1	= phases[0] + phases[1]
	phi_3	= phases[0] - phases[1]
	tb1	= t*np.exp(1j*phi_1 )*factors[0]
	tb2	= t
	tb3	= t*np.exp(1j*phi_3 )*factors[1]
	tt4	= t

	maj_op, overlaps, par	= majorana_leads(tb1, tb2, tb3, tt4)

	maj_box.change(majoranas = maj_op)
	tunnel		= maj_box.constr_tunnel()

	sys		= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=1)
	sys.solve(qdq=False, rotateq=False)

	return sys.current[0]

	
def bias_sweep(indices, bias, Vg, I, maj_box, par, tunnel, dband, T_lst, method):
	mu_r	= -bias/2
	mu_l	= bias/2
	mu_lst	= { 0:mu_l, 1:mu_r}
	Ea	= maj_box.adj_charging(Vg)
	sys 	= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=1)
	sys.solve(qdq=False, rotateq=False)
	occ	= sys.phi0[:Ea.size]

	return [indices, sys.current[0], max(occ), min(occ)]

def abs_leads(tb10, tb11, tb20, tb21, tm20, tm21, tm30, tm31, tt30, tt31, tt40, tt41, eps=0):
	overlaps	= np.array([1, 2, 3, 4 ] )*eps
	overlaps	= fbr.default_overlaps(8, overlaps)

	maj_op		=  [fc.maj_operator(index=0, lead=[0], coupling=[tb10]), fc.maj_operator(index=1, lead=[0], coupling=[tb11]), \
				fc.maj_operator(index=2, lead=[0,1], coupling=[tb20, tm20]), fc.maj_operator(index=3, lead=[0,1], coupling=[tb21,tm21]), \
				fc.maj_operator(index=4, lead=[1,2], coupling=[tm30, tt30]), fc.maj_operator(index=5, lead=[1,2], coupling=[tm31, tt31]), \
				fc.maj_operator(index=6, lead=[2], coupling=[tt40]), fc.maj_operator(index=7, lead=[2], coupling=[tt41]) ]
	par		= np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] )
	return maj_op, overlaps, par

def tunnel_mart(tb1, tb2, tm2, tm3, tt3, tt4):
	tunnel=np.array([[ \
	[ 0.+0.j, 0.+0.j, tb1-1.j*tb2, 0], \
	[ 0.+0.j, 0.+0.j, 0.+0.j, tb1+1.j*tb2], \
	[ np.conj(tb1)+1.j*np.conj(tb2), 0.+0.j, 0.+0.j, 0.+0.j], \
	[ 0.+0.j, np.conj(tb1)-1.j*np.conj(tb2), 0.+0.j, 0.+0.j]], \
	[[ 0.+0.j, 0.+0.j, -1.j*tm2, tm3], \
	[  0.+0.j, 0.+0.j, -tm3, 1.j*tm2], \
	[ 1.j*np.conj(tm2), -np.conj(tm3), 0.+0.j, 0.+0.j], \
	[ np.conj(tm3), -1.j*np.conj(tm2), 0.+0.j, 0.+0.j]], \
	[[ 0.+0.j, 0.+0.j, 0.+0.j, tt3-1.j*tt4], \
	[  0.+0.j, 0.+0.j, -tt3-1.j*tt4, 0.+0.j], \
	[ 0.+0.j, -np.conj(tt3)+1.j*np.conj(tt4), 0.+0.j, 0.+0.j], \
	[ np.conj(tt3)+1.j*np.conj(tt4), 0.+0.j, 0.+0.j, 0.+0.j]]])
	return tunnel

def format_func(value, tick_number):
    # find number of multiples of pi/2
    N = int(np.round(4 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/4$"
    elif N == 2:
        return r"$\pi/2$"
    elif N == -1:
        return r"$-\pi/4$"
    elif N == -2:
        return r"$-\pi/2$"
    elif N % 2 > 0:
        return r"${0}\pi/4$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)

if __name__=='__main__':
	main()
