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
import data_directory

import multiprocessing
from joblib import Parallel, delayed
from time import perf_counter
import scipy.optimize as opt

def main():
	eps	= 0e-4
	eps12 	= 0e-5
	eps34 	= 0e-5

	eps23	= 0e-3

	dphi	= 1e-5
	
	gamma 	= 0.1
	t 	= np.sqrt(gamma/(2*np.pi))+0.j

	phase1	= np.exp( +0j/2*np.pi + 1j*dphi )
	phase3	= np.exp( +0j/2*np.pi + 1j*dphi )

	theta_1	= 0.00*np.pi + dphi
	theta_2	= 0.37*np.pi - dphi
	theta_3	= 1.00*np.pi + 2*dphi
	theta_4	= 1/4*np.pi - 2*dphi
	factors	= [1.00, 1, 0.75, 1]*1/np.sqrt(1)

	thetas	= np.array([theta_1, theta_2, theta_3, theta_4])

	quasi_zero	= 0e-5

	tb1	= t*phase1
	tb2     = t
	tb3     = t*phase3

	tt4	= t

	tb11	= t
	tb21	= t
	tb31	= t

	tt41	= t

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

	model	= 1
	
	test_run	= False

	if model == 1:
		maj_op, overlaps, par	= majorana_leads(tb1, tb2, tb3, tt4, eps12, eps23, eps34)
	else:
		maj_op, overlaps, par	= abs_leads(tb1, tb11, tb2, tb21, tb3, tb31, tt4, tt41, eps)

	maj_box		= bc.majorana_box(maj_op, overlaps, Vg, 'asymmetric_box')
	maj_box.diagonalize()
	Ea		= maj_box.elec_en
	tunnel		= maj_box.constr_tunnel()

	if test_run:
		sys	= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=1)

		sys.solve(qdq=False, rotateq=False)

		print('Eigenenergies:', sys.Ea)
		print('Density matrix:', sys.phi0 )
		print('Current:', sys.current )

	fig, (ax1,ax2)	= plt.subplots(1, 2)


	bias_variation	= False
	if bias_variation:
		points	= 100
		m_bias	= 1e2
		x	= np.linspace(-m_bias, m_bias, points)
		y	= x
		X,Y	= np.meshgrid(x, y)
		I	= bias_sc.scan_and_plot(fig, ax1, X, Y, maj_box, t, par, tunnel, dband, mu_lst, T_lst, method, model)

	x	= np.linspace(-np.pi/2 -dphi , np.pi/2 + dphi, 10)
	X,Y	= np.meshgrid(x, x)
	tunnel_scan.phase_scan_and_plot(fig, ax1, X, Y, factors, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas=thetas, recalculate=False)

	x	= np.linspace(0, 2, 10)
	X, Y	= np.meshgrid(x, x)
	Y	+= dphi
	phases	= [0.0*np.pi, 0, 0.2*np.pi, 0]
	tunnel_scan.abs_scan_and_plot(fig, ax2, X, Y, phases, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas=thetas, recalculate=False)

	fig.tight_layout()
	plt.show()

def majorana_leads(tb1, tb2, tb3, tt4, eps12=0, eps23=0, eps34=0):
	overlaps	= np.array([[0, eps12, 0, 0], [0, 0, eps23, 0], [0, 0, 0, eps34], [0, 0, 0, 0]] )
	maj_op		= [fc.maj_operator(index=0, lead=[0], coupling=[tb1]), fc.maj_operator(index=1, lead=[0], coupling=[tb2]), \
					fc.maj_operator(index=2, lead=[0], coupling=[tb3]), fc.maj_operator(index=3, lead=[1], coupling=[tt4]) ]
	par		= np.array([0,0,1,1])
	return maj_op, overlaps, par

def abs_leads(tb10, tb11, tb20, tb21, tb30, tb31, tt40, tt41, eps=0):
	overlaps	= np.array([1, 2, 3, 4 ] )*eps
	overlaps	= fbr.default_overlaps(8, overlaps)

	maj_op		=  [fc.maj_operator(index=0, lead=[0], coupling=[tb10]), fc.maj_operator(index=1, lead=[0], coupling=[tb11]), \
				fc.maj_operator(index=2, lead=[0], coupling=[tb20]), fc.maj_operator(index=3, lead=[0], coupling=[tb21]), \
				fc.maj_operator(index=4, lead=[0], coupling=[tb30]), fc.maj_operator(index=5, lead=[0], coupling=[tb31]), \
				fc.maj_operator(index=6, lead=[1], coupling=[tt40]), fc.maj_operator(index=7, lead=[1], coupling=[tt41]) ]
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

if __name__=='__main__':
	main()
