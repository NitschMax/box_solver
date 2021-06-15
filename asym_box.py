import numpy as np
import qmeq
import matplotlib.pyplot as plt
import os
import fock_class as fc
import fock_tunnel_mat as ftm
import fock_basis_rotation as fbr
import box_class as bc

import multiprocessing
from joblib import Parallel, delayed
from time import perf_counter

def main():
	eps	= 1e-4
	eps12 	= 1e-3
	eps34 	= 1e-3

	eps23	= 0e-3

	dphi	= 1e-5
	
	gamma 	= 0.1
	t 	= np.sqrt(gamma/(2*np.pi))+0.j

	phase	= np.exp( 0j/3*np.pi + 1j*dphi )

	theta_1	= np.exp( 0j/4*np.pi + 1j*dphi )
	theta_2	= np.exp( 0j/4*np.pi - 1j*dphi )
	theta_3	= np.exp( 0j/4*np.pi + 2j*dphi )
	theta_4	= np.exp( 0j/4*np.pi - 2j*dphi )

	tb1	= t
	tb2     = t
	tb3     = t

	tt4	= t

	tb11	= t
	tb21	= t
	tb31	= t

	tt41	= t


	T1	= 1e1
	T2 	= T1

	bias	= 2e2
	mu1	= bias/2
	mu2	= -bias/2

	dband	= 1e5
	Vg	= +0e1
	
	T_lst 	= { 0:T1 , 1:T1}
	mu_lst 	= { 0:mu1 , 1:mu2}
	method	= 'Redfield'
	method	= 'Pauli'
	method	= '1vN'
	method	= 'Lindblad'

	model	= 1

	if model == 1:
		maj_op, overlaps, par	= majorana_leads(tb1, tb2, tb3, tt4, eps12, eps23, eps34)
	else:
		maj_op, overlaps, par	= abs_leads(tb1, tb11, tb2, tb21, tb3, tb31, tt4, tt41, eps)

	maj_box		= bc.majorana_box(maj_op, overlaps, Vg)
	maj_box.diagonalize()
	Ea		= maj_box.elec_en
	tunnel		= maj_box.constr_tunnel()
	print(Ea, par, tunnel)

	sys	= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=1)

	sys.solve(qdq=False, rotateq=False)

	print('Eigenenergies:', sys.Ea)
	print('Density matrix:', sys.phi0 )
	print('Current:', sys.current )

	fig, (ax1,ax2)	= plt.subplots(1, 2)

	points	= 100
	m_bias	= 1e2
	x	= np.linspace(-m_bias, m_bias, points)
	y	= x
	
	X,Y	= np.meshgrid(x, y)
	I	= np.zeros(X.shape, dtype=np.float64 )
	max_occ	= []
	min_occ	= []

	num_cores	= 4
	unordered_res	= Parallel(n_jobs=num_cores)(delayed(bias_sweep)(indices, bias, X[indices], I, maj_box, par, tunnel, dband, T_lst, method) for indices, bias in np.ndenumerate(Y) ) 

	for el in unordered_res:
		I[el[0] ]	= el[1]
		max_occ.append(el[2] )
		min_occ.append(el[3] )
	max_occ	= max(max_occ)
	min_occ	= min(min_occ)
	print('Maximal occupation:', max_occ)
	print('Minimal occupation:', min_occ)
	
	c	= ax1.pcolor(X, Y, I, shading='auto')
	cbar	= fig.colorbar(c, ax=ax1)

	angles	= np.linspace(0, 2*np.pi, 1000) + dphi
	Vg	= 0e1
	maj_box.adj_charging(Vg)
	Ea	= maj_box.elec_en
	mu_lst	= { 0:mu1, 1:mu2}

	I	= []
	for phi in angles:
		tb2	= np.exp(1j*phi)*t
		tm3	= np.exp(1j*phi)*t

		tb21	= tb2*theta_2
		tm31	= tm3*theta_3

		if model == 1:
			maj_op, overlaps, par	= majorana_leads(tb1, tb2, tb3, tt4)
		else:
			maj_op, overlaps, par	= abs_leads(tb1, tb11, tb2, tb21, tb3, tb31, tt4, tt41, eps)

		maj_box.change(majoranas = maj_op)
		tunnel		= maj_box.constr_tunnel()

		sys		= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=1)
		sys.solve(qdq=False, rotateq=False)
		I.append(sys.current[0])

	ax2.plot(angles, I, label=method)

	fs	= 12

	ax1.locator_params(axis='both', nbins=5 )
	ax2.locator_params(axis='both', nbins=5 )
	cbar.ax.locator_params(axis='y', nbins=7 )
	
	ax1.tick_params(labelsize=fs)
	ax2.tick_params(labelsize=fs)

	cbar.ax.set_title('current', size=fs)
	cbar.ax.tick_params(labelsize=fs)

	ax2.grid(True)
	ax2.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
	ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_func) )
	ax2.set_xlabel(r'$\exp( i \Phi )$', fontsize=fs)
	ax2.set_ylabel('current', fontsize=fs)
	ax1.set_xlabel(r'$V_g$', fontsize=fs)
	ax1.set_ylabel(r'$V_{bias}$', fontsize=fs)
	ax2.set_ylim(bottom=0)

	fig.tight_layout()
	
	plt.show()

def bias_sweep(indices, bias, Vg, I, maj_box, par, tunnel, dband, T_lst, method):
	mu_r	= -bias/2
	mu_l	= bias/2
	mu_lst	= { 0:mu_l, 1:mu_r}
	Ea	= maj_box.adj_charging(Vg)
	sys 	= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=1)
	sys.solve(qdq=False, rotateq=False)
	occ	= sys.phi0[:Ea.size]

	return [indices, sys.current[0], max(occ), min(occ)]

def majorana_leads(tb1, tb2, tb3, tt4, eps12=0, eps23=0, eps34=0):
	overlaps	= np.array([[0, eps12, 0, 0], [0, 0, eps23, 0], [0, 0, 0, eps34], [0, 0, 0, 0]] )
	maj_op		= [fc.maj_operator(index=0, lead=[0], coupling=[tb1]), fc.maj_operator(index=1, lead=[0], coupling=[tb2]), \
					fc.maj_operator(index=2, lead=[0], coupling=[tb3]), fc.maj_operator(index=3, lead=[1], coupling=[tt4]) ]
	par		= np.array([0,0,1,1])
	return maj_op, overlaps, par

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
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)

if __name__=='__main__':
	main()
