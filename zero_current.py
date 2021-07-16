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
import scipy.optimize as opt

def main():
	eps	= 1e-4
	eps12 	= 0e-6
	eps34 	= 0e-6

	eps23	= 0e-3

	dphi	= 1e-6
	
	gamma 	= 0.1
	t 	= np.sqrt(gamma/(2*np.pi))+0.j

	phase1	= np.exp( +1j/2*np.pi + 1j*dphi )
	phase3	= np.exp( +0j/2*np.pi + 1j*dphi )

	tb1	= t*phase1
	tb2     = t
	tb3     = t*phase3

	tt4	= t

	T1	= 1e1
	T2 	= T1

	bias	= 2e2
	mu1	= bias/2
	mu2	= -bias/2

	dband	= 1e5
	Vg	= +0e1
	
	T_lst 	= { 0:T1 , 1:T1}
	mu_lst 	= { 0:mu1 , 1:mu2}
	method	= 'Lindblad'
	method	= '1vN'

	maj_op, overlaps, par	= majorana_leads(tb1, tb2, tb3, tt4, eps12, eps23, eps34)

	maj_box		= bc.majorana_box(maj_op, overlaps, Vg)
	maj_box.diagonalize()
	Ea		= maj_box.elec_en
	tunnel		= maj_box.constr_tunnel()

	sys	= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=1)

	sys.solve(qdq=False, rotateq=False)

	print('Eigenenergies:', sys.Ea)
	print('Density matrix:', sys.phi0 )
	print('Current:', sys.current )

	fig, (ax1,ax2)	= plt.subplots(1,2)

	x	= np.linspace(-np.pi/2-dphi, np.pi/2+dphi, 100)
	X, Y	= np.meshgrid(x, x)
	X	+= dphi
	Y	-= dphi
	I2	= np.zeros(X.shape, dtype=np.float64)
	x0		= [1,1]
	x0upperNeigh	= x0

	for indices, phase1 in np.ndenumerate(X):
		if indices[1] == 0:
			x0	= x0upperNeigh
		print(indices, x0)
		phase3	= Y[indices]
		phases	= [phase1, phase3]
		current_abs_value	= lambda factors: current_2d(phases, maj_box, t, Ea, dband, mu_lst, T_lst, method, factors)
		roots	= opt.fmin(current_abs_value, x0=x0, full_output=True, maxiter=200 )
		flag	= roots[4]
		if flag	== 2:
			roots	= opt.fmin(current_abs_value, x0=[1,1], full_output=True, maxiter=200 )
		x0	= roots[0]
		if indices[1] == 0:
			x0upperNeigh	= x0
		print(roots[1] )
		I2[indices]	= roots[1]


	I2	= np.round(I2, 5)
	I2	= np.ceil(I2)

	c2	= ax2.contourf(X, Y, I2)
	cbar2	= fig.colorbar(c2, ax=ax2)
	print('Factors with minimal current:', str(roots[0] ) )
	print('Minimal current: ', roots[1] )

	x	= np.linspace(1e-5, 1, 100 )
	y	= x
	
	X,Y	= np.meshgrid(x, y)
	I	= np.ones(X.shape, dtype=np.float64 )
	x0		= [np.pi/4, np.pi/4]
	x0upperNeigh	= x0

	for indices, t1 in np.ndenumerate(X):
		if indices[1] == 0:
			x0	= x0upperNeigh
		t3	= Y[indices]
		factors	= [t1, t3]
		roots	= opt.fmin(current_2d, x0=x0, args=(maj_box, t, Ea, dband, mu_lst, T_lst, method, factors ), maxiter=100, full_output=True )
		flag	= roots[4]
		if flag	== 2:
			roots	= opt.fmin(current_abs_value, x0=[np.pi/4,np.pi/4], full_output=True, maxiter=200 )
		x0	= roots[0]
		if indices[1] == 0:
			x0upperNeigh	= x0
		print(roots[1] )
		I[indices]	= roots[1]

	fs	= 13
	ax1.set_xlabel(r'$t_1$', fontsize=fs)
	ax1.set_ylabel(r'$t_3$', fontsize=fs)
	ax2.set_xlabel(r'$\Phi_{avg}$', fontsize=fs)
	ax2.set_ylabel(r'$\Phi_{diff}$', fontsize=fs)

	I	= np.round(I, 5)
	I	= np.ceil(I)

	c1	= ax1.contourf(X, Y, I)
	cbar1	= fig.colorbar(c1, ax=ax1)

	ax1.locator_params(axis='both', nbins=5 )
	ax2.locator_params(axis='both', nbins=5 )
	cbar1.ax.locator_params(axis='y', nbins=7 )
	cbar2.ax.locator_params(axis='y', nbins=7 )
	
	ax1.tick_params(labelsize=fs)
	ax2.tick_params(labelsize=fs)

	cbar1.ax.set_title('rounded current', size=fs)
	cbar1.ax.tick_params(labelsize=fs)
	cbar2.ax.set_title('current', size=fs)
	cbar2.ax.tick_params(labelsize=fs)

	ax2.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
	ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_func) )
	ax2.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
	ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_func) )

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
