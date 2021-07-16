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
	eps12 	= 0e-5
	eps34 	= 0e-5

	eps23	= 0e-3

	dphi	= 1e-5
	
	gamma 	= 0.1
	t 	= np.sqrt(gamma/(2*np.pi))+0.j

	phase1	= np.exp( +0j/2*np.pi + 1j*dphi )
	phase3	= np.exp( +0j/2*np.pi + 1j*dphi )

	theta_1	= np.exp( 2j/5*np.pi + 1j*dphi )
	theta_2	= np.exp( 3j/4*np.pi - 1j*dphi )
	theta_3	= np.exp( 2j/3*np.pi + 2j*dphi )
	theta_4	= np.exp( 0j/4*np.pi - 2j*dphi )

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

	bias	= 2e2
	mu1	= bias/2
	mu2	= -bias/2

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

	maj_box		= bc.majorana_box(maj_op, overlaps, Vg)
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
		X, Y, I		= bias_scan(maj_box, t, par, tunnel, dband, mu_lst, T_lst, method, model, thetas)
		xlablel1	= r'$V_g$'
		ylablel1	= r'$V_bias$'

	x	= np.linspace(0, 2, 100)
	X, Y	= np.meshgrid(x, x)
	Y	+= dphi
	I	= np.zeros(X.shape, dtype=np.float64)
	phases	= [0.0*np.pi, 0.2*np.pi]
	current_abs_value	= lambda factors: current(phases, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas=[], factors=factors)
	roots	= opt.fmin(current_abs_value, x0=[1,1], full_output=True )
	print('Factors with minimal current:', str(roots[0] ) )
	print('Minimal current: ', roots[1] )
	xlabel1	= r't_1'
	ylabel1	= r't_3'

	for indices,el in np.ndenumerate(I):
		break
		factors	= [X[indices], Y[indices] ]
		I[indices]	= current_abs_value(factors)
	
	c	= ax2.contourf(X, Y, I)
	cbar	= fig.colorbar(c, ax=ax2)
	ax2.scatter(roots[0][0], roots[0][1], marker='x', color='r')

	x	= np.linspace(-np.pi/2, np.pi/2, 100) + dphi
	
	X,Y	= np.meshgrid(x, x)
	I	= np.zeros(X.shape, dtype=np.float64 )
	max_occ	= []
	min_occ	= []

	factors	= [1.0, 0.0]*1/np.sqrt(1)
	print('Trying to find the roots.')
	roots	= opt.fmin(current, x0=[np.pi/4, np.pi/4], args=(maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, factors), full_output=True )
	print('Phase-diff with minimal current:', 'pi*'+str(roots[0]/np.pi) )
	print('Minimal current: ', roots[1] )

	for indices,el in np.ndenumerate(I):
		I[indices ]	= current([X[indices], Y[indices] ], maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, factors) 

	c	= ax1.contourf(X, Y, I)
	cbar2	= fig.colorbar(c, ax=ax1)

	ax1.scatter(roots[0][0], roots[0][1], marker='x', color='r')
	ax1.scatter(phases[0], phases[1], marker='o', color='black')

	fs	= 12

	ax2.locator_params(axis='both', nbins=5 )
	ax1.locator_params(axis='both', nbins=5 )
	cbar.ax.locator_params(axis='y', nbins=7 )
	cbar2.ax.locator_params(axis='y', nbins=7 )
	
	ax2.tick_params(labelsize=fs)
	ax1.tick_params(labelsize=fs)

	cbar.ax.set_title('current', size=fs)
	cbar.ax.tick_params(labelsize=fs)
	cbar2.ax.set_title('current', size=fs)
	cbar2.ax.tick_params(labelsize=fs)

	ax1.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
	ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_func) )
	ax1.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
	ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_func) )
	ax2.set_xlabel(xlabel1, fontsize=fs)
	ax2.set_ylabel(ylabel1, fontsize=fs)
	ax1.set_xlabel(r'$\Phi_{avg}$', fontsize=fs)
	ax1.set_ylabel(r'$\Phi_{diff}$', fontsize=fs)

	fig.tight_layout()
	
	plt.show()

def bias_scan(maj_box, t, par, tunnel, dband, mu_lst, T_lst, method, model, thetas=[]):
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
	return X, Y, I

def current(phases, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas=[], factors=[1.0, 1.0]):
	phi_1	= phases[0] + phases[1]
	phi_3	= phases[0] - phases[1]
	tb1	= np.exp(1j*phi_1 )*t*factors[0]
	tb2	= t
	tb3	= np.exp(1j*phi_3 )*t*factors[1]
	tt4	= t

	if len(thetas) == 4:
		tb11	= tb1*thetas[0]
		tb21	= tb2*thetas[1]
		tb31	= tb3*thetas[2]
		tt41	= tt4*thetas[3]

	if model == 1:
		maj_op, overlaps, par	= majorana_leads(tb1, tb2, tb3, tt4)
	else:
		maj_op, overlaps, par	= abs_leads(tb1, tb11, tb2, tb21, tb3, tb31, tt4, tt41)

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

def format_func(value, tick_number):
    # find number of multiples of pi/2
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N == -1:
        return r"$-\pi/2$"
    elif N == -2:
        return r"$-\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)

if __name__=='__main__':
	main()
