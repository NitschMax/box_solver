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
	epsU = 0e-1
	epsD = 0e-1

	epsL = 0e-6
	epsR = 0e-6

	epsLu	= 2e-3
	epsLd	= 1e-3
	epsRu	= 2e-3
	epsRd	= 1e-3

	epsMu	= 0e-9
	epsMd	= 0e-9

	model	= 2
	
	dphi	= 1e-6
	
	gamma 	= 0.1
	t 	= np.sqrt(gamma/(2*np.pi))+0.j
	phase	= np.exp( -0j/2*np.pi + 1j*dphi )
	theta_u	= np.exp( 1j/5*np.pi + 1j*dphi )
	theta_d	= np.exp( 0j/5*np.pi + 1j*dphi )
	faktorU	= 1e-0
	faktorD	= 1e-0
	faktorR	= 1e-1

	tLu	= t*phase
	tLd	= t
	tRu	= t*faktorR
	tRd	= t*faktorR

	tLu2	= tLu*theta_u*faktorU
	tLd2	= tLd*theta_d*faktorD
	tRu2	= tRu*theta_u
	tRd2	= tRd*theta_d

	T1	= 1e1
	T2 	= T1

	bias	= 2e2
	mu1	= +bias/2
	mu2	= -mu1

	dband	= 1e5
	Vg	= +1e1
	
	nleads 	= 2
	T_lst 	= { 0:T1 , 1:T1}
	mu_lst 	= { 0:mu1 , 1:mu2}
	method	= 'Redfield'
	method	= 'Pauli'
	method	= '1vN'
	method	= 'Lindblad'

	if model == 1:
		maj_op, overlaps, par	= simple_box(tLu, tRu, tLd, tRd, epsU, epsD, epsL, epsR)
	elif model == 2:
		maj_op, overlaps, par	= abs_box(tLu, tRu, tLd, tRd, tLu2, tRu2, tLd2, tRd2, epsLu, epsRu, epsLd, epsRd)
	else:
		maj_op, overlaps, par	= eight_box(tLu, tLd, tRu, tRd, epsLu, epsMu, epsRu, epsLd, epsMd, epsRd)

	maj_box		= bc.majorana_box(maj_op, overlaps, Vg)
	maj_box.diagonalize()
	Ea		= maj_box.elec_en
	tunnel		= maj_box.constr_tunnel()

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

	num_cores	= 4
	unordered_res	= Parallel(n_jobs=num_cores)(delayed(bias_sweep)(indices, bias, X[indices], I, maj_box, par, tunnel, dband, T_lst, method) for indices, bias in np.ndenumerate(Y) ) 
	for el in unordered_res:
		I[el[0] ]	= el[1]
	
	c	= ax1.pcolor(X, Y, I, shading='auto')
	cbar	= fig.colorbar(c, ax=ax1)

	angles	= np.linspace(dphi, 2*np.pi+dphi, 1000)
	Vg	= 0e1
	maj_box.adj_charging(Vg)
	mu_lst	= { 0:mu1, 1:mu2}

	I	= []
	for phi in angles:
		tLu	= np.exp(1j*phi)*t
		tLu2	= tLu*theta_u*faktorU

		if model == 1:
			maj_op, overlaps, par	= simple_box(tLu, tRu, tLd, tRd, epsU, epsD, epsL, epsR)
		elif model == 2:
			maj_op, overlaps, par	= abs_box(tLu, tRu, tLd, tRd, tLu2, tRu2, tLd2, tRd2, epsLu, epsRu, epsLd, epsRd)
		else:
			maj_op, overlaps, par	= eight_box(tLu, tRu, tLd, tRd, epsLu, epsMu, epsRu, epsLd, epsMd, epsRd)

		maj_box.change(majoranas = maj_op)
		tunnel		= maj_box.constr_tunnel()

		sys		= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=1)
		sys.solve(qdq=False, rotateq=False)
		I.append(sys.current[0])

	ax2.plot(angles, I, label=method)

	fs	= 12

	ax2.grid(True)

	ax1.locator_params(axis='both', nbins=5 )
	ax2.locator_params(axis='both', nbins=5 )
	cbar.ax.locator_params(axis='y', nbins=7 )
	
	ax1.tick_params(labelsize=fs)
	ax2.tick_params(labelsize=fs)

	cbar.ax.set_title('current', size=fs)
	cbar.ax.tick_params(labelsize=fs)

	ax2.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
	ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_func) )
	ax2.set_xlabel(r'$\Phi$', fontsize=fs)
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

	return [indices, sys.current[0] ]

def simple_box(tLu, tRu, tLd, tRd, epsU, epsD, epsL, epsR):
	overlaps	= np.array([[0, epsU, epsL, 0], [0, 0, 0, epsR], [0, 0, 0, epsD], [0, 0, 0, 0]] )
	maj_op		= [fc.maj_operator(index=0, lead=[0], coupling=[tLu]), fc.maj_operator(index=1, lead=[1], coupling=[tRu]), \
					fc.maj_operator(index=2, lead=[0], coupling=[tLd]), fc.maj_operator(index=3, lead=[1], coupling=[tRd]) ]
	par		= np.array([0,0,1,1])
	return maj_op, overlaps, par

def abs_box(tLu1, tRu1, tLd1, tRd1, tLu2, tRu2, tLd2, tRd2, epsLu, epsRu, epsLd, epsRd):
	maj_op		= [fc.maj_operator(index=0, lead=[0], coupling=[tLu1]), fc.maj_operator(index=1, lead=[0], coupling=[tLu2]), \
				fc.maj_operator(index=2, lead=[1], coupling=[tRu1]), fc.maj_operator(index=3, lead=[1], coupling=[tRu2]), \
				fc.maj_operator(index=4, lead=[0], coupling=[tLd1]), fc.maj_operator(index=5, lead=[0], coupling=[tLd2]), 
				fc.maj_operator(index=6, lead=[1], coupling=[tRd1]), fc.maj_operator(index=7, lead=[1], coupling=[tRd2]) ]
	N		= len(maj_op )
	overlaps	= fbr.default_overlaps(N, [epsLu, epsRu, epsLd, epsRd] )
	par		= np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
	return maj_op, overlaps, par

def eight_box(tLu, tRu, tLd, tRd, epsLu, epsMu, epsRu, epsLd, epsMd, epsRd):
	maj_op		= [fc.maj_operator(index=0, lead=[0], coupling=[tLu]), fc.maj_operator(index=1, lead=[], coupling=[]), \
				fc.maj_operator(index=2, lead=[], coupling=[]), fc.maj_operator(index=3, lead=[1], coupling=[tRu]), \
				fc.maj_operator(index=4, lead=[0], coupling=[tLd]), fc.maj_operator(index=5, lead=[], coupling=[]), 
				fc.maj_operator(index=6, lead=[], coupling=[]), fc.maj_operator(index=7, lead=[1], coupling=[tRd]) ]
	N		= len(maj_op )
	nullen		= np.zeros((4, 4) )
	overlapsU	= np.diag([epsLu, epsMu, epsRu], k=1 )
	overlapsD	= np.diag([epsLd, epsMd, epsRd], k=1 )
	overlaps	= np.matrix( np.block( [[overlapsU, nullen], [nullen, overlapsD]] ) )

	par		= np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
	return maj_op, overlaps, par

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
