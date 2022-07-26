import numpy as np
import qmeq
import matplotlib.pyplot as plt
import os
import fock_class as fc
import fock_tunnel_mat as ftm
import fock_basis_rotation as fbr
import box_class as bc
import box_solver as bs

import multiprocessing
from joblib import Parallel, delayed
from time import perf_counter
from scipy.linalg import eig
from scipy.special import digamma

def main():
	epsU = 0e-3
	epsD = +0e-3

	epsL = 0e-3
	epsR = 0e-3

	epsLu	= +0e-3
	epsLd	= +0e-8
	epsRu	= 0e-4
	epsRd	= 0e-5

	epsMu	= 0e-3
	epsMd	= 0e-3

	model	= 1

	dphi	= +0e-6

	gamma 	= 1.0
	t 	= np.sqrt(gamma/(2*np.pi))+0.j
	phase	= np.exp(+1j/2*np.pi + 1j*dphi )
	theta_u	= np.exp( 1j/5*np.pi + 1j*dphi )
	theta_d	= np.exp( 0j/5*np.pi + 1j*dphi )
	faktorU	= 1e-0
	faktorD	= 1e-0
	faktorR	= 1e-0

	tLu	= t*phase
	tLd	= t
	tRu	= t*faktorR
	tRd	= t*faktorR

	tLu2	= tLu*theta_u*faktorU
	tLd2	= tLd*theta_d*faktorD
	tRu2	= tRu*theta_u
	tRd2	= tRd*theta_d

	T1	= 1e2
	T2 	= T1

	bias	= 2e3
	mu1	= +bias/2
	mu2	= -mu1

	dband	= 1e6
	Vg	= +0e1

	principle_int_l	= digamma(0.5-1j*mu1/(2*np.pi*T1) ).real - np.log(dband/(2*np.pi*T1 ) )
	principle_int_r	= digamma(0.5-1j*mu2/(2*np.pi*T1) ).real - np.log(dband/(2*np.pi*T1 ) )
	print(principle_int_l )
	print(principle_int_r )
	
	T_lst 	= { 0:T1 , 1:T1}
	mu_lst 	= { 0:mu1 , 1:mu2}
	method	= 'Pauli'
	method	= 'Lindblad'
	method	= 'Redfield'
	method	= '1vN'
	itype	= 1

	if model == 1:
		maj_op, overlaps, par	= bs.simple_box(tLu, tRu, tLd, tRd, epsU, epsD, epsL, epsR)
	elif model == 2:
		maj_op, overlaps, par	= bs.abs_box(tLu, tRu, tLd, tRd, tLu2, tRu2, tLd2, tRd2, epsLu, epsRu, epsLd, epsRd)
	elif model == 3:
		maj_op, overlaps, par	= bs.eight_box(tLu, tRu, tLd, tRd, epsLu, epsMu, epsRu, epsLd, epsMd, epsRd, epsL, epsR)
	elif model == 4:
		maj_op, overlaps, par	= bs.six_box(tLu, tRu, tLd, tRd, epsLu, epsMu, epsRu, epsD, epsL, epsR)

	maj_box		= bc.majorana_box(maj_op, overlaps, Vg)
	maj_box.diagonalize()
	Ea		= maj_box.elec_en
	tunnel		= maj_box.constr_tunnel()
	maj_box.print_eigenstates()

	sys	= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=itype)

	sys.solve(qdq=False, rotateq=False)

	print('Eigenenergies:', sys.Ea)
	print('Density matrix:', sys.phi0 )
	print('Current:', sys.current )
	np.set_printoptions(precision=1)

	kernel	= sys.kern
	print(kernel)
	#eigensys	= eig(kernel)

	method	= 'Lindblad'
	method	= '1vN'
	itype	= 2
	sys	= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=itype)
	sys.solve(qdq=False, rotateq=False)
	kernel2	= sys.kern
	diff_kernel	= kernel - kernel2
	print(np.round(diff_kernel, 15) )

	return

if __name__=='__main__':
	main()
