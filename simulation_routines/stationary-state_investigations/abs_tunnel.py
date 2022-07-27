from sympy.utilities.iterables import multiset_permutations
from itertools import permutations
import numpy as np
import qmeq
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import os

import box_solver as bs
import box_class as bc

def main():
	epsLu		= 1e-3
	epsLd		= 1e-4
	epsRu		= 1e-5
	epsRd		= 1e-6
	dphi	= 1e-5

	bias	= 2e2
	mu1	= bias/2
	mu2	= -mu1

	gamma 	= 0.1
	t 	= np.sqrt(gamma/(2*np.pi))+0.j
	phaseu	= np.exp(0j/4*np.pi + 1j*dphi )
	phased	= np.exp(0j/3*np.pi + 1j*dphi )
	phase	= np.exp(0j/2*np.pi + 1j*dphi )
	tLu1	= t*phase
	tLd1	= t
	tRu1	= t
	tRd1	= t

	tLu2	= tLu1
	tLd2	= tLd1
	tRu2	= tRu1
	tRd2	= tRd1
	
	T1	= 1e1
	T2 	= 1e1

	dband	= 1e5
	Vg	= +1e1
	
	nleads 	= 2
	T_lst 	= { 0:T1 , 1:T1}
	mu_lst 	= { 0:mu1 , 1:mu2}

	maj_op,overlaps,par	= bs.abs_box(tLu1, tLd1, tRu1, tRd1, tLu2, tLd2, tRu2, tRd2, epsLu, epsRu, epsLd, epsRd)
	maj_box		= bc.majorana_box(maj_op, overlaps, Vg)
	maj_box.diagonalize()
	Ea		= maj_box.elec_en
	tunnel		= maj_box.constr_tunnel()

	method	= 'Pauli'
	method	= 'Lindblad'
	U	= rotation()

	sys	= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=1)

	sys.solve(qdq=False, rotateq=False)
	print('Eigenenergies:', sys.Ea)
	print('Density matrix:', sys.phi0 )
	print('Current:', sys.current )
	fig, (ax1,ax2)	= plt.subplots(1, 2)

	N	= 100
	m_bias	= 1e2
	x	= np.linspace(-m_bias, m_bias, N)
	y	= x
	
	X,Y	= np.meshgrid(x, y)
	I	= []
	for bias in y:
		mu_r	= -bias/2
		mu_l	= bias/2
		mu_lst	= { 0:mu_l, 1:mu_r}
		I_y	= []
		for Vg in x:
			Ea	= maj_box.adj_charging(Vg)
			sys	= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=1)
			sys.solve(qdq=False, rotateq=False)
			I_y.append(sys.current[0] )

		I.append(I_y)

	c	= ax1.pcolor(X, Y, I, shading='auto')
	fig.colorbar(c, ax=ax1)

	angles	= np.linspace(dphi, 2*np.pi+dphi, 1000)
	bias	= 2e2
	mu1	= bias/2
	mu2	= -mu1
	Vg	= 0
	Ea	= maj_box.adj_charging(Vg)
	
	mu_lst	= { 0:mu1, 1:mu2}

	I	= []
	for phi in angles:
		tLu1	= np.exp(1j*phi)*t
		tLu2	= tLu1
		maj_op,overlaps,par	= bs.abs_box(tLu1, tRu1, tLd1, tRd1, tLu2, tRu2, tLd2, tRd2, epsLu, epsRu, epsLd, epsRd)
		maj_box.change(majoranas = maj_op)
		tunnel		= maj_box.constr_tunnel()

		sys	= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=1)
		sys.solve(qdq=False, rotateq=False)
		I.append(sys.current[0])
	
	ax2.plot(angles, I)
	ax2.grid(True)
	ax2.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
	ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_func) )
	ax2.set_xlabel(r'$\exp( i \Phi )$')
	ax2.set_ylabel('current')
	ax1.set_xlabel(r'$V_g$')
	ax1.set_ylabel(r'$V_{bias}$')
	plt.tight_layout()
	plt.show()

def rotation():
	N		= 4
	set_of_states	= construct_states(N)
	
	return 0


def abs_tunnel(tLu1, tLd1, tRu1, tRd1, tLu2, tLd2, tRu2, tRd2):
	tlst		= np.array([[tLu1, tLu2], [tLd1, tLd2], [tRu1, tRu2], [tRd1, tRd2]] )

	N		= 4
	set_of_states	= construct_states(N)
	number_states	= int(len(set_of_states) )
	mydict		= dict(zip([list_to_int(x) for x in set_of_states], range(len(set_of_states) ) ) )
	get_ind		= lambda lst: mydict.get(list_to_int(lst) )

	tunnelL		= np.matrix(np.zeros((number_states, number_states), dtype=np.complex128 ) )
	tunnelR		= np.matrix(np.zeros((number_states, number_states), dtype=np.complex128 ) )

	for ket in set_of_states[int(number_states/2):]:
		ket_ind		= get_ind(ket)
		for k in range(int(N/2)):
			bra	= ket.copy()
			occ	= ket[k]
			bra[k]	= invert(occ)
			bra_ind	= get_ind(bra)
			low_occ	= np.sum(ket[:k] )
			tunnelL[bra_ind, ket_ind]	= (-1)**low_occ*(1*tlst[k,0] + 1j*(-1)**occ*tlst[k,1] )

		for k in range(int(N/2), N):
			bra	= ket.copy()
			occ	= ket[k]
			bra[k]	= invert(occ)
			bra_ind	= get_ind(bra)
			low_occ	= np.sum(ket[:k] )
			tunnelR[bra_ind, ket_ind]	= (-1)**low_occ*(1*tlst[k,0] + 1j*(-1)**occ*tlst[k,1] )

	tunnelL		+= tunnelL.getH()
	tunnelR		+= tunnelR.getH()

	return np.array([tunnelL, tunnelR] )


def abs_energy(epslu=0, epsld=0, epsru=0, epsrd=0, Vg=0):
	N		= 4
	set_of_states	= construct_states(N)
	hamiltonian	= np.diag([epslu, epsld, epsru, epsrd ])
	eigen_energies	= np.array([np.sum(np.dot(hamiltonian, x))-np.mod(np.sum(x), 2)*Vg for x in set_of_states] )
	return eigen_energies
	
	
def invert(occ):
	if occ == 0:
		return 1
	else:
		return 0

def get_index(lst, dict):
	index	= dict.get(list_to_int(lst) )
	return index

def list_to_int(lst):
	return int(''.join(map(str, lst)), 2)

def int_to_list(zahl):
	return split('{0:04b}'.format(zahl) )

def split(word):
	return np.array([int(char) for char in word] )


def construct_states(N=2):
	set_of_states	= [] 
	
	numbers_el	= list(range(0,N+1,2) )
	for n in numbers_el:
		state		= np.zeros(N, dtype=np.int8)
		state[:n]	= 1
		permuted_states = np.flip(list(multiset_permutations(state ) ), axis=0)
		set_of_states.append(permuted_states )
		
	numbers_el	= list(range(1,N+1,2) )
	for n in numbers_el:
		state		= np.zeros(N, dtype=np.int8)
		state[:n]	= 1
		permuted_states = np.flip(list(multiset_permutations(state ) ), axis=0)
		set_of_states.append(permuted_states )
	
	set_of_states	= np.concatenate(set_of_states)
	return set_of_states

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
