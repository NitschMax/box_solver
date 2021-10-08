import numpy as np
import matplotlib.pyplot as plt
import fock_class as fc
from scipy.linalg import eigh

def main():
	N		= 2

	overlaps	= np.array([[0, 0, 1, 0], [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0]] )*1e-3
	ham, U		= rotated_system(N, overlaps)
	print(overlaps)
	print(ham)

	overlaps	= default_overlaps(N)
	ham, U		= rotated_system(N, overlaps)
	print(overlaps)
	print(ham)


def default_overlaps(N, eps=[]):
	half_number	= int(N/2)
	if len(eps) != half_number:
		eps	= np.ones(half_number)
	nullen		= np.zeros((1, N ) )
	overlaps	= np.array([[]])
	for i in range(half_number):
		one_d_over		= nullen.copy()
		one_d_over[0, 2*i+1]	= eps[i]
		if i == 0:
			overlaps	= one_d_over
			overlaps	= np.concatenate((overlaps, nullen), axis=0)
		else:
			overlaps	= np.concatenate((overlaps, one_d_over), axis=0)
			overlaps	= np.concatenate((overlaps, nullen), axis=0)

	return overlaps

def rotated_system(N, overlaps):
	#overlaps	-= np.transpose(overlaps)
	#overlaps	/= 2
	fock_states	= fc.set_of_fock_states(N)
	number_states	= fock_states.len
	half_number	= int(number_states/2)
	hamilton	= hamilton_matrix(N, fock_states, overlaps )
	hamilton_e	= hamilton[:half_number,:half_number]
	hamilton_o	= hamilton[half_number:,half_number:]

	eigval_e, eigvec_e	= np.linalg.eigh(hamilton_e)
	eigval_o, eigvec_o	= np.linalg.eigh(hamilton_o)
	nullen			= np.zeros((half_number, half_number) )
	U			= np.matrix(np.block([[eigvec_e, nullen], [nullen, eigvec_o] ] ) )

	hamilton	= np.dot(U.getH(), np.dot(hamilton, U) ).real
	hamilton	= np.diagonal(hamilton)
	energies	= hamilton - np.amin(hamilton)

	return energies, U
	

def hamilton_matrix(N, fock_states, overlaps):
	hamilton_mat	= np.zeros((fock_states.len, fock_states.len), dtype=np.complex128 )
	for ket_ind in range(fock_states.len):
		state	= fock_states.get_state(ket_ind)
		for indexL in range(len(overlaps) ):
			gammaL	= fc.maj_operator(indexL)
			an_creL	= gammaL.maj_into_occ()

			for indexR in range(len(overlaps) ):
				gammaR	= fc.maj_operator(indexR)
				an_creR	= gammaR.maj_into_occ()
				op_comb	= associative_rule(an_creL, an_creR).flatten()

				epsilon	= -1j/2*overlaps[indexL, indexR]
				for arr in op_comb:
					state_cp	= state.copy()
					successive_operation(arr, state_cp)
					if state_cp.valid:
						state_cp.multiply(epsilon)
						bra_ind	= fock_states.find(state_cp )
						hamilton_mat[bra_ind, ket_ind] += state_cp.fac
		
	return hamilton_mat

def maj_into_occ(a, b):
	if b == 0:
		gamma	= [fc.operator(a, 1, 1), fc.operator(a, 0, 1) ]
	else:
		gamma	= [fc.operator(a, 1, 1j), fc.operator(a, 0, -1j) ]
	return gamma


def associative_rule(arr1, arr2):
	return np.matmul(np.transpose([arr1]), [arr2])

def successive_operation(op_arr, vec):
	for op in op_arr:
		op.act_on(vec)

def index_mapping(ind):
	k	= int(np.floor(ind/2) )
	l	= np.mod(ind, 2)
	return k, l

if __name__=='__main__':
	main()
