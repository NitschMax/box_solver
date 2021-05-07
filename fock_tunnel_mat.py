import numpy as np
import matplotlib.pyplot as plt
import fock_class as fc

def main():
	N		= 2
	t		= 1e-6
	tLu		= t
	tLd		= t
	tRu		= t
	tRd		= t

	maj_op		= [fc.maj_operator(index=0, lead=0, coupling=tLu), fc.maj_operator(index=1, lead=1, coupling=tRu), \
		fc.maj_operator(index=2, lead=0, coupling=tLd), fc.maj_operator(index=3, lead=1, coupling=tRd) ]
	tunnel		= constr_tunnel_mat(N, maj_op)
	print(tunnel)

def constr_tunnel_mat(N, maj_op):
	fock_states	= fc.set_of_fock_states(N)
	if np.mod(N, 2) != 0:
		print('Number of Majoranas is not allowed!')
		return 0

	number_states	= fock_states.len
	half_number	= int(number_states/2)

	leads	= []
	for majorana in maj_op:
		leads	= np.concatenate((leads, majorana.lead) )
	number_leads	= int(max(leads ) ) +1

	counter	= 0
	tunnel	= np.zeros((number_leads, number_states, number_states), dtype=np.complex128)
	for majorana in maj_op:
		if not majorana.couples:
			continue
		for ket_ind in range(fock_states.len):
			state	= fock_states.get_state(ket_ind)
			occ_op	= majorana.maj_into_occ()
			for op in occ_op:
				state_cp	= state.copy()
				op.act_on(state_cp)
				if state_cp.valid:
					bra_ind		= fock_states.find(state_cp)
					for lead_ind, lead in enumerate(majorana.lead):
						if ket_ind > bra_ind:
							coupling	= majorana.coupling[lead_ind]
						else:
							coupling	= np.conj(majorana.coupling[lead_ind] )

						tunnel[lead, bra_ind, ket_ind]	+= coupling*state_cp.fac


	return tunnel

if __name__=='__main__':
	main()
