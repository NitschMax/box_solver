import time_evolution as te
import fock_basis_rotation as fbr
import fock_class as fc

def main():
	np.set_printoptions(precision=3)
	eps12 	= 1e-5
	eps23	= 2e-5
	eps34 	= 3e-5

	eps	= 1e-3

	dphi	= 1e-6
	
	gamma 	= 1e+0
	gamma_u	= 1e+0
	t 	= np.sqrt(gamma/(2*np.pi))+0.j
	t_u	= np.sqrt(gamma_u/(2*np.pi))+0.j

	phases	= np.array([+1/2*np.pi-dphi, 0, +1/2*np.pi+dphi, 0] )
	phases	= np.exp(1j*phases )

	th	= [0.30, 0.30, 0.30, 0.30]
	th	= [+0.15, -0.40, +0.00, +0.20]
	th	= [0.00, 0.00, 0.00, 0.00]
	th	= [0.00, 0.00, 0.30, 0.00]

	thetas	= np.array(th )*np.pi + np.array([1, 2, 3, 4] )*dphi

	theta_phases	= np.exp( 1j*thetas)

	tunnel_mult	= [0.3, 0.3, 0.3, 0.3]
	tunnel_mult	= [0, 0, 0, 0]
	tunnel_mult	= [0.8, 0.7, 0.5, 1.0]
	tunnel_mult	= [1, 1, 1, 1]

	T1	= 1e2
	T2 	= T1

	v_bias	= 2e3
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
	itype	= 1

	model	= 1			# 1: Simple box, 2: Box with ABSs on every connection

def majorana_noise_box(tb1, tb2, tm2, tm3, tt4, eps12, ep23, eps34):
	overlaps	= np.array([[0, eps12, 0, 0], [0, 0, eps23, 0], [0, 0, 0, eps34], [0, 0, 0, 0]] )
	maj_op		= [fc.maj_operator(index=0, lead=[0], coupling=[tb1]), fc.maj_operator(index=1, lead=[0,1], coupling=[tb2,tm2]), \
					fc.maj_operator(index=2, lead=[1], coupling=[tm3]), fc.maj_operator(index=3, lead=[2], coupling=[tt4]) ]
	par		= np.array([0,0,1,1])
	return maj_op, overlaps, par

def abs_noise_box(tb10, tb11, tb20, tb21, tm20, tm21, tm30, tm31, tt40, tt41, eps=0)
	overlaps	= fbr.default_overlaps(8, overlaps)

	maj_op		=  [fc.maj_operator(index=0, lead=[0], coupling=[tb10]), fc.maj_operator(index=1, lead=[0], coupling=[tb11]), \
				fc.maj_operator(index=2, lead=[0,1], coupling=[tb20,tm20]), fc.maj_operator(index=3, lead=[0,1], coupling=[tb21,tm21]), \
				fc.maj_operator(index=4, lead=[1], coupling=[tm30]), fc.maj_operator(index=5, lead=[1], coupling=[tm31]), \
				fc.maj_operator(index=6, lead=[2], coupling=[tt40]), fc.maj_operator(index=7, lead=[2], coupling=[tt41]) ]
	par		= np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] )
	return maj_op, overlaps, par

if __name__=='__main__':
	main()
