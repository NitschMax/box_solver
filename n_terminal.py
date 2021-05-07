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
	eps12 	= 2e-1
	eps34 	= 1e-1

	eps23	= 0e-1

	dphi	= 1e-5
	
	gamma 	= 0.1
	t 	= np.sqrt(gamma/(2*np.pi))+0.j
	phase	= np.exp(0j/2*np.pi + 1j*dphi )

	t1	= t*phase
	t2	= t
	t3	= t
	t4	= t
	t5	= 0

	T1	= 1e1
	T2 	= T1

	bias	= 2e2
	mu1	= bias/2
	mu2	= -mu1
	mu3	= 0

	dband	= 1e5
	Vg	= +1e1
	
	nleads 	= 2
	T_lst 	= { 0:T1 , 1:T1, 2:T1}
	mu_lst 	= { 0:mu1 , 1:mu2, 2:mu3}
	method	= 'Lindblad'

	maj_op, overlaps, par	= two_leads(t1, t2, t3, t4, t5, eps12, eps23, eps34 )

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
	fig.colorbar(c, ax=ax1)

	angles	= np.linspace(dphi, 2*np.pi+dphi, 1000)
	Vg	= 0e1
	maj_box.adj_charging(Vg)
	mu_lst	= { 0:mu1, 1:mu2}

	I	= []
	for phi in angles:
		t4	= np.exp(1j*phi)*t

		maj_op, overlaps, par	= two_leads(t1, t2, t3, t4, t5, eps12, eps23, eps34 )

		maj_box.change(majoranas = maj_op)
		tunnel		= maj_box.constr_tunnel()

		sys		= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=1)
		sys.solve(qdq=False, rotateq=False)
		I.append(sys.current[0])

	ax2.plot(angles, I, label=method)

	ax2.grid(True)
	ax2.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
	ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_func) )
	ax2.set_xlabel(r'$\exp( i \Phi )$')
	ax2.set_ylabel('current')
	ax1.set_xlabel(r'$V_g$')
	ax1.set_ylabel(r'$V_{bias}$')
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

def two_leads(t1, t2, t3, t4, t5, eps12, eps23, eps34):
	overlaps	= np.array([[0, eps12, 0, 0], [0, 0, eps23, 0], [0, 0, 0, eps34], [0, 0, 0, 0]] )
	maj_op		= [fc.maj_operator(index=0, lead=[0], coupling=[t1]), fc.maj_operator(index=1, lead=[0,1], coupling=[t2, t3]), \
					fc.maj_operator(index=2, lead=[1], coupling=[t4]), fc.maj_operator(index=3, lead=[2], coupling=[t5]) ]
	par		= np.array([0,0,1,1])
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
