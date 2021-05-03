
import numpy as np
import qmeq
import matplotlib.pyplot as plt
import os
import fock_class as fc
import fock_tunnel_mat as ftm
import fock_basis_rotation as fbr
import box_class as bc

def main():
	epsU = 2e-3
	epsD = 1e-3

	epsL = 2e-9
	epsR = 1e-9

	dphi	= 1e-5
	
	bias	= 2e2
	mu1	= bias/2
	mu2	= -mu1
	gamma 	= 0.1
	t 	= np.sqrt(gamma/(2*np.pi))+0.j
	tLu	= np.exp(0j/2*np.pi + 1j*dphi )*t
	tLd	= t
	tRu	= t
	tRd	= t
	
	T1	= 5e1
	T2 	= 5e1

	dband	= 1e5
	Vg	= +1e1
	
	nleads 	= 2
	T_lst 	= { 0:T1 , 1:T1}
	mu_lst 	= { 0:mu1 , 1:mu2}
	method	= 'Lindblad'

	N		= 2
	maj_op,overlaps	= simple_box(tLu, tRu, tLd, tRd, epsU, epsD, epsL, epsR)
	maj_box		= bc.majorana_box(maj_op, overlaps, Vg)
	maj_box.diagonalize()
	Ea		= maj_box.elec_en
	tunnel		= maj_box.constr_tunnel()
	
	sys	= qmeq.Builder_many_body(Ea=Ea, Na=np.array([0,0,1,1]), Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=1)

	sys.solve(qdq=False, rotateq=False)
	print('Eigenenergies:', sys.Ea)
	print('Density matrix:', sys.phi0 )
	print('Current:', sys.current )
	fig, (ax1,ax2)	= plt.subplots(1, 2)

	points	= 100
	m_bias	= 1e3
	x	= np.linspace(-m_bias, m_bias, points)
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
			sys 	= qmeq.Builder_many_body(Ea=Ea, Na=np.array([0,0,1,1]), Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=1)
			sys.solve(qdq=False, rotateq=False)
			I_y.append(sys.current[0])

		I.append(I_y)

	c	= ax1.pcolor(X, Y, I, shading='auto')
	fig.colorbar(c, ax=ax1)

	angles	= np.linspace(dphi, 2*np.pi+dphi, 1000)
	Vg	= 0
	maj_box.adj_charging(Vg)
	mu_lst	= { 0:mu1, 1:mu2}

	I	= []
	for phi in angles:
		tLu	= np.exp(1j*phi)*t

		maj_op,overlaps	= simple_box(tLu, tRu, tLd, tRd, epsU, epsD, epsL, epsR)
		maj_box.change(majoranas = maj_op)
		tunnel		= maj_box.constr_tunnel()

		sys		= qmeq.Builder_many_body(Ea=Ea, Na=np.array([0,0,1,1]), Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=1)
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

	fig.tight_layout()
	
	plt.show()

def simple_box(tLu, tRu, tLd, tRd, epsU, epsD, epsL, epsR):
	overlaps	= np.array([[0, epsU, epsL, 0], [0, 0, 0, epsR], [0, 0, 0, epsD], [0, 0, 0, 0]] )
	maj_op		= [fc.maj_operator(index=0, lead=0, coupling=tLu), fc.maj_operator(index=1, lead=1, coupling=tRu), \
					fc.maj_operator(index=2, lead=0, coupling=tLd), fc.maj_operator(index=3, lead=1, coupling=tRd) ]
	return maj_op, overlaps

def abs_box(tLu1, tRu1, tLd1, tRd1, tLu2, tRu2, tLd2, tRd2, epsLu, epsRu, epsLd, epsRd):
	maj_op		= [fc.maj_operator(index=0, lead=0, coupling=tLu1), fc.maj_operator(index=1, lead=0, coupling=tLu2), \
				fc.maj_operator(index=2, lead=1, coupling=tRu1), fc.maj_operator(index=3, lead=1, coupling=tRu2), \
				fc.maj_operator(index=4, lead=0, coupling=tLd1), fc.maj_operator(index=5, lead=0, coupling=tLd2), 
				fc.maj_operator(index=6, lead=1, coupling=tRd1), fc.maj_operator(index=7, lead=1, coupling=tRd2) ]
	overlaps	= fbr.default_overlaps(4, [epsLu, epsRu, epsLd, epsRd] )
	return maj_op, overlaps

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
