import numpy as np
import qmeq
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

def main():
	epsU = 0.0
	epsD = 0.0
	dphi	= 1e-10
	
	mu1	= 1e2
	mu2	= -mu1
	gamma = 0.1
	t 	= np.sqrt(gamma/(2*np.pi))+0.j
	phase	= np.exp(1j/4*np.pi-dphi )
	tLu	= t*phase
	tLd	= t
	tRu	= t
	tRd	= t
	
	T1	= 5e1
	T2 	= 5e1

	dband	= 1e5
	Vg	= 1.0
	
	nleads 	= 2
	T_lst 	= { 0:T1 , 1:T1}
	mu_lst 	= { 0:mu1 , 1:mu2}

	tunnel	= tunnel_from_t(tLu, tLd, tRu, tRd)

	sys	= qmeq.Builder_many_body(Ea=np.array([0.,epsU+epsD,epsU,epsD]), Na=np.array([0,0,1,1]), Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype='Lindblad', itype=1)

	sys.solve(qdq=False, rotateq=False)
	print('Eigenenergies:', sys.Ea)
	print('Density matrix:', sys.phi0 )
	print('Current:', sys.current )
	fig, (ax1,ax2)	= plt.subplots(1, 2)

	N	= 100
	m_bias	= 1e3
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
			Ea	= np.array([0., epsU+epsD, epsU-Vg, epsD-Vg])
			sys 	= qmeq.Builder_many_body(Ea=Ea, Na=np.array([0,0,1,1]), Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype='Lindblad', itype=1)
			sys.solve(qdq=False, rotateq=False)
			I_y.append(sys.current[0])

		I.append(I_y)

	c	= ax1.pcolor(X, Y, I, shading='auto')
	fig.colorbar(c, ax=ax1)

	angles	= np.linspace(1e-3, 2*np.pi, 1000)
	Vg	= 0
	mu_lst	= { 0:mu1, 1:mu2}

	I	= []
	for phi in angles:
		tLu	= np.exp(1j*phi)*t
		tunnel	= tunnel_from_t(tLu, tLd, tRu, tRd)
		sys	= qmeq.Builder_many_body(Ea=np.array([0.,epsU+epsD,epsU-Vg,epsD-Vg]), Na=np.array([0,0,1,1]), Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype='Lindblad', itype=1)
		sys.solve(qdq=False, rotateq=False)
		I.append(sys.current[0])

	ax2.grid(True)
	ax2.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
	ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_func) )
	ax2.set_xlabel(r'$\exp( i \Phi )$')
	ax2.set_ylabel('current')
	ax1.set_xlabel(r'$V_g$')
	ax1.set_ylabel(r'$V_{bias}$')
	plt.tight_layout()
	plt.show()

	ax1.plot(angles, I)
	plt.show()

def tunnel_from_t(tLu, tLd, tRu, tRd):
	tunnel=np.array([[
	[ 0.+0.j, 0.+0.j, tLu, tLd],
	[ 0.+0.j, 0.+0.j, tLd, -tLu],
	[ np.conj(tLu),  np.conj(tLd), 0.+0.j, 0.+0.j],
	[ np.conj(tLd), np.conj(-tLu), 0.+0.j, 0.+0.j]],
	[[ 0.+0.j, 0.+0.j, -1.j*tRu, -1.j*tRd],
	[  0.+0.j, 0.+0.j, 1.j*tRd, -1.j*tRu],
	[ 1.j*np.conj(tRu), -1.j*np.conj(tRd), 0.+0.j, 0.+0.j],
	[ 1.j*np.conj(tRd), 1.j*np.conj(tRu), 0.+0.j, 0.+0.j]]])

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
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)


if __name__=='__main__':
	main()
