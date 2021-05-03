import numpy as np
import qmeq
import matplotlib.pyplot as plt

def main():
	e_u	= 1e-3
	e_d	= e_u
	
	bias	= 1e3
	mu_r	= -bias/2
	mu_l	= bias/2
	T	= 5e1
	gamma	= 1
	t	= np.sqrt(gamma/(2*np.pi) )
	t_lu	= np.exp(1j/2*np.pi)*t
	t_ld	= t
	t_ru	= t
	t_rd	= t
	Vg	= -100

	nleads	= 2
	nsingle	= 2
	hsingle	= { (0,0):e_u-Vg, (1,1):e_d-Vg }
	mulst	= { 0:mu_l, 1:mu_r}
	tlst	= { 0:T, 1:T}

	tleads	= { (0,0):t_lu, (0,1):t_ld, (1,0):t_ru, (1,1):t_rd}
	coulomb	= { (0,1,1,0):+2*Vg}

	system	= qmeq.Builder(nleads=nleads, nsingle=nsingle, hsingle=hsingle, mulst=mulst, tlst=tlst, tleads=tleads, dband=1e5, kerntype='Lindblad', coulomb=coulomb)
	system.solve()
	print('Eigenenergies:', system.Ea)
	print('Density matrix:', system.phi0 )
	print('Current:', system.current )

	fig, (ax1,ax2)	= plt.subplots(1,2)
	degrees	= np.linspace(1e-5, 2*np.pi, 1000)
	I	= []
	for phi in degrees:
		t_lu	= np.exp(1j*phi)*t
		tleads	= { (0,0):t_lu, (0,1):t_ld, (1,0):t_ru, (1,1):t_rd}
		system.change(tleads=tleads)
		system.solve()
		I.append(system.current[0] )
	
	ax1.plot(degrees, I)
	
	t_lu	= np.exp(1j/2*np.pi)*t
	tleads	= { (0,0):t_lu, (0,1):t_ld, (1,0):t_ru, (1,1):t_rd}
	system.change(tleads=tleads)

	N	= 100
	m_bias	= 1e3
	x	= np.linspace(-m_bias, m_bias, N)
	y	= x
	
	X,Y	= np.meshgrid(x, y)
	I	= []
	for bias in y:
		mu_r	= -bias/2
		mu_l	= bias/2
		mulst	= { 0:mu_l, 1:mu_r}
		I_y	= []
		for Vg in x:
			coulomb	= { (0,1,1,0):+2*Vg}
			hsingle	= { (0,0):e_u-Vg, (1,1):e_d-Vg }
			system.change(mulst=mulst, hsingle=hsingle, coulomb=coulomb )
			system.solve()
			currents	= system.current
			I_y.append(currents[0] )

		I.append(I_y)

	c	= ax2.pcolor(X, Y, I, shading='auto')
	fig.colorbar(c, ax=ax2)
	

	plt.show()


if __name__=='__main__':
	main()
