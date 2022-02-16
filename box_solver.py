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
from scipy.linalg import eig
from scipy.special import digamma
import matplotlib.colors as colors

def main():
	np.set_printoptions(precision=7)
	epsU = 0e-4
	epsD = +0.0e-4

	epsL = 0e-3
	epsR = 0e-3

	epsLu	= +1e-4
	epsLd	= +1e-6
	epsRu	= +1e-5
	epsRd	= +1e-5

	epsMu	= +1e-6
	epsMd	= +1e-6

	model	= 2

	dphi	= +1e-6

	gamma 	= 1.0
	t 	= np.sqrt(gamma/(2*np.pi))+0.j
	phase	= np.exp(+1j/2*np.pi + 1j*dphi )
	phaseR	= np.exp(+0j/2*np.pi + 1j*dphi )
	theta_u	= np.exp( 1.0j/3*np.pi + 1j*dphi )
	theta_d	= np.exp( 0.0j/2*np.pi + 1j*dphi )
	faktorU	= 1.0
	faktorD	= 1.0
	faktorR	= 1e-0

	tLu	= t*phase
	tLd	= t
	tRu	= t*phaseR*faktorR
	tRd	= t*faktorR

	tLu2	= tLu*theta_u*faktorU
	tLd2	= tLd*theta_d*faktorD
	tRu2	= tRu*theta_u*faktorU
	tRd2	= tRd*theta_d*faktorD

	T1	= 1e2
	T2 	= T1

	bias	= 2e3
	mu1	= +bias/2
	mu2	= -mu1/1

	dband	= 1e4
	Vg	= +0e1
	
	T_lst 	= { 0:T1 , 1:T1}
	mu_lst 	= { 0:mu1 , 1:mu2}
	method	= 'Redfield'
	method	= 'Pauli'
	method	= 'Lindblad'
	method	= '1vN'
	itype	= 2

	maj_op, overlaps, par	= box_definition(model, tLu, tRu, tLd, tRd, tLu2, tRu2, tLd2, tRd2, epsU, epsD, epsL, epsR, epsLu, epsRu, epsLd, epsRd, epsMu, epsMd)

	maj_box		= bc.majorana_box(maj_op, overlaps, Vg)
	maj_box.diagonalize()
	Ea		= maj_box.elec_en
	tunnel		= maj_box.constr_tunnel()
	maj_box.print_eigenstates()

	sys	= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=itype)

	sys.solve(qdq=False, rotateq=False)
	kernel	= sys.kern
	eigensys	= eig(kernel)

	print('Eigenenergies:', sys.Ea)
	print('Density matrix:', sys.phi0 )
	print('Current:', sys.current )

	principle_int_l	= (digamma(0.5-1j*mu1/(2*np.pi*T1) ).real - np.log(dband/(2*np.pi*T1 ) ) )/(2*np.pi)
	if model == 1:
		analyt_current	= gamma*(epsU/2)**2/(1+(2*principle_int_l)**2)
	elif model == 2:
		analyt_current	= 'Unknown yet'
	elif model == 3:
		analyt_current	= gamma*(epsLu/2)**2/(1+(2*principle_int_l)**2)
	elif model == 4:
		analyt_current	= gamma*(epsLu/2)**2/(1+(2*principle_int_l)**2)
	print('Analytical current:', analyt_current )

	bias_plot	= True
	bias_plot	= False
	if bias_plot:
		bias_plot_create(maj_box, par, tunnel, dband, T_lst, mu_lst, method, itype, gamma)

	bias_variation	= True
	bias_variation	= False
	if bias_variation:
		bias_variation_create(Vg, maj_box, par, tunnel, dband, T_lst, method, itype)
		
	phase_plot	= True
	phase_plot	= False

	vary_left	= False
	vary_left	= True

	vary_right	= True
	vary_right	= False

	models		= [1, 3]
	models		= [model]

	if phase_plot:
		phase_plot_create(maj_box, vary_left, vary_right, models, gamma, dband, mu_lst, T_lst, method, itype, faktorU, faktorD, faktorR, theta_u, theta_d, tLu, tRu, tLd, tRd, tLu2, tRu2, tLd2, tRd2, epsU, epsD, epsL, epsR, epsLu, epsRu, epsLd, epsRd, epsMu, epsMd)

	energy_plot	= True
	energy_plot	= False
	if energy_plot:
		energy_plot_create(maj_box, theta_u, faktorU, model, t, tLu, tRu, tLd, tRd, tLu2, tRu2, tLd2, tRd2, epsU, epsD, epsL, epsR, epsLu, epsRu, epsLd, epsRd, epsMu, epsMd, dband, mu_lst, T_lst, method, itype)


	rate_plot	= True
	rate_plot	= False
	if rate_plot:
		rate_plot_create(t, theta_u, theta_d, faktorU, faktorD, faktorR, maj_box, model, tLu, tRu, tLd, tRd, tLu2, tRu2, tLd2, tRd2, epsU, epsD, epsL, epsR, epsLu, epsRu, epsLd, epsRd, epsMu, epsMd, dband, mu_lst, T_lst, method, itype)
	
	exp_minimization_plot	= False
	exp_minimization_plot	= True

	if exp_minimization_plot:
		exp_minimization_plot_create(maj_box, Ea, par, dband, mu_lst, T_lst, method, itype, model, t, theta_u, theta_d, phaseR, faktorU, faktorD, faktorR, epsU, epsD, epsL, epsR, epsLu, epsRu, epsLd, epsRd, epsMu, epsMd, dphi)

def exp_minimization_plot_create(maj_box, Ea, par, dband, mu_lst, T_lst, method, itype, model, t, theta_u, theta_d, phaseR, faktorU, faktorD, faktorR, epsU, epsD, epsL, epsR, epsLu, epsRu, epsLd, epsRd, epsMu, epsMd, dphi):
	points	= 100
	angles	= np.linspace(dphi, 2*np.pi+dphi, points)
	faktors	= np.linspace(0, 2, points)
	gamma	= 2*np.pi*np.abs(t)**2

	X, Y	= np.meshgrid(angles, faktors)

	num_cores	= 4
	I	= Parallel(n_jobs=num_cores)(delayed(some_func)(X[indices], Y[indices], maj_box, Ea, par, dband, mu_lst, T_lst, method, itype, model, t, theta_u, theta_d, phaseR, faktorU, faktorD, faktorR, epsU, epsD, epsL, epsR, epsLu, epsRu, epsLd, epsRd, epsMu, epsMd, dphi) for indices, angle in np.ndenumerate(X) )
	I	= np.array(I).reshape((points, points) )

	fs	= 24

	fig, ax1	= plt.subplots(1,1)
	c	= ax1.pcolormesh(X, Y, I/gamma, norm=colors.LogNorm(vmin=I.min(), vmax=I.max()), cmap='PuBu_r', shading='auto', rasterized=True)

	#cticks	= [0, 1]
	cbar	= fig.colorbar(c, ax=ax1, extend='max' )
	ax1.locator_params(axis='both', nbins=5 )
	ax1.set_xlabel(r'$\phi_L$', fontsize=fs)
	ax1.set_ylabel(r'$t_{Ld}/t_{Lu} $', fontsize=fs)

	ax1.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
	ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_func) )

	ax1.tick_params(labelsize=fs)
	#cbar.ax.set_ylim(-1, 1 )
	#cbar.ax.locator_params(axis='y', nbins=7 )
	cbar.ax.set_title(r'$I/e \Gamma$', size=fs)
	cbar.ax.tick_params(labelsize=fs)
	plt.tight_layout()
	plt.show()
	plt.clf()

def some_func(phi, faktor, maj_box, Ea, par, dband, mu_lst, T_lst, method, itype, model, t, theta_u, theta_d, phaseR, faktorU, faktorD, faktorR, epsU, epsD, epsL, epsR, epsLu, epsRu, epsLd, epsRd, epsMu, epsMd, dphi):
	phase	= np.exp(1j*phi + 1j*dphi)
	tLu	= t*phase
	tLd	= t*faktor

	tRu	= t*phaseR*faktorR
	tRd	= t*faktorR

	tLu2	= tLu*theta_u*faktorU
	tLd2	= tLd*theta_d*faktorD
	tRu2	= tRu*theta_u*faktorU
	tRd2	= tRd*theta_d*faktorD

	maj_op, overlaps, par	= box_definition(model, tLu, tRu, tLd, tRd, tLu2, tRu2, tLd2, tRd2, epsU, epsD, epsL, epsR, epsLu, epsRu, epsLd, epsRd, epsMu, epsMd)

	maj_box.change(majoranas = maj_op)
	tunnel		= maj_box.constr_tunnel()

	sys		= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=itype)
	sys.solve(qdq=False, rotateq=False)
	return sys.current[0]


def bias_plot_create(maj_box, par, tunnel, dband, T_lst, mu_lst, method, itype, gamma):
	fig, ax1	= plt.subplots(1, 1)

	points	= 100
	m_bias	= mu_lst[0] - mu_lst[1]
	x	= np.linspace(-m_bias, m_bias, points)
	y	= x
	
	X,Y	= np.meshgrid(x, y)
	I	= np.zeros(X.shape, dtype=np.float64 )
	T1	= T_lst[0]

	num_cores	= 4
	unordered_res	= Parallel(n_jobs=num_cores)(delayed(bias_sweep)(indices, bias, X[indices], I, maj_box, par, tunnel, dband, T_lst, method, itype) for indices, bias in np.ndenumerate(Y) )
	for el in unordered_res:
		I[el[0] ]	= el[1]
	
	fs	= 24

	c	= ax1.pcolormesh(X/T1, Y/T1, I/gamma, shading='auto', rasterized=True)
	cticks	= [-1, 0, 1]
	cbar	= fig.colorbar(c, ax=ax1, ticks=cticks)
	ax1.locator_params(axis='both', nbins=5 )
	ax1.set_xlabel(r'$V_g/T$', fontsize=fs)
	ax1.set_ylabel(r'$V_{bias}/T$', fontsize=fs)
	ax1.tick_params(labelsize=fs)
	cbar.ax.set_ylim(-1, 1 )
	cbar.ax.locator_params(axis='y', nbins=7 )
	cbar.ax.set_title(r'$I/e \Gamma$', size=fs)
	cbar.ax.tick_params(labelsize=fs)
	plt.tight_layout()
	plt.show()
	plt.clf()

def bias_variation_create(Vg, maj_box, par, tunnel, dband, T_lst, method, itype):
	T1	= T_lst[0]
	x	= np.linspace(5*T1, 3*dband, 1000)
	maxI	= 0
	platzhalter	= 0
	for itype in [1,2]:
		I	= np.zeros(x.shape )
		I	= np.array([bias_sweep(0, item, Vg, I, maj_box, par, tunnel, dband, T_lst, method, itype) for item in x] )[:,1]
		if itype == 1:
			label	= 'with P. I.'
			maxI	= np.max(I )
		else:
			label	= 'without P. I.'
		plt.plot(x/T1, I, label=label )
		platzhalter	= I
	plt.xlabel(r'$V_{bias}/T_1$' )
	plt.ylabel(r'$I_{rem}$' )
	plt.scatter(2*dband/T1, maxI, marker='x', label=r'$2 \cdot$bandwidth')
	#plt.yscale('log')
	plt.legend()
	plt.grid(True)
	plt.show()

def phase_plot_create(maj_box, vary_left, vary_right, models, gamma, dband, mu_lst, T_lst, method, itype, faktorU, faktorD, faktorR, theta_u, theta_d, tLu, tRu, tLd, tRd, tLu2, tRu2, tLd2, tRd2, epsU, epsD, epsL, epsR, epsLu, epsRu, epsLd, epsRd, epsMu, epsMd):
	if vary_left and vary_right:
		variation_label	= r'$\Phi_L, \, Phi_R$'
	elif vary_left:
		variation_label	= r'$\Phi_L$'
	elif vary_right:
		variation_label	= r'$\Phi_R$'

	fig, ax2	= plt.subplots(1, 1)

	dphi	= 1e-6
	angles	= np.linspace(dphi, 2*np.pi+dphi, 1000)
	Vg	= 0e1
	maj_box.adj_charging(Vg)
	t 	= np.sqrt(gamma/(2*np.pi))+0.j

	for model in models:
		print('Calculating phase plot for model', model)
		I	= []

		maj_op, overlaps, par	= box_definition(model, tLu, tRu, tLd, tRd, tLu2, tRu2, tLd2, tRd2, epsU, epsD, epsL, epsR, epsLu, epsRu, epsLd, epsRd, epsMu, epsMd)
		if model == 1:
			plot_label	= 'Pure Box'
			xi	= (epsU - epsD)/2/gamma
		elif model == 2:
			plot_label	= 'ABS Box'
			xi	= 1
		elif model == 3:
			plot_label	= '\'Dirty\' Box'
			xi	= (epsLu - epsLd)/2/gamma
		else:
			plot_label	= ''

		maj_box		= bc.majorana_box(maj_op, overlaps, Vg)
		maj_box.diagonalize()
		Ea		= maj_box.elec_en

		for phi in angles:
			if vary_left:
				tLu	= np.exp(1j*phi)*t
				tLu2	= tLu*theta_u*faktorU

			if vary_right:
				tRu	= np.exp(1j*phi)*t*faktorR
				tRu2	= tRu*theta_u*faktorU

			maj_op, overlaps, par	= box_definition(model, tLu, tRu, tLd, tRd, tLu2, tRu2, tLd2, tRd2, epsU, epsD, epsL, epsR, epsLu, epsRu, epsLd, epsRd, epsMu, epsMd)

			maj_box.change(majoranas = maj_op)
			tunnel		= maj_box.constr_tunnel()

			sys		= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=itype)
			sys.solve(qdq=False, rotateq=False)
			I.append(sys.current[0])
		I	= np.array(I)
		ax2.plot(angles, I/gamma/xi**2, label=plot_label, linewidth=3)
		#ax2.plot(angles, 100*I/gamma, label=plot_label, linewidth=3)

	maj_op, overlaps, par	= box_definition(model, tLu, tRu, tLd, tRd, tLu2, tRu2, tLd2, tRd2, epsU, epsD, epsL, epsR, epsLu, epsRu, epsLd, epsRd, epsMu, epsMd)
	maj_box.diagonalize()
	Ea	= maj_box.elec_en

	fs	= 18

	ax2.grid(True)
	ax2.locator_params(axis='x', nbins=5 )
	ax2.locator_params(axis='y', nbins=3 )
	ax2.tick_params(labelsize=fs)

	ax2.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
	ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_func) )
	ax2.set_xlabel(variation_label, fontsize=fs)
	#ax2.set_ylabel(r'$100 \cdot I/e \Gamma $', fontsize=fs)
	ax2.set_ylabel(r'$I_{rem}/(e \Gamma \xi^2) $', fontsize=fs)

	#ax2.yaxis.tick_right()
	#ax2.yaxis.set_label_position("right")

	#ax2.set_ylim(bottom=0, top=1)
	#ax2.set_xlim([np.pi/2-0.1, np.pi/2+0.1] )

	fig.tight_layout()

	plt.legend(fontsize=18, loc=0)
	plt.show()

def energy_plot_create(maj_box, theta_u, faktorU, model, t, tLu, tRu, tLd, tRd, tLu2, tRu2, tLd2, tRd2, epsU, epsD, epsL, epsR, epsLu, epsRu, epsLd, epsRd, epsMu, epsMd, dband, mu_lst, T_lst, method, itype ):
	fig, ax2	= plt.subplots(1, 1)

	energy_range	= np.linspace(1e-9, 1e-3, 100)

	Vg	= 0e1
	maj_box.adj_charging(Vg)

	gamma	= 2*np.pi*np.abs(t)**2
	phi	= np.pi/2
	tLu	= np.exp(1j*phi)*t
	tLu2	= tLu*theta_u*faktorU

	I	= []
	for eps in energy_range:
		epsLu	= 0*eps
		epsLd	= 0*eps
		epsRu	= 0*eps
		epsRd	= 0*eps

		epsMu	= 0e-2
		epsMd	= 0e-2

		epsU	= 1*eps
		epsD	= 2*eps

		maj_op, overlaps, par	= box_definition(model, tLu, tRu, tLd, tRd, tLu2, tRu2, tLd2, tRd2, epsU, epsD, epsL, epsR, epsLu, epsRu, epsLd, epsRd, epsMu, epsMd)

		maj_box.change(majoranas = maj_op, overlaps=overlaps)
		maj_box.diagonalize()
		tunnel		= maj_box.constr_tunnel()
		Ea		= maj_box.elec_en

		sys		= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=itype)
		sys.solve(qdq=False, rotateq=False)
		I.append(sys.current[0])

	I	= np.array(I)
	fs	= 20

	print((energy_range/2/gamma)**2/I*gamma )
	ax2.plot((energy_range/gamma)**2, I/gamma, label='numerical result', marker='x')
	ax2.plot((energy_range/gamma)**2, (energy_range/2/gamma)**2, label='analytical prediction simple box')
	ax2.grid(True)
	ax2.locator_params(axis='both', nbins=5 )
	ax2.tick_params(labelsize=fs)

	ax2.set_xlabel(r'$\left( \frac{\epsilon}{\Gamma} \right)^2$', fontsize=fs)
	ax2.set_ylabel(r'$\frac{I}{e \Gamma}$', fontsize=fs)
	ax2.set_ylim(bottom=0)

	plt.legend()
	fig.tight_layout()
	plt.show()

def rate_plot_create(t, theta_u, theta_d, faktorU, faktorD, faktorR, maj_box, model, tLu, tRu, tLd, tRd, tLu2, tRu2, tLd2, tRd2, epsU, epsD, epsL, epsR, epsLu, epsRu, epsLd, epsRd, epsMu, epsMd, dband, mu_lst, T_lst, method, itype):
	fig, ax2	= plt.subplots(1, 1)

	rate_range	= np.linspace(1e-7, 1e-0, 1000)
	Vg	= 0e1
	maj_box.adj_charging(Vg)

	I	= []
	for rate in rate_range:
		t 	= np.sqrt(rate/(2*np.pi))+0.j
		
		phi	= np.pi/2
		tLu	= np.exp(1j*phi)*t
		tLd	= t
		tRu	= t*faktorR
		tRd	= t*faktorR

		tLu2	= tLu*theta_u*faktorU
		tLd2	= tLd*theta_d*faktorD
		tRu2	= tRu*theta_u*faktorU
		tRd2	= tRd*theta_d*faktorD

		maj_op, overlaps, par	= box_definition(model, tLu, tRu, tLd, tRd, tLu2, tRu2, tLd2, tRd2, epsU, epsD, epsL, epsR, epsLu, epsRu, epsLd, epsRd, epsMu, epsMd)

		maj_box.change(majoranas = maj_op)
		tunnel		= maj_box.constr_tunnel()
		Ea		= maj_box.elec_en

		sys		= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=itype)
		sys.solve(qdq=False, rotateq=False)
		I.append(sys.current[0])

	I	= np.array(I)
	fs	= 18

	ax2.plot(rate_range, I, linewidth=3, label=r'$\epsilon_{Lu} = 0.1$')
	ax2.grid(True)
	ax2.locator_params(axis='x', nbins=3 )
	ax2.locator_params(axis='y', nbins=4 )
	ax2.tick_params(labelsize=fs)

	ax2.set_xlabel(r'$\Gamma$', fontsize=fs)
	ax2.set_ylabel(r'$I_{rem}/e$', fontsize=fs)
	ax2.set_ylim(bottom=0)

	#plt.legend(fontsize=fs)
	fig.tight_layout()
	plt.show()

def bias_sweep(indices, bias, Vg, I, maj_box, par, tunnel, dband, T_lst, method, itype):
	mu_r	= -bias/2
	mu_l	= bias/2
	mu_lst	= { 0:mu_l, 1:mu_r}
	Ea	= maj_box.adj_charging(Vg)
	sys 	= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=itype)
	sys.solve(qdq=False, rotateq=False)

	return [indices, sys.current[0] ]

def box_definition(model, tLu, tRu, tLd, tRd, tLu2, tRu2, tLd2, tRd2, epsU, epsD, epsL, epsR, epsLu, epsRu, epsLd, epsRd, epsMu, epsMd):
	if model == 1:
		maj_op, overlaps, par	= simple_box(tLu, tRu, tLd, tRd, epsU, epsD, epsL, epsR)
	elif model == 2:
		maj_op, overlaps, par	= abs_box(tLu, tRu, tLd, tRd, tLu2, tRu2, tLd2, tRd2, epsLu, epsRu, epsLd, epsRd)
	elif model == 3:
		maj_op, overlaps, par	= eight_box(tLu, tRu, tLd, tRd, epsLu, epsMu, epsRu, epsLd, epsMd, epsRd, epsL, epsR)
	elif model == 4:
		maj_op, overlaps, par	= six_box(tLu, tRu, tLd, tRd, epsLu, epsMu, epsRu, epsD, epsL, epsR)

	return maj_op, overlaps, par


def simple_box(tLu, tRu, tLd, tRd, epsU, epsD, epsL, epsR):
	maj_op		= [fc.maj_operator(index=0, lead=[0], coupling=[tLu]), fc.maj_operator(index=1, lead=[0], coupling=[tLd]), \
				fc.maj_operator(index=2, lead=[1], coupling=[tRu]), fc.maj_operator(index=3, lead=[1], coupling=[tRd]) ]
	overlaps	= np.zeros( (4,4) )
	overlaps[0,1]	+= epsL
	overlaps[0,2]	+= epsU
	overlaps[1,3]	+= epsD
	overlaps[2,3]	+= epsR
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

def eight_box(tLu, tRu, tLd, tRd, epsLu, epsMu, epsRu, epsLd, epsMd, epsRd, epsL, epsR):
	maj_op		= [fc.maj_operator(index=0, lead=[0], coupling=[tLu]), fc.maj_operator(index=1, lead=[0], coupling=[tLd]),
				fc.maj_operator(index=2, lead=[], coupling=[]), fc.maj_operator(index=3, lead=[], coupling=[]), \
		 		fc.maj_operator(index=4, lead=[], coupling=[]), fc.maj_operator(index=5, lead=[], coupling=[]), \
				fc.maj_operator(index=6, lead=[1], coupling=[tRu]), fc.maj_operator(index=7, lead=[1], coupling=[tRd]) \
 ]
	N		= len(maj_op )
	overlaps	= np.zeros((N, N) )
	overlaps[0, 1]	= epsL
	overlaps[2, 3]	= epsMu
	overlaps[4, 5]	= epsMd
	overlaps[6, 7]	= epsR

	overlaps[0, 2]	= epsLu
	overlaps[3, 6]	= epsRu

	overlaps[1, 4]	= epsLd
	overlaps[5, 7]	= epsRd

	par		= np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
	return maj_op, overlaps, par

def six_box(tLu, tRu, tLd, tRd, epsLu, epsMu, epsRu, epsD, epsL, epsR):
	maj_op		= [fc.maj_operator(index=0, lead=[0], coupling=[tLu]), fc.maj_operator(index=1, lead=[0], coupling=[tLd]), \
				fc.maj_operator(index=2, lead=[], coupling=[]), fc.maj_operator(index=3, lead=[], coupling=[]), \
				fc.maj_operator(index=4, lead=[1], coupling=[tRu]), fc.maj_operator(index=5, lead=[1], coupling=[tRd]) ]
	N		= len(maj_op )
	overlaps	= np.zeros((6, 6) )
	overlaps[0,1]	+= epsL
	overlaps[2,3]	+= epsMu
	overlaps[2,3]	+= epsR

	overlaps[0,2]	+= epsLu
	overlaps[3,4]	+= epsRu
	overlaps[1,5]	+= epsD

	par		= np.array([0,0,0,0,1,1,1,1])
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
