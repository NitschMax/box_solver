import numpy as np
import qmeq
from joblib import Parallel, delayed

def scan_and_plot(fig, ax, X, Y, maj_box, t, par, tunnel, dband, mu_lst, T_lst, method, model, thetas=[]):
	bias_variation	= False
	I		= bias_scan(X, Y, maj_box, t, par, tunnel, dband, mu_lst, T_lst, method, model, thetas)
	
	ax.set_xlabel(r'$V_g$')
	ax.set_ylabel(r'$V_{bias}$')
	c	= ax.pcolor(X, Y, I, shading='auto')
	cbar	= fig.colorbar(c, ax=ax)

	return I

def bias_scan(X, Y, maj_box, t, par, tunnel, dband, mu_lst, T_lst, method, model, thetas=[]):
	I	= np.zeros(X.shape, dtype=np.float64 )
	max_occ	= []
	min_occ	= []

	num_cores	= 4
	unordered_res	= Parallel(n_jobs=num_cores)(delayed(bias_sweep)(indices, bias, X[indices], I, maj_box, par, tunnel, dband, T_lst, method) for indices, bias in np.ndenumerate(Y) ) 

	for el in unordered_res:
		I[el[0] ]	= el[1]
		max_occ.append(el[2] )
		min_occ.append(el[3] )
	max_occ	= max(max_occ)
	min_occ	= min(min_occ)
	print('Maximal occupation:', max_occ)
	print('Minimal occupation:', min_occ)
	return I

def bias_sweep(indices, bias, Vg, I, maj_box, par, tunnel, dband, T_lst, method):
	mu_r	= -bias/2
	mu_l	= bias/2
	mu_lst	= { 0:mu_l, 1:mu_r}
	Ea	= maj_box.adj_charging(Vg)
	sys 	= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=1)
	sys.solve(qdq=False, rotateq=False)
	occ	= sys.phi0[:Ea.size]

	return [indices, sys.current[0], max(occ), min(occ)]

