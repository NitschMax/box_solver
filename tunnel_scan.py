import numpy as np
import scipy.optimize as opt
import asym_box as box
import qmeq

def abs_scan(X, Y, phases, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas=[]):
	I	= np.zeros(X.shape, dtype=np.float64)
	current_abs_value	= lambda factors: current(phases, factors, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas=[])
	roots	= opt.fmin(current_abs_value, x0=[1,1], full_output=True )

	print('Factors with minimal current:', str(roots[0] ) )
	print('Minimal current: ', roots[1] )

	for indices,el in np.ndenumerate(I):
		factors	= [X[indices], Y[indices] ]
		I[indices]	= current_abs_value(factors)

	return I, roots

def abs_scan_and_plot(fig, ax, X, Y, phases, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas=[]):
	I, roots	= abs_scan(X, Y, phases, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas=[])
	c		= ax.contourf(X, Y, I)
	cbar		= fig.colorbar(c, ax=ax)
	ax.scatter(roots[0][0], roots[0][1], marker='x', color='r')
	ax.set_xlabel(r'$t_1$')
	ax.set_ylabel(r'$t_3$')

	return I, roots

def current(phases, factors, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas=[]):
	phi_1	= phases[0] + phases[1]
	phi_3	= phases[0] - phases[1]
	tb1	= np.exp(1j*phi_1 )*t*factors[0]
	tb2	= t
	tb3	= np.exp(1j*phi_3 )*t*factors[1]
	tt4	= t

	if len(thetas) == 4:
		tb11	= tb1*thetas[0]
		tb21	= tb2*thetas[1]
		tb31	= tb3*thetas[2]
		tt41	= tt4*thetas[3]

	if model == 1:
		maj_op, overlaps, par	= box.majorana_leads(tb1, tb2, tb3, tt4)
	else:
		maj_op, overlaps, par	= box.abs_leads(tb1, tb11, tb2, tb21, tb3, tb31, tt4, tt41)

	maj_box.change(majoranas = maj_op)
	tunnel		= maj_box.constr_tunnel()

	sys		= qmeq.Builder_many_body(Ea=Ea, Na=par, Tba=tunnel, dband=dband, mulst=mu_lst, tlst=T_lst, kerntype=method, itype=1)
	sys.solve(qdq=False, rotateq=False)

	return sys.current[0]


