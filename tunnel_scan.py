import numpy as np
import scipy.optimize as opt
import asym_box as box
import qmeq
import matplotlib.pyplot as plt

def phase_scan(X, Y, factors, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas=[]):
	print('Trying to find the roots.')
	roots	= opt.fmin(current, x0=[np.pi/4, np.pi/4], args=(factors, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas), full_output=True )
	print('Phase-diff with minimal current:', 'pi*'+str(roots[0]/np.pi) )
	print('Minimal current: ', roots[1] )

	I	= np.zeros(X.shape, dtype=np.float64 )

	for indices,el in np.ndenumerate(I):
		I[indices ]	= current([X[indices], Y[indices] ], factors, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas) 
	return I, roots

def phase_scan_and_plot(fig, ax, X, Y, factors, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas=[]):
	I, roots	= phase_scan(X, Y, factors, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas=[])

	ax.scatter(roots[0][0], roots[0][1], marker='x', color='r')
	c	= ax.contourf(X, Y, I)
	cbar	= fig.colorbar(c, ax=ax)

	fs	= 12
	ax.locator_params(axis='both', nbins=5 )
	cbar.ax.locator_params(axis='y', nbins=7 )
	
	ax.tick_params(labelsize=fs)

	cbar.ax.set_title('current', size=fs)
	cbar.ax.tick_params(labelsize=fs)
	ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
	ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func) )
	ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
	ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func) )
	ax.set_xlabel(r'$\Phi_{avg}$', fontsize=fs)
	ax.set_ylabel(r'$\Phi_{diff}$', fontsize=fs)

	return I, roots

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
	fs		= 12

	ax.scatter(roots[0][0], roots[0][1], marker='x', color='r')
	ax.set_xlabel(r'$t_1$', fontsize=fs)
	ax.set_ylabel(r'$t_3$', fontsize=fs)

	cbar.ax.set_title('current', size=fs)
	cbar.ax.tick_params(labelsize=fs)
	ax.locator_params(axis='both', nbins=5 )
	cbar.ax.locator_params(axis='y', nbins=7 )
	ax.tick_params(labelsize=fs)

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

def format_func(value, tick_number):
    # find number of multiples of pi/2
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N == -1:
        return r"$-\pi/2$"
    elif N == -2:
        return r"$-\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)



