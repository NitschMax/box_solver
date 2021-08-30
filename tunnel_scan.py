import numpy as np
import scipy.optimize as opt
import asym_box as box
import qmeq
import matplotlib.pyplot as plt
import data_directory as dd
import os
from joblib import Parallel, delayed

from matplotlib import ticker, cm

def phase_zero_scan(X, Y, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult, recalculate, num_cores, save_result):
	prefix		= 'phase-zero-scan/x-{:1.2f}xpi-{:1.2f}xpi-{}_y-{:1.2f}xpi-{:1.2f}xpi-{}'.format(X[0,0]/np.pi, X[-1,-1]/np.pi, len(X[0] ), Y[0,0]/np.pi, Y[-1,-1]/np.pi, len(Y[:,0] ) )

	file	= dd.dir(maj_box, t, Ea, dband, mu_lst, T_lst, method, model, phases=[], factors=[], thetas=thetas, tunnel_mult=tunnel_mult, prefix=prefix)
	file	= file[0] + file[1] + '.npy'

	if os.path.isfile(file ) and (not recalculate):
		print('Loading data.')
		X, Y, I	= np.load(file )
	else:
		print('Data not already calculated. Calculation ongoing')
		I	= np.zeros(X.shape, dtype=np.float64)
		
		unordered_res	= Parallel(n_jobs=num_cores)(delayed(factor_opt_min)(indices, X, Y, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult) for indices, bias in np.ndenumerate(X) ) 
		for el in unordered_res:
			I[el[0] ]	= el[1]

		if save_result:
			print('Result is saved in:', file)
			np.save(file, [X, Y, I] )
		else:
			print('Result was not saved!')
		print('Finished!')

	return X, Y, I

def factor_opt_min(indices, X, Y, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult):
	print('Calculation: ', indices[0]*len(X[0]) +indices[1], '/', X.size )
	result	= opt.fmin(factor_func, args=([X[indices], 0, Y[indices], 0], maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult), \
			x0=[1,1], full_output=True, maxiter=200 )[1]
	return [indices, result]

def factor_func(factors, phases, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult):
	return current(phases, [factors[0], 1, factors[1], 1], maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult)


def phase_zero_scan_and_plot(fig, ax, X, Y, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas=[], tunnel_mult=[1, 1, 1, 1], recalculate=False, num_cores=3, save_result=True, logscale=False):
	X,Y,I	= phase_zero_scan(X, Y, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult, recalculate, num_cores, save_result)

	if logscale:
		c	= ax.contourf(X, Y, I, locator=ticker.LogLocator() )
	else:
		c	= ax.contourf(X, Y, I )

	cbar	= fig.colorbar(c, ax=ax)

	fs	= 15
	ax.set_xlabel(r'$\Phi_{avg}$', fontsize=fs)
	ax.set_ylabel(r'$\Phi_{diff}$', fontsize=fs)

	fs	= 13
	ax.locator_params(axis='both', nbins=5 )
	cbar.ax.locator_params(axis='y', nbins=5 )

	ax.tick_params(labelsize=fs)
	cbar.ax.set_title('mc', size=fs)
	cbar.ax.tick_params(labelsize=fs)

	ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
	ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func) )
	ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
	ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func) )

	return X, Y, I


def abs_zero_scan(X, Y, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult, recalculate, num_cores, save_result):
	prefix		= 'prefactor-zero-scan/x-{:1.1f}-{:1.1f}-{}_y-{:1.1f}-{:1.1f}-{}'.format(X[0,0], X[-1,-1], len(X[0] ), Y[0,0], Y[-1,-1], len(Y[:,0] ) )

	file	= dd.dir(maj_box, t, Ea, dband, mu_lst, T_lst, method, model, phases=[], factors=[], thetas=thetas, tunnel_mult=tunnel_mult, prefix=prefix)
	file	= file[0] + file[1] + '.npy'
	print(file)

	if os.path.isfile(file ) and (not recalculate):
		print('Loading data.')
		X, Y, I	= np.load(file )
	else:
		print('Data not already calculated. Calculation ongoing')
		I	= np.zeros(X.shape, dtype=np.float64)

		unordered_res	= Parallel(n_jobs=num_cores)(delayed(phase_opt_min)(indices, X, Y, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult) for indices, t1 in np.ndenumerate(X) )

		for el in unordered_res:
			I[el[0] ]	= el[1]

		if save_result:
			print('Result is saved in:', file)
			np.save(file, [X, Y, I] )
		else:
			print('Result was not saved!')
		print('Finished!')

	return X, Y, I

def phase_opt_min(indices, X, Y, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult):
	print('Calculation: ', indices[0]*len(X[0]) +indices[1], '/', X.size )
	result	= opt.fmin(phase_func, x0=[np.pi/4,np.pi/4], full_output=True, maxiter=200, args=([X[indices], 1, Y[indices], 1], maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult) )[1]
	return [indices, result]

def phase_func(phases, factors, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult):
	return current([phases[0], 1, phases[1], 1], factors, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult)

def abs_zero_scan_and_plot(fig, ax, X, Y, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas=[], tunnel_mult=[1, 1, 1, 1], recalculate=False, num_cores=3, save_result=True, logscale=False):
	X,Y,I	= abs_zero_scan(X, Y, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult, recalculate, num_cores, save_result)

	fs	= 15
	ax.set_xlabel(r'$t_1 \, [t]$', fontsize=fs)
	ax.set_ylabel(r'$t_3 \, [t]$', fontsize=fs)

	fs	= 13
	if logscale:
		c	= ax.contourf(X, Y, I, locator=ticker.LogLocator() )
	else:
		c	= ax.contourf(X, Y, I )

	cbar	= fig.colorbar(c, ax=ax)
	ax.locator_params(axis='both', nbins=5 )
	cbar.ax.locator_params(axis='y', nbins=5 )

	ax.tick_params(labelsize=fs)
	cbar.ax.set_title('mc', size=fs)
	cbar.ax.tick_params(labelsize=fs)

	return X, Y, I

def phase_scan(X, Y, factors, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult, recalculate, num_cores):
	print('Trying to find the roots.')
	current_phases	= lambda phases: current([phases[0], 0, phases[1], 0], factors, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas=thetas, tunnel_mult=tunnel_mult)
	roots	= opt.fmin(current_phases, x0=[np.pi/4,np.pi/4], full_output=True )
	print('Phase-diff with minimal current:', 'pi*'+str(roots[0]/np.pi) )
	print('Minimal current: ', roots[1] )

	minC	= np.mod(roots[0], np.pi )-np.pi/2 
	diff	= current_phases(minC) - roots[1]
	if np.abs(diff) > 1e-16:
		print('Result for minimum not Pi-periodic! Difference:', diff)

	prefix	= 'phase-scan/x-{:1.2f}xpi-{:1.2f}xpi-{}_y-{:1.2f}xpi-{:1.2f}xpi-{}_'.format(X[0,0]/np.pi, X[-1,-1]/np.pi, len(X[0] ), Y[0,0]/np.pi, Y[-1,-1]/np.pi, len(Y[:,0] ) )
	file	= dd.dir(maj_box, t, Ea, dband, mu_lst, T_lst, method, model, phases=[], factors=factors, thetas=thetas, tunnel_mult=tunnel_mult, prefix=prefix)
	file	= file[0] + file[1] + '.npy'

	if os.path.isfile(file ) and (not recalculate):
		print('Loading data.')
		X, Y, I	= np.load(file )
	else:
		print('Data not already calculated. Calculation ongoing')
		I	= np.zeros(X.shape, dtype=np.float64 )

		unordered_res	= Parallel(n_jobs=num_cores)(delayed(current_with_ind)(indices, [X[indices], 0, Y[indices], 0], factors, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult) for indices, var in np.ndenumerate(X) )
		for el in unordered_res:
			I[el[0] ]	= el[1]

		np.save(file, [X, Y, I] )
		print('Finished!')

	return I, roots

def current_with_ind(indices, phases, factors, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult):
	return [indices, current(phases, factors, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult) ]

def phase_scan_and_plot(fig, ax, X, Y, factors, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas=[], tunnel_mult=[1, 1, 1, 1], recalculate=False, num_cores=6):
	I, roots	= phase_scan(X, Y, factors, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult, recalculate, num_cores)

	c	= ax.contourf(X, Y, I)
	cbar	= fig.colorbar(c, ax=ax)
	
	minC	= np.mod(roots[0], np.pi )-np.pi/2 
	ax.scatter(minC[0], minC[1], marker='x', color='r')

	fs	= 15
	ax.set_xlabel(r'$\Phi_{avg}$', fontsize=fs)
	ax.set_ylabel(r'$\Phi_{diff}$', fontsize=fs)

	fs	= 13
	ax.locator_params(axis='both', nbins=5 )
	cbar.ax.locator_params(axis='y', nbins=7 )
	
	ax.tick_params(labelsize=fs)

	cbar.ax.set_title('current', size=fs)
	cbar.ax.tick_params(labelsize=fs)
	ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
	ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func) )
	ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
	ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func) )

	return I, roots

def abs_scan(X, Y, phases, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult, recalculate, num_cores):
	current_abs_value	= lambda factors: current(phases, [factors[0], 1, factors[1], 1], maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas=thetas, tunnel_mult=tunnel_mult)
	roots	= opt.fmin(current_abs_value, x0=[1,1], full_output=True )

	print('Factors with minimal current:', str(roots[0] ) )
	print('Minimal current: ', roots[1] )

	prefix	= 'prefactor-scan/x-{:1.1f}-{:1.1f}-{}_y-{:1.1f}-{:1.1f}-{}_'.format(X[0,0]/np.pi, X[-1,-1]/np.pi, len(X[0] ), Y[0,0]/np.pi, Y[-1,-1]/np.pi, len(Y[:,0] ) )

	file	= dd.dir(maj_box, t, Ea, dband, mu_lst, T_lst, method, model, phases=phases, factors=[], thetas=thetas, tunnel_mult=tunnel_mult, prefix=prefix)
	file	= file[0] + file[1] + '.npy'

	if os.path.isfile(file ) and (not recalculate):
		print('Loading data.')
		X, Y, I	= np.load(file )
	else:
		print('Data not already calculated. Calculation ongoing')
		I	= np.zeros(X.shape, dtype=np.float64 )

		unordered_res	= Parallel(n_jobs=num_cores)(delayed(current_with_ind)(indices, phases, [X[indices], 1, Y[indices], 1], maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult) for indices, var in np.ndenumerate(X) )
		for el in unordered_res:
			I[el[0] ]	= el[1]

		np.save(file, [X, Y, I] )
		print('Finished!')

	return I, roots

def abs_scan_and_plot(fig, ax, X, Y, phases, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas=[], tunnel_mult=[1, 1, 1, 1], recalculate=False, num_cores=6):
	I, roots	= abs_scan(X, Y, phases, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas, tunnel_mult, recalculate, num_cores)
	c		= ax.contourf(X, Y, I)
	cbar		= fig.colorbar(c, ax=ax)

	xmin	= roots[0][0]
	ymin	= roots[0][1]
	if xmin < 4 and ymin < 4:
		ax.scatter(roots[0][0], roots[0][1], marker='x', color='r')
	else:
		print('Minimum result outside considered range!')
	fs	= 15
	ax.set_xlabel(r'$\frac{t_1}{t}$', fontsize=fs)
	ax.set_ylabel(r'$\frac{t_3}{t}$', fontsize=fs)

	fs		= 13
	cbar.ax.set_title('current', size=fs)
	cbar.ax.tick_params(labelsize=fs)
	ax.locator_params(axis='both', nbins=5 )
	cbar.ax.locator_params(axis='y', nbins=7 )
	ax.tick_params(labelsize=fs)

	return I, roots

def current(phases, factors, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas=[], tunnel_mult=[1, 1, 1, 1]):
	phi_1	= phases[0] + phases[2]
	phi_3	= phases[0] - phases[2]

	tb1	= t*factors[0]*np.exp(1j*phi_1 )
	tb2	= t*factors[1]
	tb3	= t*factors[2]*np.exp(1j*phi_3 )
	tt4	= t*factors[3]

	if len(thetas) == 0:
		theta_phases	= np.ones(4)
	else:
		theta_phases	= np.exp( 1j*thetas )

	tb11	= tb1*theta_phases[0]*tunnel_mult[0]
	tb21	= tb2*theta_phases[1]*tunnel_mult[1]
	tb31	= tb3*theta_phases[2]*tunnel_mult[2]
	tt41	= tt4*theta_phases[3]*tunnel_mult[3]

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



