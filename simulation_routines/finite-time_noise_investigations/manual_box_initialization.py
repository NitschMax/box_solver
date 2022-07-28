import fock_class as fc
import time_evolution as te
import box_class as bc

### Routine to control correct function of transport_setup_class, otherwise not needed for initialization
def majorana_noise_box(tc, energies):
	overlaps	= np.diag(energies, k=1)
	maj_op		= [fc.maj_operator(index=0, lead=[0], coupling=[tc[0]]), fc.maj_operator(index=1, lead=[0,1], coupling=[tc[1],tc[2]]), \
					fc.maj_operator(index=2, lead=[1], coupling=[tc[3]]), fc.maj_operator(index=3, lead=[2], coupling=[tc[4]]) ]
	par		= np.array([0,0,1,1])
	return maj_op, overlaps, par

def box_preparation(t, t_u, phases, factors, theta_phases, tunnel_mult, eps12, eps23, eps34, eps, Vg, model):
	tb1, tb2, tb3, tt4, tb11, tb21, tb31, tt41	= te.tunnel_coupl(t, t_u, phases, factors, theta_phases, tunnel_mult)
	maj_op, overlaps, par	= te.box_definition(model, tb1, tb2, tb3, tt4, tb11, tb21, tb31, tt41, eps12, eps23, eps34, eps)

	maj_box		= bc.majorana_box(maj_op, overlaps, Vg, 'asymmetric_box')
	maj_box.diagonalize()
	Ea		= maj_box.elec_en
	tunnel		= maj_box.constr_tunnel()

	return Ea, tunnel, par

def lead_connections(model, t, t_u, phases, factors, theta_phases, tunnel_mult, lead=0):
	tb1, tb2, tb3, tt4, tb11, tb21, tb31, tt41	= te.tunnel_coupl(t, t_u, phases, factors, theta_phases, tunnel_mult)
	if model == 1:
		if lead	== 0:
			tunnel_amplitudes	= [tb1, tb2, tb3]
		elif lead == 1:
			tunnel_amplitudes	= [tt4]
	elif model == 2:
		if lead	== 0:
			tunnel_amplitudes	= [tb1, tb2, tb3, tb11, tb21, tb31]
		elif lead == 1:
			tunnel_amplitudes	= [tt4, tt41]
	elif model == 3:
		if lead	== 0:
			tunnel_amplitudes	= [tb1, tb2, tb3, tb11, tb21]
		elif lead == 1:
			tunnel_amplitudes	= [tt4]
	return np.array(tunnel_amplitudes )

def abs_block():
	num_occ		= 16
	dof		= 128
	rho0		= np.zeros(dof )
	rho0[0]		= 1
	return rho0

def box_definition(model, tb1, tb2, tb3, tt4, tb11, tb21, tb31, tt41, eps12, eps23, eps34, eps):
	if model == 1:
		maj_op, overlaps, par	= abox.majorana_leads(tb1, tb2, tb3, tt4, eps12, eps23, eps34)
	elif model == 2:
		maj_op, overlaps, par	= abox.abs_leads(tb1, tb11, tb2, tb21, tb3, tb31, tt4, tt41, eps)
	elif model == 3:
		maj_op, overlaps, par	= abox.six_maj(tb1, tb2, tb3, tt4, eps12, eps23, eps34, eps, tb11=tb11, tb21=tb21)
	
	return maj_op, overlaps, par

def tunnel_coupl(t, t_u, phases, factors, theta_phases, tunnel_mult):
	tb1	= t*phases[0]*factors[0]
	tb2     = t*phases[1]*factors[1]
	tb3     = t*phases[2]*factors[2]
	tt4	= t_u*phases[3]*factors[3]

	tb11	= tb1*theta_phases[0]*tunnel_mult[0]
	tb21	= tb2*theta_phases[1]*tunnel_mult[1]
	tb31	= tb3*theta_phases[2]*tunnel_mult[2]
	tt41	= tt4*theta_phases[3]*tunnel_mult[3]

	return tb1, tb2, tb3, tt4, tb11, tb21, tb31, tt41



