import numpy as np

def tunnel_from_gamma(gamma):
	return np.sqrt(gamma/(2*np.pi))+0.j

def measure_of_vector(block_state, state):
	return np.sqrt(np.sum(np.abs(block_state - state)**2) )
