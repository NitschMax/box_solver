import numpy as np

def tunnel_from_gamma(gamma):
    return np.sqrt(gamma/(2*np.pi))+0.j

def gamma_from_tunnel(tunnel_amp):
    return 2*np.pi*tunnel_amp**2

def measure_of_vector(block_state, state):
    return np.sqrt(np.sum(np.abs(block_state - state)**2) )
