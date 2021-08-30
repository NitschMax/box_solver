import os
import numpy as np

def dir(maj_box, t, Ea, dband, mu_lst, T_lst, method, model, phases=[], factors=[], thetas=[], tunnel_mult=[], prefix=''):
	if model == 1:
		thetas		= []
		tunnel_mult	= []
	
	dirName		= '/Users/ma0274ni/Documents/projects/majorana_box/data/' + maj_box.name + '/'
	if not os.path.isdir(dirName ):
		print('Directory not existing!', dirName)
		return 0

	fileName	= prefix

	for i,phase in enumerate(phases):
		fileName	+= 'phase{}={:1.2f}xpi_'.format(i+1, phase/np.pi)

	for i,factor in enumerate(factors ):
		fileName	+= 't{}={:1.2f}_'.format(i+1, factor)

	for i,theta in enumerate(thetas):
		fileName	+= '_theta{}={:1.2f}xpi'.format(i+1, theta/np.pi)

	if not tunnel_mult == [1, 1, 1, 1]:
		for i,tunnel_mult in enumerate(tunnel_mult):
			fileName	+= '_t2-{}={:1.2f}'.format(i+1, tunnel_mult)

	return dirName, fileName

