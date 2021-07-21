import os
import numpy as np

def dir(phases, factors, maj_box, t, Ea, dband, mu_lst, T_lst, method, model, thetas):
	if model == 1:
		thetas = []
	
	dirName		= '/Users/ma0274ni/Documents/projects/majorana_box/data/' + maj_box.name + '/'
	if not os.path.isdir(dirName ):
		print('Directory not existing!', dirName)
		return 0

	fileName	= ''

	for i,phase in enumerate(phases):
		fileName	+= 'phase{}={:1.2f}xpi_'.format(i+1, phase/np.pi)

	for i,factor in enumerate(factors ):
		fileName	+= 't{}={:1.2f}_'.format(i+1, factor)

	if len(thetas) != 0:
		print(thetas)
		for i,theta in enumerate(thetas):
			fileName	+= 'theta{}={:1.2f}xpi_'.format(i+1, theta/np.pi)

	fileName	= fileName[:-1]
		

	print(os.getcwd() )
	print(dirName, fileName)
	return dirName, fileName

