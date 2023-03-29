import os
import numpy as np

def base_dir():
	base_dir = "/Users/ma0274ni/Documents/projects/majorana_box/data/fusion_rule"
	return base_dir

def save_data(data, params, data_format=0):
	file_path = data_dir(params, data_format)
	np.save(file_path, data)
	print('Data saved in: ', file_path)

def load_data(params, data_format=0):
	file_path = data_dir(params, data_format)
	try:
		data = np.load(file_path)
		print('Data loaded from: ', file_path)
	except FileNotFoundError:
		print(f'File {file_path} not found!')
		print('Calculation neccesary.')
		data = None
	return data

def data_dir(params, data_format):
	format_dir = "format_{}".format(data_format)
	
	# Define the base directory where the data will be saved
	base_dir = "/Users/ma0274ni/Documents/projects/majorana_box/data/fusion_rule"
	cwd = os.getcwd()

	# Create a directory with a specific name based on the parameters
	dir_path = os.path.join(base_dir, format_dir)
	os.makedirs(dir_path, exist_ok=True)
	
	# Save the data to a file with a specific name based on the parameters
	file_name = "data_{}.npy".format("_".join(["{}-{}".format(k, v) for k, v in params.items()]))
	file_path = os.path.join(dir_path, file_name)
	return file_path
