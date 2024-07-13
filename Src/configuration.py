import os as os


class configuration:

	## Choose one model from the list at a time
	# ['UNet', 'GCN_UNet', 'SegNet', 'UNet2Plus', 'UNet3Plus', 'AAUNet']
	model_name = 'UNet3Plus' 
	
	## Dataset name 
	dataset_name = 'BUSI_Dataset'
	
	## GPU device
	device = 'cuda:0'

	#Number of GPU to use
	num_gpus = 2
	
	## No of folds required for cross fold validation
	k_fold = 4

	## No of epochs choosen to train the model
	num_epochs = 100

	## Path to store model weights
	model_path = ''

	## Path to store sample results for visual inspection
	map_path = ' ' 
	def __init__(self):

		##Directory in which the models will be saved for training and testing will be stored; will be created by the program.
		self.model_path = os.path.join('four_cross_fold',self.dataset_name, 'models_saving')
		
		## Directory in which the segmentation maps will be saved for visual inspection 
		self.map_path = os.path.join('four_cross_fold',self.dataset_name, 'Segmentation_map')
	
	