import itertools
import json
import numpy as np
# import argparse
import os
import numpy as np
import random
import copy


def search_lr_and_decay():
	'''
	Search the hyperparameter space of learning rate, weight decay,
	batch size, and droput rate.
	'''

	# RANDOMIZE?
	# n_trials = 30
	# lrate = 10**np.random.uniform(-5,-2,size=n_trials)
	# decay = 10**np.random.uniform(-4,-2,size=n_trials)

	#GRID
	learning_rates = np.logspace(-5, -2, num=4)  # [1e-5, 1e-4, 1e-3, 1e-2]
	decays         = np.logspace(-4, -2, num=3)  # [1e-4, 1e-3, 1e-2]
	batches        = [32]
	channels       = [16,32,64]
	losses         = ["ce","ce_dl"]

	# Define search space
	combinations = list(itertools.product(learning_rates,decays,batches,channels,losses))

	# Define rows
	rows = []
	for i,(lr,wd,batch,ch,loss) in enumerate(combinations):
		sample = {
			'id':i,
			'model':"UNet",
			'seed':476,
			'epochs':55,
			'loss':loss,
			'bands':4,
			'labels':2,
			'lrate':round(lr,5),
			'decay':round(wd,5),
			'batch':int(batch),					
			'vit_layers':2,
			'mlp_ratio':5, 
			'cnn_layers':3, 
			'channels':ch,
			'w0':0.5,
			'w1':0.5,
			'w2':0.0
		}

		rows.append(sample)

	# Save to JSON
	write_hp_file("hpo_1",rows)


def search_losses():
	'''
	Train several parameter choices for loss functions.
	'''
	# Base

	# Region-pixel

	# Hard-pixel

	# Added distance (2 losses)?
	pass


##### BEST MODELS #####
def search_dtm_penalty():
	'''
	Search the space for the weight w3 of the distance map loss.
	Uses best result from search_losses().
	'''
	pass


##### ABLATION #####
def band_ablation():
	'''
	Train a 3-band version to see benefit of RGB+NIR vs RGB-only.
	'''

	pass


def three_classes():
	'''
	Train the best setup on a 3-class problem to compare to binary mask results.
	'''
	pass


##### HELPER FUNC #####
def write_hp_file(name,rows):
	# WRITE JSON FILE
	out_file_path = f"./{name}.json"		
	with open(out_file_path,'w') as fp:
		for line in rows:
			json.dump(line,fp)
			fp.write('\n')
	print(f"Parameter file written to {out_file_path}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


##### MAIN #####
if __name__ == '__main__':
	set_seed(476)
	search_lr_and_decay()