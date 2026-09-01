import itertools
import json
import numpy as np
# import argparse
import os
import numpy as np
import random
import copy

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def search_lr_and_decay():
	'''
	Search the hyperparameter space of learning rate, weight decay,
	batch size, and droput rate.
	'''
	learning_rates = np.logspace(-5, -2, num=4)  # [1e-5, 1e-4, 1e-3, 1e-2]
	decays         = np.logspace(-4, -2, num=3)  # [1e-4, 1e-3, 1e-2]
	batches        = [16,32,64]
	channels       = [32,64]
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
			'bands':3,
			'labels':2,
			'lrate':round(lr,5),
			'decay':round(wd,5),
			'batch':int(batch),					
			'vit_layers':2, #base
			'mlp_ratio':5, #base
			'cnn_layers':3, #base
			'channels':ch   #base
		}

		rows.append(sample)

	# Save to JSON
	write_hp_file("hpo_1",rows)


def benchmark_losses():
	'''
	Train best parameter choice for all loss functions across multiple seeds.
	'''
	pass


def write_hp_file(name,rows):
	# WRITE JSON FILE
	out_file_path = f"./{name}.json"		
	with open(out_file_path,'w') as fp:
		for line in rows:
			json.dump(line,fp)
			fp.write('\n')
	print(f"Parameter file written to {out_file_path}")


if __name__ == '__main__':
	set_seed(476)
	search_lr_and_decay()