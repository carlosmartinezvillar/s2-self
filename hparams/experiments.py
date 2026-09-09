import itertools
import json
import numpy as np
import os
import numpy as np
import random
import copy

################################################################################
# HYPERPARAMETER SEARCH
################################################################################
def search_lr_and_decay():
	'''
	Saves to 'hpo_1.json'.
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
	Saves to 'hpo_2.json'
	Train several parameter choices for loss functions.
	'''

	# FIXED PARAMETERS FROM 'hpo_1'
	lrate = 0.0001
	decay = 0.001
	batch = 32
	channels = 64

	rows = []

	'''
	# Base/individual losses
	2.1 Candidates (individual)
	--------------
	CE    -- pixel (baseline)
	CW    -- pixel, class-imbalance
	Focal -- pixel, hard-examples
	Dice  -- region
	EW    -- boundary (baseline?)	
	'''
	individual_losses = ["ce","cw","fl","dl","bl"]
	focal_gammas = [1.0,2.0,3.0]
	row_counter = 0

	for loss in individual_losses:
		sample = {
			'id':row_counter,
			'model':"UNet",
			'seed':476,
			'epochs':55,
			'loss':loss,
			'bands':4,
			'labels':2,
			'lrate':lrate,
			'decay':decay,
			'batch':int(batch),					
			'vit_layers':2,
			'mlp_ratio':5, 
			'cnn_layers':3, 
			'channels':channels,
			'w0':1.0,
			'w1':0.0,
			'w2':0.0,
			'focal_gamma':0.0
		}

		if loss == 'fl':
			for j,g in enumerate(focal_gammas):
				sample['focal_gamma'] = g
				sample['id'] = row_counter + j
				rows.append(copy.deepcopy(sample))
			row_counter += len(focal_gammas)
			continue

		rows.append(sample)
		row_counter += 1

	'''
	2.2 Pixel + Region
	------------------
	CE + Dice
	CW + Dice    -- imbalance prior
	Focal + Dice -- hard-pixels

	# SWEEP: CE + DICE
	# {w0:1.0,w1:0.0,w2:0.0}
	# {w0:0.0,w1:1.0,w2:0.0}
	# {w0:0.5,w1:0.5,w2:0.0}	
	# {w0:0.8,w1:0.2,w2:0.0}
	# {w0:0.2,w1:0.8,w2:0.0}

	# SWEEP: FOCAL + DICE
	# {w0:0.8,w1:0.2,w2:0.0} x gamma=[1.0,2.0,3.0]
	# {w0:0.2,w1:0.8,w2:0.0}
	# {w0:0.5,w1:0.5,w2:0.0}

	'''
	pixel_region_losses = ["ce_dl","cw_dl","fl_dl"]
	weight_sweep = [(1.0,0.0),(0.0,1.0),(0.5,0.5),(0.8,0.2),(0.2,0.8)]
	product = list(itertools.product(weight_sweep,pixel_region_losses))
	row_counter = len(rows)

	for i,(weights,loss) in enumerate(product):
		sample = {
			'id':row_counter,
			'model':"UNet",
			'seed':476,
			'epochs':55,
			'loss':loss,
			'bands':4,
			'labels':2,
			'lrate':lrate,
			'decay':decay,
			'batch':int(batch),					
			'vit_layers':2,
			'mlp_ratio':5, 
			'cnn_layers':3, 
			'channels':channels,
			'w0':weights[0],
			'w1':weights[1],
			'w2':0.0,
			'focal_gamma':0.0
		}

		if loss == 'fl_dl':
			for j,g in enumerate(focal_gammas):
				sample['focal_gamma'] = g
				sample['id'] = row_counter+j
				rows.append(copy.deepcopy(sample))
			row_counter += len(focal_gammas)
			continue

		rows.append(sample)
		row_counter += 1

	# print("\n'hpo_2': Hyperparameter search for best loss function configurations.")
	# print('-'*100)
	# print('\n'.join([str(r) for r in rows]))
	write_hp_file("hpo_2",rows)


def search_distance_penalty():
	'''
	Writes 'hpo_3.json'.
	Search the space for the penalty weight (i.e. w_3) of the distance map loss.
	Uses best result from search_losses().
	'''


	'''
	2. Combined -- Explicit boundary added
	------------
	EW + Dice
	CE + EW
	CW + EW
	EW + Focal -- harder examples, could be best?
	'''
	# FIXED PARAMETERS FROM 'hpo_1'
	lrate = 0.0001
	decay = 0.001
	batch = 32
	channels = 64
	gamma = 1.0

	boundary_loss_weights   = [0.05,0.1,0.2,0.3,0.5]
	boundary_loss_functions = ["ce_bl","cw_bl","fl_bl","dl_bl"] #with params found by hpo2
	product = list(itertools.product(boundary_loss_weights,boundary_loss_functions))

	rows = []

	for i,(b_weight,loss) in enumerate(product):

		remaining_weight = 1.0 - b_weight

		sample = {
			'id':i,
			'model':"UNet",
			'seed':476,
			'epochs':55,
			'loss':loss,
			'bands':4,
			'labels':2,
			'lrate':lrate,
			'decay':decay,
			'batch':int(batch),					
			'vit_layers':2,
			'mlp_ratio':5, 
			'cnn_layers':3, 
			'channels':channels,
			'w0':remaining_weight,
			'w1':b_weight,
			'w2':0.0,
			'focal_gamma':0.0
		}

		if loss == 'fl_bl':
			sample['focal_gamma'] = gamma

		rows.append(sample)

	'''
	3.2 Combined
	------------
	L_3 = L_px + L_region + L_boundary (KEEP CONVEX! w1+w2+w3=1.0)
	w_1 + w_2 + w_3 = 1 (convex)

	CE + Dice + EW
	CW + Dice + EW
	Focal + Dice + EW
	'''
	current_row = len(rows)
	combined_losses = ["ce_dl_bl","cw_dl_bl","fl_dl_bl"]
	best_weighing   = [(0.5,0.5),(0.5,0.5),(0.5,0.5)] #from 'hpo_2' (2.2 above) 
	product = list(itertools.product(boundary_loss_weights,combined_losses))

	
	for (b_weight,loss) in product:

		# Adjust prior weights
		w0 = best_weighing[combined_losses.index(loss)][0]
		w1 = best_weighing[combined_losses.index(loss)][1]
		new_w0 = (1.0 - b_weight)*w0
		new_w1 = (1.0 - b_weight)*w1


		sample = {
			'id':current_row,
			'model':"UNet",
			'seed':476,
			'epochs':55,
			'loss':loss,
			'bands':4,
			'labels':2,
			'lrate':lrate,
			'decay':decay,
			'batch':int(batch),					
			'vit_layers':2,
			'mlp_ratio':5, 
			'cnn_layers':3, 
			'channels':channels,
			'w0':new_w0,
			'w1':new_w1,
			'w2':b_weight,
			'focal_gamma':0.0
		}		

		if loss == 'fl_dl_bl':
			sample['focal_gamma'] = gamma

		# write
		rows.append(sample)
		current_row += 1

	write_hp_file('hpo_3',rows)


################################################################################
# FINAL MODELS
################################################################################
def best_models():
	'''
	Sets an analogous .json to steps above to record best chosen.
	'''
	pass

################################################################################
# ABLATION
################################################################################
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

################################################################################
# HELPER/OTHER FUNCTIONS
################################################################################
def write_hp_file(name,rows):
	# WRITE JSON FILE
	out_file_path = f"./{name}.json"		
	with open(out_file_path,'w') as fp:
		for line in rows[0:-1]:
			json.dump(line,fp)
			fp.write('\n')
		json.dump(rows[-1],fp)
	print(f"Parameter file written to {out_file_path}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


################################################################################
# MAIN
################################################################################
if __name__ == '__main__':
	set_seed(476)
	search_lr_and_decay()
	search_losses()
	search_distance_penalty()