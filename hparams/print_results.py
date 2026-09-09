import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import itertools


def parse_args():
	# ARGV
	parser = argparse.ArgumentParser()
	required = parser.add_argument_group('Required arguments')
	required.add_argument('--log-dir',required=True,help='Training logs.')

	# LOAD 
	args = parser.parse_args()

	# CHECK HERE
	assert os.path.isdir(args.log_dir), f"No path found for log dir {args.log_dir}"
	return args	


def load_train_log(log_path):
	# OPEN/READ
	with open(log_path,'r') as fp:
		lines = fp.readlines()

	assert len(lines) > 1, f"Found {len(lines)} in {log_path}"

	header = lines[0].rstrip('\n').split('\t')
	epochs = np.array([l.rstrip('\n').split('\t') for l in lines[1:]]).astype(float)

	# RETURN
	return header, epochs


def get_model_best_epoch(log_path): #<< ADD DICE 
	'''
	Read a log file. 1st line is header, and each subsequent line an epoch.
	Return a dictionary with tuple values. Tuples are set as best metric value 
	and its corresponding epoch.
	'''

	# FILE EXISTS
	assert os.path.isfile(log_path), f"No log file found at {log_path}"

	# GET ID
	model_id = log_path.rstrip('.tsv').split('_')[-1]

	# LOAD
	header,epochs = load_train_log(log_path)

	# GET VALIDATION COLUMNS
	iou_idx = header.index('viou1')
	acc_idx = header.index('vacc1')
	tpr_idx = header.index('vtpr1')
	ppv_idx = header.index('vppv1')
	dic_idx = header.index('vdic1')

	# GET MAX VALUE & MAX INDEX
	best_iou = np.max(epochs[:,iou_idx])
	best_acc = np.max(epochs[:,acc_idx])
	best_tpr = np.max(epochs[:,tpr_idx])
	best_ppv = np.max(epochs[:,ppv_idx])
	best_dic = np.max(epochs[:,dic_idx])
	best_iou_epoch = np.argmax(epochs[:,iou_idx])
	best_acc_epoch = np.argmax(epochs[:,acc_idx])
	best_tpr_epoch = np.argmax(epochs[:,tpr_idx])
	best_ppv_epoch = np.argmax(epochs[:,ppv_idx])
	best_dic_epoch = np.argmax(epochs[:,dic_idx])

	best = {
		'id': model_id,
		'iou':(best_iou,best_iou_epoch),
		'acc':(best_acc,best_acc_epoch),
		'tpr':(best_tpr,best_tpr_epoch),
		'ppv':(best_ppv,best_ppv_epoch),
		'dic':(best_dic,best_dic_epoch)
	}
	return best


def get_best_results_hpo_1(log_dir):

	# LOAD HYPERPARAMETERS FROM INITIAL SWEEP
	with open('./hpo_1.json','r') as fp:
		hp_list = [json.loads(line) for line in fp.readlines() if line != "\n"]
	indexed_hp_list = {row['id']:row for row in hp_list}

	# GET SCORES
	model_ids = [row['id'] for row in hp_list]
	scores    = []
	for experiment in model_ids:
		log_file = f"{log_dir}/hpo_1/epochs_{experiment:03}.tsv"
		scores.append(get_model_best_epoch(log_file))

	# METRICS TO CHOOSE
	ious = [s['iou'] for s in scores]
	dice = [s['dic'] for s in scores]

	# SORT
	sorted_ious = sorted(enumerate(ious),key=lambda x: x[1],reverse=True)
	sorted_dice = sorted(enumerate(dice),key=lambda x: x[1],reverse=True)
	sorted_ious_idxs = [_[0] for _ in sorted_ious]
	sorted_dice_idxs = [_[0] for _ in sorted_dice]

	# PRINT
	for i in sorted_ious_idxs:

		result  = scores[i]
		hparams = indexed_hp_list[int(result['id'])]

		line_buffer = []
		line_buffer.append(f"id: {result['id']:03}")
		line_buffer.append(f"iou: {result['iou'][0]:.5f} (ep. {result['iou'][1]:02})")
		line_buffer.append(f"dice: {result['dic'][0]:.5f} (ep. {result['dic'][1]:02})")
		line_buffer.append(f"lrate: {hparams['lrate']:<6}")
		line_buffer.append(f"decay: {hparams['decay']:<6}")
		line_buffer.append(f"channels: {hparams['channels']:<3}")
		line_buffer.append(f"loss: {hparams['loss']}")

		print(' | '.join(line_buffer))


def get_best_results_hpo_2(log_dir):

	# LOAD HYPERPARAMETERS FROM INITIAL SWEEP
	with open('./hpo_2.json','r') as fp:
		hp_list = [json.loads(line) for line in fp.readlines() if line != "\n"]
	indexed_hp_list = {row['id']:row for row in hp_list}
	pass


def get_best_results_hpo_3(log_dir):

	# LOAD HYPERPARAMETERS FROM INITIAL SWEEP
	with open('./hpo_3.json','r') as fp:
		hp_list = [json.loads(line) for line in fp.readlines() if line != "\n"]
	indexed_hp_list = {row['id']:row for row in hp_list}
	pass


def get_best_model_results(log_dir):
	# --------------------------------------------------
	# VALIDATION RESULTS
	# --------------------------------------------------
	pass

	# --------------------------------------------------
	# TEST RESULTS
	# --------------------------------------------------
	pass


def plot_training_log(log_path,best_iou_epoch=None):
	'''
	Plot full time series of per epoch training and validation results.
	Two plots: loss and metrics.
	'''

	# FILE EXISTS
	assert os.path.isfile(log_path), f"No log file found at {log_path}"

	# GET IDs
	model_id = log_path.rstrip('.tsv').split('_')[-1]
	stage_nr = log_path.split('/')[-2]

	# OPEN/READ
	header,epochs = load_train_log(log_path)

	# GET LOSS COLS
	tloss_idx = header.index('tloss')
	vloss_idx = header.index('vloss')
	tloss = epochs[:,tloss_idx]
	vloss = epochs[:,vloss_idx]

	# GET TRAIN METRIC COLS
	tiou1_idx = header.index('tiou1')
	tacc1_idx = header.index('tacc1')
	ttpr1_idx = header.index('ttpr1')
	tppv1_idx = header.index('tppv1')
	tiou1 = epochs[:,tiou1_idx]
	tacc1 = epochs[:,tacc1_idx]
	ttpr1 = epochs[:,ttpr1_idx]
	tppv1 = epochs[:,tppv1_idx]

	# GET VAL METRIC COLS
	viou1_idx = header.index('viou1')
	vacc1_idx = header.index('vacc1')
	vtpr1_idx = header.index('vtpr1')
	vppv1_idx = header.index('vppv1')
	viou1 = epochs[:,viou1_idx]
	vacc1 = epochs[:,vacc1_idx]
	vtpr1 = epochs[:,vtpr1_idx]
	vppv1 = epochs[:,vppv1_idx]

	####################
	# I. PLOT -- LOSS
	####################
	# SET
	fig = plt.figure(figsize=FIG_SIZE)
	ax  = fig.add_subplot(111)
	params = {'linewidth':1.0}
	ax.set_ylabel('Loss')
	ax.set_xlabel('Epoch')
	ax.set_title(f"Training & Validation Loss (Model {model_id})")

	# PLOT
	ax.plot(tloss,label='Training',linestyle='--',**params)
	ax.plot(vloss,label='Validation',linestyle='-',**params)

	if best_iou_epoch is not None:
		ax.axvline(x=best_iou_epoch, color='black', linestyle='--')

	# SAVE
	plt.legend()
	out_path_1 = f'../figures/{stage_nr}/loss_{model_id}.png'
	plt.savefig(out_path_1)
	plt.close()
	print(f"Plot written to {out_path_1}")

	####################
	# II. PLOT - METRICS
	####################
	# CONFIG
	fig = plt.figure(figsize=FIG_SIZE)
	ax  = fig.add_subplot(111)
	params = {'linewidth':1.0}
	# ax.set_ylim((0.0,1.0))
	ax.set_ylabel('Score')
	ax.set_xlabel('Epoch')
	ax.set_title("Training & Validation Metrics")

	# PLOT
	ax.plot(tacc1,label='Train acc',linestyle='-.',**params)
	ax.plot(tiou1,label='Train IoU',linestyle='-',**params)
	# ax.plot(ttpr1,label='Train tpr',linestyle='-.',**params)
	# ax.plot(tppv1,label='Train ppv',linestyle='-.',**params)
	ax.plot(vacc1,label='Valid acc',linestyle='-.',**params)
	ax.plot(viou1,label='Valid IoU',linestyle='-',**params)
	# ax.plot(vtpr1,label='Valid tpr',linestyle='-',**params)
	# ax.plot(vppv1,label='Valid ppv',linestyle='-',**params)

	if best_iou_epoch is not None:
		ax.axvline(x=best_iou_epoch, color='black', linestyle='--')

	# SAVE
	plt.legend()
	out_path_2 = f'../figures/{stage_nr}/metrics_{model_id}.png'
	plt.savefig(out_path_2)
	plt.close()
	print(f"Plot written to {out_path_2}")


def plot_lrate_vs_decay(model_str,lrates,decays,scores):
	'''
	For 'hpo_1'
	Scatter plot with decay and learning rates.
	'''

	# easier type
	lrates = np.array(lrates)
	decays = np.array(decays)
	scores = np.array(scores)


	# Indices of the top 5 scores
	top5_idx = np.argsort(scores)[-5:]
	mask = np.zeros(len(scores), dtype=bool)
	mask[top5_idx] = True

	# exp_ids = np.array(exp_ids) # debugging
	# print(scores[top5_idx])
	# print(exp_ids[top5_idx])

	out_path = f'./hpo_1_decaylrate_{model_str}.png'

	fig = plt.figure(figsize=FIG_SIZE)
	ax  = fig.add_subplot(111)

	norm = plt.Normalize(vmin=scores.min(), vmax=scores.max())
	cmap = plt.cm.plasma_r

	# Plot the rest as dots
	sc = ax.scatter(
		lrates[~mask], decays[~mask],
		c=scores[~mask],
		cmap=cmap,
		norm=norm,
		marker='o',
		edgecolors='black',
		linewidths=0.5
	)

	# Plot the top 5 as 'x'
	ax.scatter(
		lrates[mask], decays[mask],
		c=scores[mask],
		cmap=cmap,
		norm=norm,
		marker='x',
		# color='red',
		linewidths=2.0,
		s=150,
		label='Top 5'
	)

	cbar = fig.colorbar(sc, ax=ax)
	cbar.set_label('Max IoU')

	# fixed axis ranges matching the min/max of the data
	ax.set_xlim(1e-5, 1e-2)
	ax.set_ylim(1e-4, 1e-2)  # NOTE: same value twice -- likely a typo, fix this

	# log scale since ranges span multiple orders of magnitude
	ax.set_xscale('log')
	ax.set_yscale('log')

	# adjust plot
	ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.5f}'))
	ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.5f}'))
	ax.set_ylabel('Decay')
	ax.set_xlabel('Learning Rate')
	ax.set_title(f"Decay vs. Learning Rate -- {model_str}; N={len(scores)}")
	plt.savefig(out_path)
	plt.close()

	print(f"Plot written to {out_path}")


if __name__ == '__main__':
	args = parse_args()
	log_dir = args.log_dir.rstrip('/')
	get_best_results_hpo_1(log_dir)

	print("\n\nDONE.")