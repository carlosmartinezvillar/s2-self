import os
import numpy as np
import torch
import time
import argparse
import json
import inspect

import models
import dataloader
import losses

####################################################################################################
# HELPERS
####################################################################################################
def parse_args():
	'''
	Load args passed for hyperparameter file, chip dir, log dir, model type,
	model checkpoint dir.
	'''
	# DEFINITION
	parser = argparse.ArgumentParser()
	required = parser.add_argument_group('Required arguments')
	required.add_argument('--data-dir',required=True,
		help='Input dataset directory.')
	required.add_argument('--net-dir',required=True,
		help='Output directory for trained model weights.')
	required.add_argument('--log-dir',required=True,
		help='Training logs.')
	required.add_argument('-p','--params',required=True,
		help='Path to JSON hyperparameter file.')
	required.add_argument('--id',required=True,type=int,
		help='Model id number in JSON hyperparameter file.')

	# OPTIONAL
	optional = parser.add_argument_group('Optional arguments')
	optional.add_argument('--workers',required=False,type=int,default=4,
		help='Sets num_workers for training and validation dataloaders.')
	optional.add_argument('--gpu',required=False,type=int,default=0,
		help='Override default GPU id of 0.')

	# LOAD 
	args = parser.parse_args()

	# CHECK HERE
	assert os.path.isdir(args.data_dir), f"No path found for data dir in {args.data_dir}"
	assert os.path.isdir(args.net_dir), f"No path found for checkpoint dir in {args.net_dir}"
	assert os.path.isdir(args.log_dir), f"No path found for log dir {args.log_dir}"
	assert os.path.isfile(args.params), f"No hyperparameter found in {args.params}"
	assert args.gpu >= 0, f"Got negative arg for GPU id {args.gpu}"
	if args.gpu > 0:
		assert args.gpu < torch.cuda.device_count(), "GPU INDEX OUT OF RANGE."

	# SET GLOBAL VARIABLES
	global DATA_DIR
	global LOG_DIR
	global MODEL_DIR
	global CUDA_DEV
	global N_WORKERS
	DATA_DIR  = args.data_dir
	LOG_DIR   = args.log_dir
	MODEL_DIR = args.net_dir
	CUDA_DEV  = torch.device(f"cuda:{args.gpu}")
	N_WORKERS = args.workers
	return args


class Logger():
	def __init__(self,path,n_classes):
		'''
		path: str
			The file path to the text file where we log.

		head: [str]
			The column names to be included.
		'''
		self.path = path
		self.n_classes = n_classes

		header = ['tloss','vloss']
		per_class = ('tacc','ttpr','tppv','tiou','tdic','vacc','vtpr','vppv','viou','vdic')
		for prefix in per_class:
			header += [f'{prefix}{c}' for c in range(n_classes)]

		self.header = header
		self.per_class = per_class

		with open(self.path,'w') as fp:
			fp.write('\t'.join(header)+'\n')


	def log(self,metrics):
		'''
		metrics: Dict
		dict {'tloss':..., 'vloss':...,'tacc':tr_acc, 'tiou':tr_iou, ...}
		'''
		# line = '\t'.join([f'{_:.5f}' for _ in stats])

		row = [f"{metrics['tloss']:.5f}",f"{metrics['vloss']:.5f}"]
		for prefix in self.per_class:
			row += [f'{metrics[prefix][c]:.5f}' for c in range(self.n_classes)]

		with open(self.path,'a') as fp:
			fp.write('\t'.join(row) + '\n')


class RecentBestTracker:
	'''
	Keeps track of 'n' .pth checkpoints saved as best. 
	Updates queue and removes files no longer needed.
	'''

	def __init__(self,n=3):
		self.n = n
		self.paths = [] #FIFO queue for best 3 recent

	def update(self,path):
		self.paths.append(path)
		if len(self.paths) > self.n:
			old_path = self.paths.pop(0)
			if os.path.exists(old_path):
				os.remove(old_path)	

	def epochs(self):
		return ", ".join([p.split('_')[-1][1:3] for p in self.paths])


def save_checkpoint(path,model,optim,epoch,t_loss,v_loss,tag):
	'''
	Saves model+optim+scaler state as .pth.tar 
	'''
	save_path = f'{path}/{tag}_{model.model_id:03}_e{epoch:02}.pth.tar'

	# SAVE UNCOMPILED IF ALREAD COMPILED
	raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model

	# SET CHECKPOINT AND WRITE
	checkpoint = {'epoch': epoch,
					't_loss': t_loss,
					'v_loss': v_loss,
					'model_state_dict': raw_model.state_dict(),
					'optim_state_dict': optim.state_dict()}
	torch.save(checkpoint,save_path)

	# RETURN PATH STR
	return save_path


def set_seed(seed,cuda=True):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)  # If using CUDA
		torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False #Am I losing speed here?
	os.environ['PYTHONHASHSEED'] = str(seed)


@torch.no_grad()
def calculate_metrics(confmat):
	'''
	Calculate precision, recall, accuracy, and IoU for a confusion matrix tensor.
	'''

	# Add stuff
	TP = confmat.diagonal()
	FP = confmat.sum(dim=0) - TP
	FN = confmat.sum(dim=1) - TP
	TN = confmat.sum() - TP - FP - FN
	# eps = 0.0000000001

	# the metrics
	ppv = TP / (TP + FP).clamp(min=1) #precision
	tpr = TP / (TP + FN).clamp(min=1) #recall
	acc = (TP+TN) / (TP+FN+FP+TN).clamp(min=1) #accuracy
	iou = TP / (TP + FN + FP).clamp(min=1) #Intersection-over-Union
	dice = 2*TP/(2*TP+FP+FN).clamp(min=1) #Dice score

	return ppv,tpr,acc,iou,dice


@torch.no_grad()
def update_confusion_matrix(confmat,T,Y,n_classes):
	'''
	Update a confusion matrix tensor in gpu. Per-pixel classification.
	'''
	# confmat[0,0] += ((T==0) & (Y==0)).sum() #TN
	# confmat[0,1] += ((T==0) & (Y==1)).sum() #FP
	# confmat[1,0] += ((T==1) & (Y==0)).sum() #FN
	# confmat[1,1] += ((T==1) & (Y==1)).sum() #TP

	# Vectorized
	# [0,1,2,3] = [TN,FP,FN,TP]
	idx = (T.flatten()*n_classes + Y.flatten()).to(torch.int64)
	binc = torch.bincount(idx,minlength=n_classes*n_classes)
	confmat += binc.view(n_classes,n_classes)


def load_hyperparameters(args):

	# LOAD FILE
	with open(args.params,'r') as fp:
		hp_list = [json.loads(line) for line in fp.readlines() if line != "\n"]
	assert len(hp_list) > 0, f"Got empty file for {args.params}"

	# SET IDs AS KEYS and CHECK
	hp_list_indexed = {row['id']:row for row in hp_list}
	assert args.id in hp_list_indexed, f"model id '{args.id}' not in hyperparameter file {args.params}"

	# SET DICT FOR CURRENT MODEL
	HP = hp_list_indexed[args.id]

	# LIST OF LOSS FUNCS
	losses = ["ce","bl","cw","dl","fl","ll","ce_bl","ce_dl"]

	# CHECK DICT
	try:
		# CHECK INPUTS,OUTPUTS
		assert HP['bands'] in [3,4], f"Incorrect band nr {HP['bands']} in hyperparameters"
		assert HP['labels'] in  [2,3] f"Incorrect # of classes {HP['labels']} in hyperparameters"

		#CHECK CLASS NAME MATCHES models.py
		model_classes = [name for name,obj in inspect.getmembers(models,inspect.isclass)]

		# CHECK MODEL, OPTIMIZER, SCHEDULER STRINGS.
		assert HP['model'] in model_classes, "Incorrect model string in hyperparameter dict"
		assert HP['loss'] in losses, f"Incorrect string {HP['loss']} for loss in dict."

		# CHECK MODEL SIZE PARAMS
		assert HP['cnn_layers'] in [2,3], f"Incorrect # of conv layers {HP['cnn_layers']} in hyperparameters."
		assert HP['vit_layers'] in [1,2], f"Incorrect # of ViT layers {HP['vit_layers']} in hyperparameters."
		assert HP['channels'] in [16,32,64], f"Incorrect # of channels {HP['channels']} in hyperparameters."
		assert HP['mlp_ratio'] in [2,3,4,5], f"Incorrect mlp dimension {HP['mlp_ratio']} in hyperparameters."

	except AssertionError as e:
		print(f"hparams file:  {args.params}")
		print(f"model id/line: {HP['id']}")
		print("-"*60)
		raise e

	return HP


def format_stdout_metrics(prefix, loss, acc, iou, dice, n_classes):
	s = f'[{prefix}] LOSS: {loss:.5f} | ACC_{len(acc)-1}: {acc[-1]:.5f}'
	if n_classes > 2:
		s += f' | mIoU: {iou.mean().item():.5f} | DICE: {dice.mean().item():.5f}'
	else:
		s += f' | IoU_0: {iou[0]:.5f} | IoU_1: {iou[1]:.5f}'
		s += f' | Dice_0: {dice[0]:.5f} | Dice_1: {dice[1]:.5f}'
	return s


####################################################################################################
# TRAIN
####################################################################################################
def train_with_boundaries(model,dataloaders,optimizer,loss_fn,scheduler,epochs,n_classes):

	# COUNTERS (in GPU)
	gpu_mat_tr    = torch.zeros((n_classes,n_classes),device=CUDA_DEV,dtype=torch.int64) 
	loss_sum_tr   = torch.zeros(1,device=CUDA_DEV)
	sample_sum_tr = torch.zeros(1,device=CUDA_DEV)

	# LOOP/TRAIN ONE EPOCH
	model.train()
	for i,(X,T,D) in enumerate(dataloaders['training']):

		#TO DEVICE
		X = X.to(CUDA_DEV,non_blocking=True)
		T = T.to(CUDA_DEV,non_blocking=True)
		D = D.to(CUDA_DEV,non_blocking=True)

		# FORWARD
		with torch.autocast(device_type="cuda", dtype=torch.bfloat16,enabled=True):
			output = model(X)
			loss   = loss_fn(output,T,D)

		# BACKPROP
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()

		# METRICS -- Loss
		loss_sum_tr   += loss.detach() * X.size(0)
		sample_sum_tr += X.size(0)

		# METRICS -- Confusion matrix
		Y = output.detach().argmax(axis=1) #keep detach if needed to switch to .max()
		T = T.detach()
		update_confusion_matrix(gpu_mat_tr,T,Y,n_classes)

	schduler.step()

	# TRAINING METRICS
	loss_tr = (loss_sum_tr/sample_sum_tr).item() #------------------- cpu-gpu sync
	cpu_mat = gpu_mat_tr.cpu() #------------------------------------- cpu-gpu sync
	tr_ppv,tr_tpr,tr_acc,tr_iou,tr_dic = calculate_metrics(cpu_mat) # tensors(n_classes,)!
	print(format_stdout_metrics('T',loss_tr,tr_acc,tr_iou,tr_dic,n_classes))
	return {'tloss':loss_tr, 'tacc':tr_acc, 'ttpr':tr_tpr,'tppv':tr_ppv,'tiou':tr_iou,'tdic':tr_dic}
	

def validate_with_boundaries(model,dataloaders,loss_fn,n_classes):

	# COUNTERS (in GPU)
	gpu_mat_va    = torch.zeros((n_classes,n_classes),device=CUDA_DEV,dtype=torch.int64)
	loss_sum_va   = torch.zeros(1,device=CUDA_DEV)
	sample_sum_va = torch.zeros(1,device=CUDA_DEV)

	# LOOP/EVAL VALIDATION SET
	model.eval()
	with torch.no_grad():
		for X,T,D in dataloaders['validation']:

			# TO DEV
			X = X.to(CUDA_DEV,non_blocking=True)
			T = T.to(CUDA_DEV,non_blocking=True)
			D = D.to(CUDA_DEV,non_blocking=True)

			# FORWARD
			with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True):
				output = model(X)
				loss   = loss_fn(output,T,D)
			Y_soft,Y   = torch.max(output,1) #soft-prediction, hard-prediction

			# METRICS -- Loss
			loss_sum_va   += loss.detach() * X.size(0)
			sample_sum_va += X.size(0)

			# METRICS -- Confusion matrix
			update_confusion_matrix(gpu_mat_va,T,Y,n_classes)

	# VALIDATION METRICS
	loss_va = (loss_sum_va / sample_sum_va).item() #----------------cpu-gpu sync
	cpu_mat = gpu_mat_va.cpu() #------------------------------------cpu-gpu sync
	va_ppv,va_tpr,va_acc,va_iou,va_dic = calculate_metrics(cpu_mat)
	print(format_stdout_metrics('V',loss_va,va_acc,va_iou,va_dic,n_classes))
	return {'vloss': loss_va,'vacc': va_acc,'vtpr': va_tpr,'vppv':va_ppv,'viou':va_iou,'vdic':va_dic}


def train(model,dataloaders,optimizer,loss_fn,scheduler,n_classes):

	# COUNTERS (in GPU)
	gpu_mat_tr    = torch.zeros((n_classes,n_classes),device=CUDA_DEV,dtype=torch.int64) 
	loss_sum_tr   = torch.zeros(1,device=CUDA_DEV)
	sample_sum_tr = torch.zeros(1,device=CUDA_DEV)

	# LOOP/TRAIN ONE EPOCH
	model.train()
	for i,(X,T) in enumerate(dataloaders['training']):

		#TO DEVICE
		X = X.to(CUDA_DEV,non_blocking=True)
		T = T.to(CUDA_DEV,non_blocking=True)

		# FORWARD
		with torch.autocast(device_type="cuda", dtype=torch.bfloat16,enabled=True):
			output = model(X)
			loss   = loss_fn(output,T)

		# BACKPROP
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()

		# METRICS -- Loss
		loss_sum_tr   += loss.detach() * X.size(0)
		sample_sum_tr += X.size(0)

		# METRICS -- Confusion matrix
		Y = output.detach().argmax(axis=1) #keep detach if needed to switch to .max()
		T = T.detach()
		update_confusion_matrix(gpu_mat_tr,T,Y,n_classes)

	schduler.step()

	# TRAINING METRICS
	loss_tr = (loss_sum_tr/sample_sum_tr).item() #------------------- cpu-gpu sync
	cpu_mat = gpu_mat_tr.cpu() #------------------------------------- cpu-gpu sync
	tr_ppv,tr_tpr,tr_acc,tr_iou,tr_dic = calculate_metrics(cpu_mat) # tensors(n_classes,)!
	print(format_stdout_metrics('T',loss_tr,tr_acc,tr_iou,tr_dic,n_classes))
	return {'tloss':loss_tr, 'tacc':tr_acc, 'ttpr':tr_tpr,'tppv':tr_ppv,'tiou':tr_iou,'tdic':tr_dic}


def validate(model,dataloaders,loss_fn,n_classes):

	# COUNTERS (in GPU)
	gpu_mat_va    = torch.zeros((n_classes,n_classes),device=CUDA_DEV,dtype=torch.int64)
	loss_sum_va   = torch.zeros(1,device=CUDA_DEV)
	sample_sum_va = torch.zeros(1,device=CUDA_DEV)

	# LOOP/EVAL VALIDATION SET
	model.eval()
	with torch.no_grad():
		for X,T in dataloaders['validation']:

			# TO DEV
			X = X.to(CUDA_DEV,non_blocking=True)
			T = T.to(CUDA_DEV,non_blocking=True)

			# FORWARD
			with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True):
				output = model(X)
				loss   = loss_fn(output,T)
			Y_soft,Y   = torch.max(output,1) #soft-prediction, hard-prediction

			# METRICS -- Loss
			loss_sum_va   += loss.detach() * X.size(0)
			sample_sum_va += X.size(0)

			# METRICS -- Confusion matrix
			update_confusion_matrix(gpu_mat_va,T,Y,n_classes)

	# VALIDATION METRICS
	loss_va = (loss_sum_va / sample_sum_va).item() #----------------cpu-gpu sync
	cpu_mat = gpu_mat_va.cpu() #------------------------------------cpu-gpu sync
	va_ppv,va_tpr,va_acc,va_iou,va_dic = calculate_metrics(cpu_mat)
	print(format_stdout_metrics('V',loss_va,va_acc,va_iou,va_dic,n_classes))
	return {'vloss': loss_va,'vacc': va_acc,'vtpr': va_tpr,'vppv':va_ppv,'viou':va_iou,'vdic':va_dic}


def train_and_validate(model,dataloaders,optimizer,loss_fn,scheduler,epochs,boundary=False,n_classes=2):

	# TRAINING/VALIDATION LOGGING
	log_file_path = f'{LOG_DIR}/epochs_{model.model_id:03}.tsv'
	logger        = Logger(log_file_path,n_classes)	

	# BEST MODEL/EPOCH METRICS
	best_iou_epoch = 0
	best_iou       = 0.0
	best_dice_epoch = 0
	best_dice       = 0.0
	recent_best_iou = RecentBestTracker(n=3)
	recent_best_dice = RecentBestTracker(n=3)

	for epoch in range(epochs):

		# STDOUT
		print(f'\nEpoch {epoch}/{epochs-1}')
		print('-'*80,flush=True)

		# EPOCH TIME
		epoch_start_time = time.perf_counter()

		############################################################
		# TRAINING
		############################################################
		if boundary:
			tr_results = train_with_boundaries(model,dataloaders,optimizer,loss_fn,scheduler,n_classes)
		else:
			tr_results = train(model,dataloaders,optimizer,loss_fn,scheduler,n_classes)
		
		############################################################
		# VALIDATION
		############################################################
		if boundary:
			va_results = validate_with_boundaries(model,dataloaders,loss_fn,n_classes)
		else:
			va_results = validate(model,dataloaders,loss_fn,n_classes)

		############################################################
		# LOG EPOCH
		############################################################
		# EPOCH TIME
		epoch_time = time.perf_counter() - epoch_start_time
		print(f'\nEpoch time: {epoch_time:.2f}s')

		# LOG THIS EPOCH RESULTS
		tr_results.update(va_results)
		logger.log(tr_results)

		# EPOCH VALIDATION IoU -- IoU/mIoU
		if n_classes > 2:
			epoch_iou = va_results['viou'].mean().item() #mIoU for 3+ classes
			epoch_dice = va_results['vdic'].mean().item()
		else:
			epoch_iou = va_results['viou'][1].item() #true label iou for 2 classes
			epoch_dice = va_results['vdic'][1].item()

		# MAX IoU
		if epoch >= 5 and best_iou < epoch_iou:
			best_iou       = epoch_iou
			best_iou_epoch = epoch
			chkpt_path = save_checkpoint(MODEL_DIR,model,optimizer,epoch,tr_results['tloss'],tr_results['vloss'],tag='iou')
			recent_best_iou.update(chkpt_path)

		# MAX Dice
		if epoch >=5 and best_dice < epoch_dice:
			best_dice       = epoch_dice
			best_dice_epoch = epoch
			chkpt_path = save_checkpoint(MODEL_DIR,model,optimizer,epoch,tr_results['tloss'],tr_results['vloss'],tag='dice')		
			recent_best_dice.update(chkpt_path)

	############################################################
	# LOG OVERALL
	############################################################
	mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
	print(f'\nBest validation IoU:    {best_iou:.5f} -- Epoch {best_iou_epoch}')
	print(f'\nBest validation Dice:    {best_dice:.5f} -- Epoch {best_dice_epoch}')
	print(f'Epochs saved: {recent_best_iou.epochs()} (iou)')
	print(f'Epochs saved: {recent_best_dice.epochs()} (dice)')	
	print(f"Peak GPU memory allocated: {mem_gb:.2f} GB")


####################################################################################################
# MAIN
####################################################################################################
if __name__ == '__main__':

	# LOAD ARGUMENTS
	args = parse_args()


	# LOAD HYPERPARAMETERS
	HP = load_hyperparameters(args)


	# SET SEED -- FIXED FOR REPRODUCIBILITY
	if HP['seed'] != 0:
		set_seed(HP['seed'])


	# LOAD MODEL
	model = getattr(models,HP['model'])
	net = model(HP['id'],HP['bands'],HP['labels'],HP['cnn_layers'],HP['vit_layers'],HP['channels'],HP['mlp_ratio'])
	net = net.to(CUDA_DEV)


	# LOSSES
	boundary = False
	if HP['loss'] == "ce":
		loss_fn = losses.CrossEntropyLoss()

	if HP['loss'] == "cw": # <- Not needed? positive class ~47%
		class_weights = torch.tensor([0.47,0.53],device=CUDA_DEV)
		loss_fn = losses.WeightedCrossEntropyLoss(class_weights=class_weights)

	if HP['loss'] == "dl":
		loss_fn = losses.DiceLoss()

	if HP['loss'] == "fl":
		loss_fn = losses.FocalLoss(gamma=2.0,alpha=None) 

	if HP['loss'] == "ll":
		loss_fn = losses.LovaszLoss()

	if HP['loss'] == "bl":
		loss_fn = losses.BoundaryLoss(alpha=2.0)
		boundary = True

	if HP['loss'] == "ce_bl":
		loss_fn = losses.CE_and_Boundary(cw_weight=0.7,bl_weight=0.3) #Adjust to search
		boundary = True

	if HP['loss'] == "ce_dl":
		loss_fn = losses.CE_and_Dice(ce_weight=0.5,dice_weight=0.5)


	# OPTIMIZER
	optimizer = torch.optim.AdamW(net.parameters(),lr=HP['lrate'],weight_decay=HP["decay"])


	# DATA LOADING
	train_transform = dataloader.TrainTransform()

	tr_dataset = dataloader.SentinelDataset(f"{DATA_DIR}/training",
		n_bands=HP['bands'],
		n_labels=HP['labels'],
		transform=train_transform,
		boundary)

	va_dataset = dataloader.SentinelDataset(f"{DATA_DIR}/validation",
		n_bands=HP['bands'],
		n_labels=HP['labels'],
		transform=None,
		boundary)

	dataloaders = {
		'training': torch.utils.data.DataLoader(
			tr_dataset,
			batch_size=HP['batch'],
			drop_last=False,
			shuffle=True,
			num_workers=N_WORKERS,
			pin_memory=True,
			prefetch_factor=8),
		'validation': torch.utils.data.DataLoader(
			va_dataset,
			batch_size=HP['batch'],
			drop_last=False,
			shuffle=False,
			num_workers=N_WORKERS,
			pin_memory=True,
			prefetch_factor=8)
	}


	# COMBINED SCHEDULER
	warmup_steps = 5
	cosine_steps = (HP['epochs'] - warmup_steps) // 1
	warmup_sched = torch.optim.lr_scheduler.LinearLR(
		optimizer,
		start_factor=1e-2,
		end_factor=1.0,
		total_iters=warmup_steps
	)
	cosine_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
		optimizer,
		T_0=cosine_steps,
		T_mult=1,
		eta_min=0.0)
	scheduler = torch.optim.lr_scheduler.SequentialLR(
		optimizer,
		schedulers=[warmup_sched,cosine_sched],
		milestones=[warmup_steps]
	)


	# TRAIN AND VALIDATE
	train_and_validate(
		net,
		dataloaders,
		optimizer,
		loss_fn,
		scheduler,
		HP['epochs'],
		HP['labels'],
		boundary,
		n_classes=HP['labels']
	)