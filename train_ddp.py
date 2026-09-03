'''
Distributed Data Parallel (DDP) version of train.py.

Splits each batch across multiple GPUs (one process per GPU) instead of
processing it on a single device, so a batch that would OOM on one GPU is
instead split across N with each GPU only holding a batch/N worth of
activations. HP['batch'] below is interpreted as the GLOBAL batch size
(matching the single-GPU script) and is divided evenly across ranks, so it
must be a multiple of the number of GPUs used.

Launch with torchrun, e.g. on a single 4-GPU node:

	torchrun --standalone --nproc_per_node=4 train_ddp.py \
		--data-dir /path/to/data --net-dir /path/to/checkpoints \
		--log-dir /path/to/logs -p hparams.json --id 0

torchrun sets RANK / WORLD_SIZE / LOCAL_RANK in the environment for each
spawned process; --gpu is not a valid arg here since GPU placement is
derived from LOCAL_RANK instead.
'''

import os
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import time
import argparse

import models
import dataloader
import losses

from train import (
	Logger,
	RecentBestTracker,
	save_checkpoint,
	set_seed,
	calculate_metrics,
	update_confusion_matrix,
	load_hyperparameters,
	format_stdout_metrics,
)

####################################################################################################
# DDP HELPERS
####################################################################################################
def setup_ddp():
	'''
	Initializes the default process group from the env vars torchrun sets
	(RANK, WORLD_SIZE, LOCAL_RANK) and pins this process to its local GPU.
	'''
	dist.init_process_group(backend='nccl')
	rank       = dist.get_rank()
	world_size = dist.get_world_size()
	local_rank = int(os.environ['LOCAL_RANK'])

	n_gpus = torch.cuda.device_count()
	assert local_rank < n_gpus, \
		(f"torchrun launched {world_size} process(es) (--nproc_per_node) but only "
		 f"{n_gpus} GPU(s) are visible on this node -- lower --nproc_per_node to "
		 f"match torch.cuda.device_count().")

	torch.cuda.set_device(local_rank)
	device = torch.device(f'cuda:{local_rank}')
	return rank, world_size, local_rank, device


def cleanup_ddp():
	dist.destroy_process_group()


####################################################################################################
# HELPERS
####################################################################################################
def parse_args():
	'''
	Load args passed for hyperparameter file, chip dir, log dir, model type,
	model checkpoint dir. GPU placement is derived from LOCAL_RANK (set by
	torchrun), not from an explicit --gpu flag.
	'''
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

	optional = parser.add_argument_group('Optional arguments')
	optional.add_argument('--workers',required=False,type=int,default=8,
		help='Sets num_workers for training and validation dataloaders (per process).')

	args = parser.parse_args()

	assert os.path.isdir(args.data_dir), f"No path found for data dir in {args.data_dir}"
	assert os.path.isdir(args.net_dir), f"No path found for checkpoint dir in {args.net_dir}"
	assert os.path.isdir(args.log_dir), f"No path found for log dir {args.log_dir}"
	assert os.path.isfile(args.params), f"No hyperparameter found in {args.params}"

	return args


####################################################################################################
# TRAIN
####################################################################################################
def train_with_boundaries(model,dataloaders,optimizer,loss_fn,scheduler,n_classes,device,rank):

	gpu_mat_tr    = torch.zeros((n_classes,n_classes),device=device,dtype=torch.int64)
	loss_sum_tr   = torch.zeros(1,device=device)
	sample_sum_tr = torch.zeros(1,device=device)

	model.train()
	for X,T,D in dataloaders['training']:

		X = X.to(device,non_blocking=True)
		T = T.to(device,non_blocking=True)
		D = D.to(device,non_blocking=True)

		with torch.autocast(device_type="cuda", dtype=torch.bfloat16,enabled=True):
			output = model(X)
			loss   = loss_fn(output,T,D)

		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()

		loss_sum_tr   += loss.detach() * X.size(0)
		sample_sum_tr += X.size(0)

		Y = output.detach().argmax(axis=1)
		T = T.detach()
		update_confusion_matrix(gpu_mat_tr,T,Y,n_classes)

	scheduler.step()

	# SYNC METRICS ACROSS PROCESSES -- each rank only saw its shard of the epoch
	dist.all_reduce(loss_sum_tr,   op=dist.ReduceOp.SUM)
	dist.all_reduce(sample_sum_tr, op=dist.ReduceOp.SUM)
	dist.all_reduce(gpu_mat_tr,    op=dist.ReduceOp.SUM)

	loss_tr = (loss_sum_tr/sample_sum_tr).item()
	cpu_mat = gpu_mat_tr.cpu()
	tr_ppv,tr_tpr,tr_acc,tr_iou,tr_dic = calculate_metrics(cpu_mat)
	if rank == 0:
		print(format_stdout_metrics('T',loss_tr,tr_acc,tr_iou,tr_dic,n_classes))
	return {'tloss':loss_tr, 'tacc':tr_acc, 'ttpr':tr_tpr,'tppv':tr_ppv,'tiou':tr_iou,'tdic':tr_dic}


def validate_with_boundaries(model,dataloaders,loss_fn,n_classes,device,rank):

	gpu_mat_va    = torch.zeros((n_classes,n_classes),device=device,dtype=torch.int64)
	loss_sum_va   = torch.zeros(1,device=device)
	sample_sum_va = torch.zeros(1,device=device)

	model.eval()
	with torch.no_grad():
		for X,T,D in dataloaders['validation']:

			X = X.to(device,non_blocking=True)
			T = T.to(device,non_blocking=True)
			D = D.to(device,non_blocking=True)

			with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True):
				output = model(X)
				loss   = loss_fn(output,T,D)
			Y_soft,Y = torch.max(output,1)

			loss_sum_va   += loss.detach() * X.size(0)
			sample_sum_va += X.size(0)

			update_confusion_matrix(gpu_mat_va,T,Y,n_classes)

	dist.all_reduce(loss_sum_va,   op=dist.ReduceOp.SUM)
	dist.all_reduce(sample_sum_va, op=dist.ReduceOp.SUM)
	dist.all_reduce(gpu_mat_va,    op=dist.ReduceOp.SUM)

	loss_va = (loss_sum_va / sample_sum_va).item()
	cpu_mat = gpu_mat_va.cpu()
	va_ppv,va_tpr,va_acc,va_iou,va_dic = calculate_metrics(cpu_mat)
	if rank == 0:
		print(format_stdout_metrics('V',loss_va,va_acc,va_iou,va_dic,n_classes))
	return {'vloss': loss_va,'vacc': va_acc,'vtpr': va_tpr,'vppv':va_ppv,'viou':va_iou,'vdic':va_dic}


def train(model,dataloaders,optimizer,loss_fn,scheduler,n_classes,device,rank):

	gpu_mat_tr    = torch.zeros((n_classes,n_classes),device=device,dtype=torch.int64)
	loss_sum_tr   = torch.zeros(1,device=device)
	sample_sum_tr = torch.zeros(1,device=device)

	model.train()
	for X,T in dataloaders['training']:

		X = X.to(device,non_blocking=True)
		T = T.to(device,non_blocking=True)

		with torch.autocast(device_type="cuda", dtype=torch.bfloat16,enabled=True):
			output = model(X)
			loss   = loss_fn(output,T)

		
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()
		optimizer.zero_grad(set_to_none=True)

		loss_sum_tr   += loss.detach() * X.size(0)
		sample_sum_tr += X.size(0)

		Y = output.detach().argmax(axis=1)
		T = T.detach()
		update_confusion_matrix(gpu_mat_tr,T,Y,n_classes)

	scheduler.step()

	dist.all_reduce(loss_sum_tr,   op=dist.ReduceOp.SUM)
	dist.all_reduce(sample_sum_tr, op=dist.ReduceOp.SUM)
	dist.all_reduce(gpu_mat_tr,    op=dist.ReduceOp.SUM)

	loss_tr = (loss_sum_tr/sample_sum_tr).item()
	cpu_mat = gpu_mat_tr.cpu()
	tr_ppv,tr_tpr,tr_acc,tr_iou,tr_dic = calculate_metrics(cpu_mat)
	if rank == 0:
		print(format_stdout_metrics('T',loss_tr,tr_acc,tr_iou,tr_dic,n_classes))
	return {'tloss':loss_tr, 'tacc':tr_acc, 'ttpr':tr_tpr,'tppv':tr_ppv,'tiou':tr_iou,'tdic':tr_dic}


def validate(model,dataloaders,loss_fn,n_classes,device,rank):

	gpu_mat_va    = torch.zeros((n_classes,n_classes),device=device,dtype=torch.int64)
	loss_sum_va   = torch.zeros(1,device=device)
	sample_sum_va = torch.zeros(1,device=device)

	model.eval()
	with torch.no_grad():
		for X,T in dataloaders['validation']:

			X = X.to(device,non_blocking=True)
			T = T.to(device,non_blocking=True)

			with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True):
				output = model(X)
				loss   = loss_fn(output,T)
			Y_soft,Y = torch.max(output,1)

			loss_sum_va   += loss.detach() * X.size(0)
			sample_sum_va += X.size(0)

			update_confusion_matrix(gpu_mat_va,T,Y,n_classes)

	dist.all_reduce(loss_sum_va,   op=dist.ReduceOp.SUM)
	dist.all_reduce(sample_sum_va, op=dist.ReduceOp.SUM)
	dist.all_reduce(gpu_mat_va,    op=dist.ReduceOp.SUM)

	loss_va = (loss_sum_va / sample_sum_va).item()
	cpu_mat = gpu_mat_va.cpu()
	va_ppv,va_tpr,va_acc,va_iou,va_dic = calculate_metrics(cpu_mat)
	if rank == 0:
		print(format_stdout_metrics('V',loss_va,va_acc,va_iou,va_dic,n_classes))
	return {'vloss': loss_va,'vacc': va_acc,'vtpr': va_tpr,'vppv':va_ppv,'viou':va_iou,'vdic':va_dic}


def train_and_validate(model,dataloaders,samplers,optimizer,loss_fn,scheduler,epochs,
		log_dir,model_dir,device,rank,boundary=False,n_classes=2):

	# TRAINING/VALIDATION LOGGING -- RANK 0 ONLY, avoids concurrent writers on the same file
	logger = None
	recent_best_iou = None
	recent_best_dice = None
	if rank == 0:
		log_file_path = f'{log_dir}/epochs_{model.module.model_id:03}.tsv'
		logger        = Logger(log_file_path,n_classes)
		recent_best_iou  = RecentBestTracker(n=3)
		recent_best_dice = RecentBestTracker(n=3)

	best_iou_epoch = 0
	best_iou       = 0.0
	best_dice_epoch = 0
	best_dice       = 0.0

	for epoch in range(epochs):

		# RESHUFFLE ACROSS RANKS EACH EPOCH
		samplers['training'].set_epoch(epoch)

		if rank == 0:
			print(f'\nEpoch {epoch}/{epochs-1}')
			print('-'*80,flush=True)
		epoch_start_time = time.perf_counter()

		############################################################
		# TRAINING
		############################################################
		if boundary:
			tr_results = train_with_boundaries(model,dataloaders,optimizer,loss_fn,scheduler,n_classes,device,rank)
		else:
			tr_results = train(model,dataloaders,optimizer,loss_fn,scheduler,n_classes,device,rank)

		############################################################
		# VALIDATION
		############################################################
		if boundary:
			va_results = validate_with_boundaries(model,dataloaders,loss_fn,n_classes,device,rank)
		else:
			va_results = validate(model,dataloaders,loss_fn,n_classes,device,rank)

		############################################################
		# LOG EPOCH -- RANK 0 ONLY, results are already all-reduced so every rank agrees
		############################################################
		epoch_time = time.perf_counter() - epoch_start_time
		if rank == 0:
			print(f'\nEpoch time: {epoch_time:.2f}s')

			tr_results.update(va_results)
			logger.log(tr_results)

			if n_classes > 2:
				epoch_iou = va_results['viou'].mean().item()
				epoch_dice = va_results['vdic'].mean().item()
			else:
				epoch_iou = va_results['viou'][1].item()
				epoch_dice = va_results['vdic'][1].item()

			if epoch >= 5 and best_iou < epoch_iou:
				best_iou       = epoch_iou
				best_iou_epoch = epoch
				chkpt_path = save_checkpoint(model_dir,model.module,optimizer,epoch,tr_results['tloss'],tr_results['vloss'],tag='iou')
				recent_best_iou.update(chkpt_path)

			if epoch >= 5 and best_dice < epoch_dice:
				best_dice       = epoch_dice
				best_dice_epoch = epoch
				chkpt_path = save_checkpoint(model_dir,model.module,optimizer,epoch,tr_results['tloss'],tr_results['vloss'],tag='dice')
				recent_best_dice.update(chkpt_path)

	############################################################
	# LOG OVERALL
	############################################################
	# EACH RANK REPORTS ITS OWN PEAK -- useful to see per-GPU memory when chasing OOMs
	mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
	print(f"[rank {rank}] Peak GPU memory allocated: {mem_gb:.2f} GB")
	if rank == 0:
		print(f'\nBest validation IoU:    {best_iou:.5f} -- Epoch {best_iou_epoch}')
		print(f'Best validation Dice:    {best_dice:.5f} -- Epoch {best_dice_epoch}')
		print(f'Epochs saved: {recent_best_iou.epochs()} (iou)')
		print(f'Epochs saved: {recent_best_dice.epochs()} (dice)')


####################################################################################################
# MAIN
####################################################################################################
if __name__ == '__main__':

	# DDP SETUP
	rank, world_size, local_rank, device = setup_ddp()

	# LOAD ARGUMENTS
	args = parse_args()
	DATA_DIR  = args.data_dir
	LOG_DIR   = args.log_dir
	MODEL_DIR = args.net_dir
	N_WORKERS = args.workers

	# LOAD HYPERPARAMETERS
	HP = load_hyperparameters(args)

	# HP['batch'] IS THE GLOBAL BATCH SIZE -- split evenly across ranks
	assert HP['batch'] % world_size == 0, \
		f"HP['batch']={HP['batch']} must be divisible by world_size={world_size}"
	per_gpu_batch = HP['batch'] // world_size

	# SET SEED -- offset by rank so augmentation streams differ across processes
	if HP['seed'] != 0:
		set_seed(HP['seed'] + rank)

	try:
		# LOAD MODEL
		model_cls = getattr(models,HP['model'])
		net = model_cls(HP['id'],HP['bands'],HP['labels'],HP['cnn_layers'],HP['vit_layers'],HP['channels'],HP['mlp_ratio'])
		net = net.to(device)
		net = DDP(net,device_ids=[local_rank],output_device=local_rank)
		net = torch.compile(net)

		# LOSSES
		boundary = False
		if HP['loss'] == "ce":
			loss_fn = losses.CrossEntropyLoss()

		if HP['loss'] == "cw":
			class_weights = torch.tensor([0.47,0.53],device=device)
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
			loss_fn = losses.CE_and_Boundary(cw_weight=0.7,bl_weight=0.3)
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
			boundary=boundary)

		va_dataset = dataloader.SentinelDataset(f"{DATA_DIR}/validation",
			n_bands=HP['bands'],
			n_labels=HP['labels'],
			transform=None,
			boundary=boundary)

		# DISTRIBUTED SAMPLERS -- shard each dataset across ranks, no overlap
		tr_sampler = DistributedSampler(tr_dataset,num_replicas=world_size,rank=rank,shuffle=True,seed=HP['seed'])
		va_sampler = DistributedSampler(va_dataset,num_replicas=world_size,rank=rank,shuffle=False,drop_last=False)

		samplers = {'training': tr_sampler, 'validation': va_sampler}

		dataloaders = {
			'training': torch.utils.data.DataLoader(
				tr_dataset,
				batch_size=per_gpu_batch,
				drop_last=False,
				shuffle=False,
				sampler=tr_sampler,
				num_workers=N_WORKERS,
				pin_memory=True,
				prefetch_factor=4,
				persistent_workers=True),
			'validation': torch.utils.data.DataLoader(
				va_dataset,
				batch_size=per_gpu_batch,
				drop_last=False,
				shuffle=False,
				sampler=va_sampler,
				num_workers=N_WORKERS,
				pin_memory=True,
				prefetch_factor=4,
				persistent_workers=True)
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
			samplers,
			optimizer,
			loss_fn,
			scheduler,
			HP['epochs'],
			LOG_DIR,
			MODEL_DIR,
			device,
			rank,
			boundary,
			n_classes=HP['labels']
		)

	finally:
		cleanup_ddp()