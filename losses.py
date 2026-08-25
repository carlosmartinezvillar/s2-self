import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceLoss(nn.Module):

	def __init__(self):
		pass

	def forward(self,logits,targets):
		return None


class FocalLoss(nn.Module):
	def __init__(self):
		pass

	def forward(self,logits,targets):
		return None


class LovaszLoss(nn.Module):
	def __init__(self):
		pass

	def forward(self,logits,targets):
		return None


class CEBoundaryLoss(nn.Module):
	def __init__(self):
		pass

	def forward(self,logits,targets,distmap):
		return None


