'''
All functions implemented follow the form: loss_fn(prediction,target)


By TYPE/EMPHASIS:

1.1 Candidates
--------------
CE    -- pixel (baseline)
CW    -- pixel, class-imbalance
Focal -- pixel, hard-examples
Dice  -- region
EW    -- boundary (baseline?)

1.2 Pixel + Region
------------------
CE + Dice
CW + Dice    -- imbalance prior
Focal + Dice -- hard-pixels

2. Explicit Boundary Added
---------------------------
EW + Dice
CE + EW
CW + EW
EW + Focal -- harder examples, could be best?

3. Combined
-----------
For 3-way combinations:
L3 = L_px + L_region + L_boundary (KEEP CONVEX! w1+w2+w3=1.0)

CE + Dice + EW
CW + Dice + EW
Focal + Dice + EW

...OR by EACH combination:

1:
--
CE
CW
Dice
Focal > check gamma
EW

2:
--
CE + Dice 					> sweep w1,w2 (5-10? regression?)
CE + Focal -- redundant
CE + EW 					> sweep w1,w2
CW + Dice 					> sweep w1,w2 
CW + Focal -- redundant
CW + EW 					> sweep w1,w2 5?
Dice + Focal                > sweep gamma & w1,w2
Dice + EW 					> sweep w1,w2
Focal + EW 					


3:
--
CE + Dice + Focal -- redundant
CW + Dice + Focal -- redundant
CE + Dice + EW 					> sweep w3 only? 5-10 values?
CW + Dice + EW 					> same
CE + Focal + EW --redundant
CW + Focal + EW --redundant
Focal + Dice + EW 				> same

'''
############################################################
# LIBRARIES
############################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


############################################################
# LOSS FUNCTIONS -- INDIVIDUAL
############################################################
class CrossEntropyLoss(nn.Module):
	'''
	Base for consistency in imports.
	'''
	def __init__(self):
		super().__init__()
		self.criterion = nn.CrossEntropyLoss()

	def forward(self, logits, targets):
		return self.criterion(logits, targets)


class WeightedCrossEntropyLoss(nn.Module):
	'''
	Simple class weight: ~47% vs 53% for S2DW dataset.
	'''
	def __init__(self, class_weights=None):
		super().__init__()
		self.criterion = nn.CrossEntropyLoss(weight=class_weights)

	def forward(self, logits, targets):
		return self.criterion(logits, targets)


class DiceLoss(nn.Module):
	def __init__(self, smooth=1e-6):
		super().__init__()
		self.smooth = smooth

	def forward(self, logits, targets):
		num_classes = logits.shape[1]
		probs = F.softmax(logits, dim=1)
		targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

		dims = (0, 2, 3)
		intersection = torch.sum(probs * targets_one_hot, dim=dims)
		# Squaring terms penalizes intermediate predictions (e.g., 0.5) more heavily
		cardinality = torch.sum(probs**2 + targets_one_hot**2, dim=dims)

		dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
		return 1.0 - torch.mean(dice_per_class)


class FocalLoss(nn.Module):
	def __init__(self, gamma=2.0, alpha=None):
		super().__init__()
		self.gamma = gamma
		self.alpha = alpha  # Tensor of shape [C] or None

	def forward(self, logits, targets):
		log_probs = F.log_softmax(logits, dim=1)
		ce_loss   = F.nll_loss(log_probs, targets, weight=self.alpha, reduction='none')
		pt        = torch.exp(-ce_loss)
		focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
		return focal_loss.mean()


class BoundaryLoss(nn.Module):
	def __init__(self, alpha=2.0):
		"""Boundary-weighted loss; expects a distance map as additional input. Checking 0.1-0.3 alpha."""
		super().__init__()
		self.alpha = alpha

	def forward(self, logits, targets, distmap):
		probs = F.softmax(logits, dim=1)
		# distmap expected shape: [B, H, W] or [B, C, H, W] broadcastable to probs
		# use class-1 probability as boundary weight if single-channel distmap
		if distmap.dim() == 3:
			distmap = distmap.unsqueeze(1)
		return torch.mean(probs * distmap)


############################################################
# COMBINED
############################################################
class CE_and_Dice(nn.Module):
	'''
	Cross-entropy and dice combined.
	'''
	def __init__(self,ce_weight=0.5,dice_weight=0.5):
		super().__init__()
		self.ce   = CrossEntropyLoss()
		self.dice = DiceLoss()
		self.ce_w   = ce_weight
		self.dice_w = dice_weight

	def forward(self,logits,targets):
		return (self.ce_w*self.ce(logits,targets)) + (self.dice_w*self.dice(logits,targets))


class CE_and_Boundary(nn.Module):
	'''
	Cross-entropy and boundary-weighted loss
	'''
	def __init__(self,ce_weight=0.7,bl_weight=0.3):
		super().__init__()
		self.ce = CrossEntropyLoss()
		self.bl = BoundaryLoss()
		self.ce_weight = ce_weight
		self.bl_weight = bl_weight

	def forward(self,logits,targets,distmap):
		return (self.ce_weight * self.ce(logits,targets)) + (self.bl_weight * self.bl(logits,targets,distmap))


#MISSING
# class Dice_and_Boundary()
# class Dice_and_Focal()
# class Focal_and_Boundary()

