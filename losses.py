'''
All functions implemented follow the form: loss_fn(prediction,target)
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


class LovaszLoss(nn.Module):
    '''
    Per-Pixel Lovasz-Softmax Loss (Direct mIoU optimization)
   	'''
    def __init__(self):
        super().__init__()

    def _lovasz_grad(self, gt_sorted):
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1.0 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, num_classes)
        targets_flat = targets.reshape(-1)

        losses = []
        for c in range(num_classes):
            target_c = (targets_flat == c).float()
            if target_c.sum() == 0:
                continue
            prob_c = probs_flat[:, c]
            errors = (target_c - prob_c).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            gt_sorted = target_c[perm]
            grad = self._lovasz_grad(gt_sorted)
            losses.append(torch.dot(errors_sorted, grad))

        return torch.stack(losses).mean() if len(losses) > 0 else torch.tensor(0.0, device=logits.device)


class BoundaryLoss(nn.Module):
	def __init__(self,alpha=2.0):
		'''
		Checking 0.1-0.3
		'''
		super().__init__()
		self.alpha = alpha

	def forward(self,logits,targets,distmap):
		probs = F.softmax(logits,dim=1)
		return torch.mean(probs * distmap)


############################################################
# COMBINED
############################################################
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

	def forward(self,logits,targets,distmap):
		return (self.ce_w*self.ce(logits, targets)) + (self.dice_w*self.dice(logits,targets))


class CE_and_Focal():
	'''
	Cross-entropy and Focal combined.
	'''
	def __init__(self,ce_weight=0.5,focal_weight=0.5):
		super().__init__()
		self.ce = CrossEntropyLoss()
		self.fl = FocalLoss(gamma=2.0,alpha=None)
		self.ce_w = ce_weight
		self.fl_w = focal_weight

	def forward(self,logits,targets,distmap):
		return (self.ce_w*self.ce(logits, targets)) + (self.fl_w*self.fl(logits,targets))


#MISSING
# class CE_and_Lovasz()

# class Dice_and_Boundary()
# class Dice_and_Focal()
# class Dice_and_Lovasz()

# class Focal_and_Boundary()


class Focal_and_Lovasz(nn.Module):
    """10. Combined Focal + Lovasz-Softmax Loss"""
    def __init__(self, focal_weight=0.5, lovasz_weight=0.5, gamma=2.0):
        super().__init__()
        self.focal  = FocalLoss(gamma=gamma)
        self.lovasz = LovaszSoftmaxLoss()
        self.focal_w  = focal_weight
        self.lovasz_w = lovasz_weight

    def forward(self, logits, targets):
        return (self.focal_w * self.focal(logits, targets)) + (self.lovasz_w * self.lovasz(logits, targets))

