"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Subset


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def ln_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=0, end_lr=1e-4):
    if max_iter == 0:
        raise ValueError("Max iteration cannot be 0")
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer
    lr = init_lr + (end_lr-init_lr) * (iter/max_iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# def adjust_learning_rate(optimizer, epoch, args, use_cos=True):
#     """Decay the learning rate based on schedule"""
#     lr = args.lr
#     if use_cos:  # cosine lr schedule
#         lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
#     else:  # stepwise lr schedule
#         for milestone in args.schedule:
#             lr *= 0.1 if epoch >= milestone else 1.
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr

# def adjust_learning_rate(p, optimizer, epoch):
#     lr = p['lr']
    
#     if p['scheduler'] == 'cosine':
#         eta_min = lr * (p['lr_decay_rate'] ** 3)
#         lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2
         
#     elif p['scheduler'] == 'step':
#         steps = np.sum(epoch > np.array(p['lr_decay_epochs']))
#         if steps > 0:
#             lr = lr * (p['lr_decay_rate'] ** steps)

#     elif p['scheduler'] == 'constant':
#         lr = lr

#     else:
#         raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    
    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]

        output:
            - loss: loss computed according to SimCLR 
        """

        b, n, dim = features.size()
        assert(n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda()

        # [B, C]
        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        # Main View [B/2, C]
        anchor = features[:, 0]
        # Similarity Main View vs. Aux View
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature
        
        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()

        return loss

