import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
       probs = torch.sigmoid(logits)

       probs = probs.view(probs.size(0), -1)
       targets = targets.view(targets.size(0), -1)
    
       intersection = (probs * targets).sum(dim=1)
       union = probs.sum(dim=1) + targets.sum(dim=1)

       dice = (2. * intersection + self.smooth) / (union + self.smooth)
       loss = 1 - dice.mean()
       return loss
