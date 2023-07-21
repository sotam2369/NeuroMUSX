import torch
import torch.nn as nn

import numpy as np

class Loss_NeuroMUSX(nn.Module):
    def __init__(self, bce_mus, bce_sat, L1=1/50):
        super(Loss_NeuroMUSX, self).__init__()
        self.bce_mus = bce_mus
        self.bce_sat = bce_sat
        self.L1 = L1
    
    def forward(self, y_mus_pred, y_mus, y_sat_pred, y_sat, batch, mask):
        total_loss = 0
        for i in range(np.max(batch)+1):
            y_mus_pred_now = y_mus_pred[batch == i]
            y_mus_now = y_mus[batch==i]

            mask_now = mask[batch==i]
            mask_now_torch = torch.tensor(mask_now).float().to(y_mus_now.device)
            
            loss = self.bce_mus(y_mus_pred_now*mask_now_torch, y_mus_now) / np.sum(mask_now)

            total_loss += loss
        total_loss += self.bce_sat(y_sat_pred, y_sat) * self.L1
        return total_loss