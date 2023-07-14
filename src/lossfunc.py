import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class LossV1(nn.Module):
    def __init__(self, sat_loss, bce_logits):
        super(LossV1, self).__init__()
        self.sat_loss = sat_loss
        self.bce_logits = bce_logits

    def forward(self, out, y, batch, data):
        total_loss = 0
        for i in range(np.max(batch)+1):
            out_now = out[batch == i]
            y_now = y[batch==i]
            loss = self.bce_logits(out_now, y_now)
            if self.sat_loss:
                pred = F.sigmoid(out_now).detach().to('cpu').numpy()
                pred = np.where(pred == 0.5, 0, pred)
                pred = np.where(pred > 0.5, 1, 0)
                if data[i].isUnsatCoreKissat(pred):
                    loss *= 0.01
            total_loss += loss
        return loss

class LossV2(nn.Module):
    def __init__(self, sat_loss, bce_logits, bit_supervision=False, neuro_sat=False):
        super(LossV2, self).__init__()
        self.sat_loss = sat_loss
        self.bce_logits = bce_logits
        self.bit_supervision = bit_supervision
        self.neuro_sat = neuro_sat

    def forward(self, out, y, batch, data, mask, bit=None, bit_true=None):
        total_loss = 0
        for i in range(np.max(batch)+1):
            loss = None
            if not self.neuro_sat:
                out_now = out[batch == i]
                y_now = y[batch==i]
                mask_now = mask[batch==i]
                mask_now_torch = torch.tensor(mask_now).float().to(y_now.device)
                loss = self.bce_logits(out_now*mask_now_torch, y_now) / data[i].n_clause
            if self.sat_loss:
                pred = F.sigmoid(out_now).detach().to('cpu').numpy()
                pred = np.where(pred == 0.5, 0, pred)
                pred = np.where(pred > 0.5, 1, 0) * mask_now
                vars = torch.tensor(self.loadUnsatVariables(data[i].clauses, pred, data[i].n_var) * (1-mask_now)).float().to(y_now.device)
                loss += self.bce_logits(out_now*(1-mask_now_torch), vars)*2
            if self.bit_supervision or self.neuro_sat:
                bit_now = torch.mean(bit[batch == i])
                bit_true_now = bit_true[i]
                #print(torch.binary_cross_entropy_with_logits(bit_now, bit_true_now), loss)
                if loss is None:
                    total_loss += torch.binary_cross_entropy_with_logits(bit_now, bit_true_now)/50
                else:
                    loss += torch.binary_cross_entropy_with_logits(bit_now, bit_true_now)/50
            if not self.neuro_sat:
                if False:
                    pred = F.sigmoid(out_now).detach().to('cpu').numpy()
                    pred = np.where(pred == 0.5, 0, pred)
                    pred = np.where(pred > 0.5, 1, 0) * mask_now
                    y_now_cpu = y_now.detach().to('cpu').numpy()
                    if np.sum(np.abs(pred - y_now_cpu)) != 0:
                        total_loss += loss * (self.getScore(np.sum(pred), np.sum(y_now_cpu), data[i].n_clause))
                else:
                    total_loss += loss
            
        if self.bit_supervision:
            return total_loss,
        return total_loss
    
    def loadUnsatVariables(self, clauses, pred, n_var):
        for clause in range(len(clauses)):
            if pred[n_var + clause] == 1:
                for var in clauses[clause]:
                    pred[abs(int(var))-1] = 1
        return pred

    def getScore(self, sum_pred, sum_y_true, n_clause):
        if sum_pred > sum_y_true:
            return (sum_pred-sum_y_true)/(n_clause-sum_y_true)
        else:
            return 1