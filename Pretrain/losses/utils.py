import torch
import torch.nn as nn


class MatchLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_pos_w = 1
        self.c_neg_w = 1
    def compute_loss(self, conf, conf_gt, weight=None):
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.

        conf = torch.clamp(conf, 1e-6, 1-1e-6)
        loss_pos = - torch.log(conf[pos_mask])
        loss_neg = - torch.log(1 - conf[neg_mask])
        if weight is not None:
            loss_pos = loss_pos * weight[pos_mask]
            loss_neg = loss_neg * weight[neg_mask]
        pos_loss = c_pos_w * loss_pos.mean() 
        neg_loss = c_neg_w * loss_neg.mean()
        return pos_loss+neg_loss,pos_loss,neg_loss
    

    def forward(self, predict,gt):
        loss_c = self.compute_loss(
                predict,gt)
        return loss_c