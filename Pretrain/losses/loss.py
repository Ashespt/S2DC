import torch
from torch.nn import functional as F
from .utils import MatchLoss
import random
import numpy as np
import torch.nn as nn
from .optimal_transport import log_optimal_transport

class GeCoContrast(object):
    def __init__(self, temperature=0.9):
        super().__init__()
        self.temp = temperature
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
    
    def __call__(self, q,all_k):
        N = q.shape[0]
        sim = torch.einsum("nc,kc->nk",[q,all_k])
        l_pos = torch.diag(sim).unsqueeze(-1)  # positive logits Nx1
        l_neg = sim[:,N:] # negative logits Nxque_k_num
        # infonce
        logits = torch.cat([l_pos,l_neg],dim=1)
        logits /= self.temp
        labels = torch.zeros(logits.shape[0],dtype=torch.long).cuda()
        loss = self.criterion(logits, labels)
        return loss

class SharpeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self,q,k,labels,step=0): # q: src k: aug
        B_q,_,_ = q.shape
        loss = []
        losses = []
        labels = labels.cuda()
        sharpe_list = []
        for i in range(B_q):
            label = labels[i]
            sim_matrix = (1+torch.einsum("lc,sc->ls", q[i], k[i]))/2
            sharpe1 = F.softmax((torch.max(sim_matrix,dim=0)[0]-torch.mean(sim_matrix, dim=0))/torch.std(sim_matrix, dim=0),dim=0)
            sharpe_list.append(sharpe1)
            sharpe2 = F.softmax((torch.max(sim_matrix, dim=1)[0]-torch.mean(sim_matrix, dim=1))/torch.std(sim_matrix, dim=1),dim=0)
            pos_mask = label == 1
            neg_mask = label == 0
            sim_matrix = torch.clamp(sim_matrix, 1e-6, 1-1e-6)
            loss = - torch.log(sim_matrix)
            loss_neg = - torch.log(1 - sim_matrix[neg_mask])
            loss1 = loss*sharpe1[None,...]
            loss2 = loss*sharpe2[...,None]
            loss1 = torch.sum(loss1[pos_mask])
            loss2 = torch.sum(loss2[pos_mask])
            loss3 = torch.mean(loss_neg)
            losses.append((loss1+loss2)/2+loss3)
        return sharpe_list,torch.mean(torch.stack(losses))

class GeometricLoss(object):
    def __init__(self, sinkhorn=False):
        super().__init__()
        self.loss = MatchLoss().cuda()
        self.sinkhorn = sinkhorn
        self.bin_score=torch.nn.Parameter(
                torch.tensor(1.0, requires_grad=True)).cuda()
        self.skh_iters=100

    def __call__(self,q,k,labels): # q: src k: aug
        B_q,_,_ = q.shape
        loss = []
        losses = []
        pos_losses = []
        neg_losses = []
        for i in range(B_q):
            label = labels[i]
            all_k = k[i][None,...]
            # sinkhorn
            if self.sinkhorn:
                sim_matrix = torch.einsum("nlc,nsc->nls", q[i][None,...], all_k)
                log_assign_matrix = log_optimal_transport(sim_matrix,self.bin_score,self.skh_iters)
                assign_matrix = log_assign_matrix.exp()[:,:-1,:-1]
                gt = torch.zeros(assign_matrix.shape).cuda()
                gt[0,:,:]=label
                loss,pos_loss,neg_loss = self.loss(assign_matrix, gt)
            else:
                # dual_softmax
                sim_matrix = torch.einsum("nlc,nsc->nls", q[i][None,...], all_k)
                conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
                gt = torch.zeros(conf_matrix.shape).cuda()
                gt[0,:,:]=label
                loss,pos_loss,neg_loss = self.loss(conf_matrix, gt)
            losses.append(loss)
            pos_losses.append(pos_loss)
            neg_losses.append(neg_loss)
        return torch.mean(torch.stack(losses)),torch.mean(torch.stack(pos_losses)),torch.mean(torch.stack(neg_losses))

class GeCoLoss(torch.nn.Module):
    def __init__(self,weight_p2p=0.5,weight_o2a=0.5,weight_global=0.5,sinkhorn=False):
        super().__init__()
        self.contrast_loss = GeCoContrast()
        self.geco_loss = GeometricLoss(sinkhorn=sinkhorn)
        self.sharp_loss = SharpeLoss().cuda()
        self.weight_p2p = weight_p2p
        self.weight_p2s = weight_p2s
        self.weight_global = weight_global
    def __call__(self,q_geo,q_cl,geo_pos=None,cl_pos=None,que_k_geo=None,que_k_cl=None,label=None,cl_pos_save=None,step=0):
        if geo_pos !=None:
            loss_geo,geo_pos_loss,geo_neg_loss = self.geco_loss(q_geo,geo_pos,label)#src:q_geo x aug:geo_pos
            sharpe,loss_sharp = self.sharp_loss(q_geo,geo_pos,label,step)
            loss_geo = self.weight_p2p*loss_geo+self.weight_p2s*loss_sharp
        else:
            loss_geo = torch.tensor(0).cuda()
            geo_pos_loss,geo_neg_loss,loss_sharp=loss_geo,loss_geo,loss_geo
        if cl_pos!=None:
            N,P,_ = q_cl.shape
            idx = random.sample(range(P),N)
            
            q_cl = torch.cat([q_cl[i,j,...][None,...] for i,j in zip(range(N),idx)])
            
            if que_k_cl is not None:
                cl_pos_save = torch.cat([cl_pos_save[i,j,...][None,...] for i,j in zip(range(N),idx)])
                for i,j in zip(range(N),idx):
                    tmp = cl_pos[i,0,:].clone()
                    cl_pos[i,0,:] = cl_pos[i,j,:]
                    cl_pos[i,j,:] = tmp
                # cl_pos = torch.cat([cl_pos[i,j,...][None,...] for i,j in zip(range(N),idx)])
                cl_pos = torch.cat([cl_pos[i,0,...][None,...] for i in range(N)]+[cl_pos[i,1:,...] for i in range(N)])
                cl_pos = torch.cat([cl_pos,que_k_cl])
            else:
                for i,j in zip(range(N),idx):
                    tmp = cl_pos[i,0,:].clone()
                    cl_pos[i,0,:] = cl_pos[i,j,:]
                    cl_pos[i,j,:] = tmp
                cl_pos_save = torch.cat([cl_pos[i,0,...][None,...] for i in range(N)])
                cl_pos = torch.cat([cl_pos[i,0,...][None,...] for i in range(N)]+[cl_pos[i,1:,...] for i in range(N)])
            loss_cl = self.contrast_loss(q_cl,cl_pos)
        else:
            loss_cl = torch.tensor(0).cuda()
            cl_pos_save = torch.tensor(0).cuda()
        loss_cl = loss_cl*self.weight_global
        return  sharpe,cl_pos_save,loss_geo+loss_cl,geo_pos_loss,geo_neg_loss,loss_sharp,loss_cl