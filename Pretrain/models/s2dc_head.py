import torch
import torch.nn as nn
import numpy as np
from monai.networks.nets.swin_unetr import *
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep
import argparse
import torch.nn.functional as F
from utils.superglue import log_optimal_transport
from losses.coderating import MaximalCodingRateReduction
from losses.loss import GeCoLoss
from einops import rearrange
from models.modules import CrossBlock,SelfBlock,TransformerLayer,LearnableFourierPositionalEncoding, normalize_keypoints
import matplotlib.pyplot as plt
from utils.visualization import pca_1d_visual


class projection_head(nn.Module):
    def __init__(self, in_dim:int=768, hidden_dim:int=2048, out_dim:int=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x): 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class Swin(nn.Module):
    def __init__(self, args,geco=True,cl=True):
        super(Swin, self).__init__()
        patch_size = ensure_tuple_rep(2, args.spatial_dims)
        window_size = ensure_tuple_rep(7, args.spatial_dims)
        self.swinViT = SwinViT(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
            use_v2=True
        )
        norm_name = 'instance'
        self.use_last_layer = args.use_last_layer
        
        self.geco=geco
        self.cl=cl
        self.geo_layer = args.num_geo_layer
        self.token_num = [48,96,192,384,768]
        if self.use_last_layer:
            in_dim = self.token_num[self.geo_layer]
            # in_dim = 768
            in_dim_cl = 768
        else:
            in_dim = 1152
        if self.geco:
            self.proj_head_geo = projection_head(in_dim=in_dim, hidden_dim=2048, out_dim=512)
        if self.cl:
            self.encoder_cof = UnetrBasicBlock(
                spatial_dims=args.spatial_dims,
                in_channels=16 * args.feature_size,
                out_channels=16 * args.feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=True,
            )
            self.proj_head_cl = projection_head(in_dim=in_dim_cl, hidden_dim=2048, out_dim=512)
            self.avg_pooling = nn.AdaptiveAvgPool3d((1,1,1))
        self.num_crops_side = args.roi_large//args.roi_z
        self.args = args
    
    def forward_encs(self, encs):
        b = encs[0].size()[0]
        outs = []
        for enc in encs:
            out = F.adaptive_avg_pool3d(enc, (1, 1, 1))
            outs.append(out.view(b, -1))
        outs = torch.cat(outs, dim=1)
        return outs

    def forward(self, x_in,visual_flag=False,step=0):
        hidden_states_out = self.swinViT(x_in)
        if not self.use_last_layer:
            enc0 = self.encoder1(x_in) 
            enc2 = self.encoder3(hidden_states_out[1]) # [16, 48, 48, 48, 48] 0
            enc3 = self.encoder4(hidden_states_out[2]) # [16, 96, 24, 24, 24] 1
            dec4 = self.encoder10(hidden_states_out[4]) # [16, 192, 12, 12, 12] 2
            encs = [enc0, enc1, enc2, enc3, dec4] # ([16, 384, 6, 6, 6] 3 ,[16, 768, 3, 3, 3] 4
            out = self.forward_encs(encs)
        ## visualize features
        if visual_flag: # 16,192,12,12,12
            for layer in [2]:
                b,f_d,n_t_s = hidden_states_out[layer].shape[:3]
                features = hidden_states_out[layer]
                features = rearrange(features,'(b c) d j t k->b c j t k d',b=self.args.batch_size) #b, 16,3,3,3,dim
                n_c_s = self.num_crops_side # num crop on side
                features_re = torch.zeros(self.args.batch_size,n_c_s*n_t_s,n_c_s*n_t_s,n_t_s,f_d).cuda() # b,12,12,3,d
            
                for bi in range(self.args.batch_size):
                    for k in range(n_t_s):
                        for i in range(n_c_s):
                            for j in range(n_c_s):
                                features_re[bi,i*n_t_s:(i+1)*n_t_s,j*n_t_s:(j+1)*n_t_s,:,:] = features[bi,i*n_c_s+j,...].contiguous()
                features_re = features_re.contiguous().reshape((self.args.batch_size,-1,f_d))
                pca_1d_visual(features_re.cpu().detach().numpy(),f'{self.args.logdir}/layer{layer}_pca_{step}.nii.gz',dim=f_d)
        ##
        b,f_d,s_t = hidden_states_out[self.geo_layer].shape[:3]
        out_geo = hidden_states_out[self.geo_layer].contiguous().reshape(b,f_d,s_t,s_t,s_t) 
        b,f_d,s_t = hidden_states_out[-1].shape[:3]
        out_cl = hidden_states_out[-1].contiguous().reshape(b,f_d,s_t,s_t,s_t) # 16xB,768, 3, 3, 3
        if self.cl:
            out_cl = self.proj_head_cl(self.avg_pooling(out_cl).reshape(b, -1)).as_tensor()
        else:
            out_cl=None
        ## token feature
        b,f_d,n_t_s = out_geo.shape[:3]
        out_geo = rearrange(out_geo,'(b c) d j t k->b c j t k d',b=self.args.batch_size) #b, 16,3,3,3,dim
        n_c_s = self.num_crops_side # num crop on side
        out_geo_re = torch.zeros(self.args.batch_size,n_c_s*n_t_s,n_c_s*n_t_s,n_t_s,f_d).cuda() # b,12,12,3,d
        
        for bi in range(self.args.batch_size):
            for k in range(n_t_s):
                for i in range(n_c_s):
                    for j in range(n_c_s):
                        out_geo_re[bi,i*n_t_s:(i+1)*n_t_s,j*n_t_s:(j+1)*n_t_s,:,:] = out_geo[bi,i*n_c_s+j,...].contiguous()
        out_geo_re = out_geo_re.reshape((self.args.batch_size,-1,3,f_d))
        ##
        
        if self.geco:
            out_geo_re = torch.stack([F.normalize(self.proj_head_geo(out_geo_re[i,...]), p=2, dim=2) for i in range(self.args.batch_size)])
            # out_geo_re = torch.stack([self.proj_head_geo(out_geo_re[i,...]) for i in range(self.args.batch_size)])
        else:
            out_geo_re = None
        
        return out_geo_re,out_cl

class S2DCTokenHead(nn.Module):
    def __init__(self, args,exp=200,dim=512,num_patch_side=4):
        super(S2DCTokenHead, self).__init__()
        self.geco = args.use_geo
        self.cl = args.use_cl
        self.student = Swin(args,geco=self.geco,cl=self.cl)
        self.teacher = Swin(args,geco=self.geco,cl=self.cl)
        self.dim=dim
        self.args = args
        self.criterion = GeCoLoss(args.weight_p2p,args.weight_p2s,args.weight_global,sinkhorn=args.sinkhorn)
        self.K = self.args.batch_size*exp
        self.num_crops = int((self.args.roi_large//self.args.roi_x)**2)
        num_layer_token_side = [48,24,12,6,3]
        self.num_token_side = num_layer_token_side[args.num_geo_layer]
        if self.cl:
            self.register_buffer("queue_cl", torch.randn(self.K,dim))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.queue_cl = nn.functional.normalize(self.queue_cl, dim=1)

        for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  
        
    @torch.no_grad()
    def _EMA_update_encoder_teacher(self):
        ## no scheduler here
        momentum = 0.999
        for param, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data = momentum * param_t.data + (1. - momentum) * param.data

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):	

        # gather keys before updating queue
        if torch.cuda.device_count() > 1:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_cl[ptr:ptr + batch_size,:] = keys
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

        
    def reshape_sharp(self,sharps:list):
        out_geo_re = torch.zeros(12,12,3)
        for sharp,l in zip(sharps,range(len(sharps))):
            for i in range(12):
                for j in range(12):
                    out_geo_re[i,j,l]=sharp[i+j*12]
        return out_geo_re

    def forward(self, img:dict,visual_flag:bool=False,step:int=0):
        N = self.args.batch_size
        _,_,h,w,d = img['src'].shape
        src_crops = img['src']
        crops_full_img_aug = img['aug_full']
        crops_aug = img['aug_crop']
        conf_matrix_gt= img['gt']
        q_geo,q_cl = self.student(src_crops,visual_flag,step)
        if self.geco:
            q_geo = torch.transpose(q_geo,1,2).contiguous().reshape(self.args.batch_size*self.num_token_side,-1,q_geo.shape[-1])
        if self.cl:
            q_cl = torch.stack([q_cl[i*self.num_crops:(i+1)*self.num_crops] for i in range(N)])

        self.training = True
        if self.training:
            with torch.no_grad():
                self._EMA_update_encoder_teacher()
                geo_pos,_ = self.teacher(crops_full_img_aug)
                if self.cl:
                    _, cl_pos = self.teacher(crops_aug)
                    _, cl_pos_save = self.teacher(src_crops)
                    cl_pos = torch.stack([cl_pos[i*self.num_crops:(i+1)*self.num_crops] for i in range(N)])
                    cl_pos = cl_pos.detach()
                    cl_pos_save = torch.stack([cl_pos_save[i*self.num_crops:(i+1)*self.num_crops] for i in range(N)])
                    cl_pos_save = cl_pos_save.detach()
                else:
                    cl_pos = None
                if self.geco:
                    geo_pos = torch.transpose(geo_pos,1,2).contiguous().reshape(self.args.batch_size*self.num_token_side,-1,geo_pos.shape[-1])
                    conf_matrix_gt = torch.cat([torch.stack([conf_matrix_gt[i,...]]*self.num_token_side) for i in range(self.args.batch_size)])#

        else:
            geo_pos_save,_ = self.student(src_crops)
            geo_pos,_ = self.student(crops_full_img_aug)
            geo_pos = torch.transpose(geo_pos,1,2).contiguous().reshape(self.args.batch_size*self.num_token_side,-1,geo_pos.shape[-1])
            conf_matrix_gt = conf_matrix_gt.contiguous().expand(self.args.batch_size*self.num_token_side,conf_matrix_gt.shape[1],conf_matrix_gt.shape[2])
    
        if self.cl:
            que_k_cl = self.queue_cl.clone().detach()
        else:
            que_k_cl = None
            cl_pos_save=None
        # que_k_cl = None
        sharpness,cl_pos_save,loss,geo_pos_loss,geo_neg_loss,loss_sharp,cl_loss = self.criterion(q_geo,q_cl,geo_pos=geo_pos,cl_pos=cl_pos,label=conf_matrix_gt,que_k_cl=que_k_cl,cl_pos_save=cl_pos_save,step=step)
        if visual_flag:
            sharpness = self.reshape_sharp(sharpness)
            return sharpness
        if self.cl and not visual_flag:
            self._dequeue_and_enqueue(cl_pos_save)
        return loss,geo_pos_loss,geo_neg_loss,loss_sharp,cl_loss

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
