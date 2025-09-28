import argparse
import os
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from models.s2dc_head import S2DCTokenHead
from optimizers.lr_scheduler import WarmupCosineSchedule
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from utils.data_utils import get_1k_loader, get_mri_loader, get_pet_loader,get_10k_loader
from utils.ops import aug_rand, rot_rand, monai_aug,img_monai_aug
from monai.utils import set_determinism
from utils.visualization import tsne_visual
from utils.util import AverageMeter, distributed_all_gather
from tqdm import tqdm
from utils.ops import *
import matplotlib.pyplot as plt
def main():
    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    def train(args, global_step, train_loader, val_best, scaler):
        img_transforms = monai_aug(args)
        model.train()
        run_loss = AverageMeter()
        pos_avg, neg_avg, sharp_avg, cl_avg = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        for step, batch in enumerate(train_loader):
            t1 = time()
            src_crops,crops_full_img_aug,crops_aug,conf_matrix_gt = batch
            src_crops,crops_full_img_aug,crops_aug = concat_image(src_crops), concat_image(crops_full_img_aug),concat_image(crops_aug)
            src_crops,crops_full_img_aug,crops_aug = src_crops.cuda(),crops_full_img_aug.cuda(),crops_aug.cuda()
            loss,geo_pos_loss,geo_neg_loss,loss_sharp,cl_loss = model({'src':src_crops,'aug_full':crops_full_img_aug,'aug_crop':crops_aug,'gt':conf_matrix_gt})
            

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.lrdecay:
                scheduler.step()
            optimizer.zero_grad()

            run_loss.update(loss.item(), n=args.batch_size)
            pos_avg.update(geo_pos_loss.item(), n=args.batch_size)
            neg_avg.update(geo_neg_loss.item(), n=args.batch_size)
            sharp_avg.update(loss_sharp.item(), n=args.batch_size)
            cl_avg.update(cl_loss.item(), n=args.batch_size)
            lr = optimizer.param_groups[0]["lr"]

            if args.distributed:
                if dist.get_rank() == 0:
                    print("Step:{}/{}, Loss:{:.4f}, pos:{:.4f}, neg:{:.4f}, sharp: {:.4f}, cl:{:.4f} "
                      "lr:{:.8f}, Time:{:.4f}".format(global_step, args.num_steps,
                                                               run_loss.avg, pos_avg.avg, neg_avg.avg,sharp_avg.avg,cl_avg.avg,
                                                               lr, time() - t1))
                    if global_step % 100 == 0:
                        writer.add_scalar("train/loss_total", scalar_value=run_loss.avg, global_step=global_step)
                        writer.add_scalar("train/loss_sharp", scalar_value=sharp_avg.avg, global_step=global_step)
                        writer.add_scalar("train/loss_pos", scalar_value=pos_avg.avg, global_step=global_step)
                        writer.add_scalar("train/loss_neg", scalar_value=neg_avg.avg, global_step=global_step)
                        writer.add_scalar("train/loss_cl", scalar_value=cl_avg.avg, global_step=global_step)
            else:
                print("Step:{}/{}, Loss:{:.4f}, pos:{:.4f}, neg:{:.4f},sharp:{:.4f}, cl:{:.4f} "
                      "lr:{:.8f}, Time:{:.4f}".format(global_step, args.num_steps,
                                                               run_loss.avg, pos_avg.avg, neg_avg.avg,sharp_avg.avg,cl_avg.avg,
                                                               lr, time() - t1))
            

            if global_step % args.eval_num == 0:
                torch.cuda.empty_cache()

            global_step += 1
            if args.distributed:
                val_cond = (dist.get_rank() == 0) and (global_step % args.eval_num == 0)
            else:
                val_cond = global_step % args.eval_num == 0

            if val_cond:
                checkpoint = {
                        "global_step": global_step,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                save_ckp(checkpoint, logdir + f"/model_step{str(global_step)}.pt")
                print('saving checkpoint')
                model.train()
                writer.add_scalar("train/loss_total", scalar_value=run_loss.avg, global_step=global_step)
                writer.add_scalar("train/loss_sharp", scalar_value=sharp_avg.avg, global_step=global_step)
                writer.add_scalar("train/loss_pos", scalar_value=pos_avg.avg, global_step=global_step)
                writer.add_scalar("train/loss_neg", scalar_value=neg_avg.avg, global_step=global_step)
                writer.add_scalar("train/loss_cl", scalar_value=cl_avg.avg, global_step=global_step)
        return global_step, loss


    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
    parser.add_argument("--num_steps", default=100000, type=int, help="number of training iterations")
    parser.add_argument("--adversarial", action="store_true", help="adversarial loss")
    parser.add_argument("--visual_steps", default=100, type=int, help="number of training iterations")
    parser.add_argument("--eval_num", default=100, type=int, help="evaluation frequency")
    parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps")
    parser.add_argument("--feature_dim", default=512, type=int, help="warmup steps")
    parser.add_argument("--num_domains", default=2, type=int, help="domain numbers")
    parser.add_argument("--queue_num", default=200, type=int, help="domain numbers")
    parser.add_argument("--crop_foreground", action="store_true", help="use monai Dataset class")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--modality", default="PET_CT", type=str, help="PET/CT/PET_CT")
    parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--data_type", default="data_1k", type=str,)
    parser.add_argument("--use_last_layer", action="store_true")
    parser.add_argument("--use_geo", action="store_true")
    parser.add_argument("--use_cl", action="store_true")
    parser.add_argument("--use_sharp", action="store_true")
    parser.add_argument("--sinkhorn", action="store_true")
    parser.add_argument("--random_seed", default=20, type=int, help="random seed")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--a_min", default=-1000, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=1000, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=1.5, type=float, help="spacing in z direction")
    parser.add_argument("--weight_p2s", default=0.5, type=float, help="")
    parser.add_argument("--weight_p2p", default=0.5, type=float, help="")
    parser.add_argument("--weight_global", default=0.5, type=float, help="")
    parser.add_argument("--roi_large", default=288, type=int, help="roi size in x direction")
    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
    parser.add_argument("--num_geo_layer", default=2, type=int, help="which layer to apply ugco")
    parser.add_argument("--lr", default=4e-4, type=float, help="learning rate")
    parser.add_argument("--decay", default=0.1, type=float, help="decay rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--lrdecay", action="store_true", help="enable learning rate decay")
    parser.add_argument("--token_head", action="store_true", help="without teacher")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="maximum gradient norm")
    parser.add_argument("--loss_type", default="SSL", type=str)
    parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--resume", default=None, type=str, help="resume training")
    parser.add_argument("--local-rank", type=int, default=0, help="local rank")
    parser.add_argument("--grad_clip", action="store_true", help="gradient clip")
    parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--smartcache_dataset", action="store_true", help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")
    parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
    parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")

    args = parser.parse_args()
    logdir = args.logdir
    args.amp = not args.noamp
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        print(f'WORLD_SIZE {int(os.environ["WORLD_SIZE"])}')
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
        args.wdsize = int(os.environ["WORLD_SIZE"])
    args.epochs = args.num_steps / (args.batch_size*args.wdsize)
    args.device = "cuda:0"
    args.world_size = 1
    args.rank = 0

    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        print("Training with a single process on 1 GPUs.")
    assert args.rank >= 0

    if args.rank == 0:
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(logdir)
    else:
        writer = None


    random_seed = args.random_seed
    if random_seed is not None and (isinstance(random_seed, int) or isinstance(random_seed, float)):
        set_determinism(seed=random_seed)
    
    model = S2DCTokenHead(args,exp=args.queue_num,num_patch_side=args.roi_large//args.roi_z)
   
    model.cuda()

    if args.opt == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    global_step = 0
    if args.resume:
        model_pth = args.resume
        model_dict = torch.load(model_pth)
        state_dict = model_dict["state_dict"]
        if "module." in list(state_dict.keys())[0]:
            print("Tag 'module.' found in state dict - fixing!")
            for key in list(state_dict.keys()):
                state_dict[key.replace("module.", "")] = state_dict.pop(key)
        model.load_state_dict(state_dict)
        global_step = model_dict['global_step']

    if args.lrdecay:
        if args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)

        elif args.lr_schedule == "poly":

            def lambdas(step):
                return (1 - float(step/(args.batch_size*args.wdsize)) / float(args.epochs)) ** 0.9

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
        else:
            scheduler = None
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[args.local_rank],find_unused_parameters=True)
    
    if args.data_type == 'data_1k':
        train_loader = get_1k_loader(args)
    elif args.data_type == 'mri':
        train_loader = get_mri_loader(args)
    elif args.data_type == 'data_10k':
        train_loader = get_10k_loader(args)
    elif args.data_type == 'pet':
        train_loader = get_pet_loader(args)
    else:
        raise TypeError('not implemented')

    
    best_val = 1e8
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None
    while global_step < args.num_steps:
        global_step, loss = train(args, global_step, train_loader, best_val, scaler)
    checkpoint = {"epoch": args.epochs, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}

    if args.distributed:
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), logdir + "final_model.pth")
        dist.destroy_process_group()
    else:
        torch.save(model.state_dict(), logdir + "/final_model.pth")
    save_ckp(checkpoint, logdir + "/model_final_epoch.pt")


if __name__ == "__main__":
    main()
