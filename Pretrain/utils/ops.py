import numpy as np
import torch
from numpy.random import randint
from monai.transforms import (
    Compose,
    CropForeground,
    NormalizeIntensity,
    Orientation,
    RandRotate90,
    RandSpatialCropSamples,
    RandAdjustContrast,
    ScaleIntensityRange,
    RandGaussianNoise,
    RandShiftIntensity,
    SpatialPad,
    ToTensor,
)

def patch_rand_drop(args, x, x_rep=None, max_drop=0.3, max_block_sz=0.25, tolr=0.05):
    c, h, w, z = x.size()
    n_drop_pix = np.random.uniform(0, max_drop) * h * w * z
    mx_blk_height = int(h * max_block_sz)
    mx_blk_width = int(w * max_block_sz)
    mx_blk_slices = int(z * max_block_sz)
    tolr = (int(tolr * h), int(tolr * w), int(tolr * z))
    total_pix = 0
    while total_pix < n_drop_pix:
        rnd_r = randint(0, h - tolr[0])
        rnd_c = randint(0, w - tolr[1])
        rnd_s = randint(0, z - tolr[2])
        rnd_h = min(randint(tolr[0], mx_blk_height) + rnd_r, h)
        rnd_w = min(randint(tolr[1], mx_blk_width) + rnd_c, w)
        rnd_z = min(randint(tolr[2], mx_blk_slices) + rnd_s, z)
        if x_rep is None:
            x_uninitialized = torch.empty(
                (c, rnd_h - rnd_r, rnd_w - rnd_c, rnd_z - rnd_s), dtype=x.dtype, device=args.local_rank
            ).normal_()
            x_uninitialized = (x_uninitialized - torch.min(x_uninitialized)) / (
                torch.max(x_uninitialized) - torch.min(x_uninitialized)
            )
            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_uninitialized
        else:
            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_rep[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z]
        total_pix = total_pix + (rnd_h - rnd_r) * (rnd_w - rnd_c) * (rnd_z - rnd_s)
    return x


def rot_rand(args, x_s):
    img_n = x_s.size()[0]
    x_aug = x_s.detach().clone()
    device = torch.device(f"cuda:{args.local_rank}")
    x_rot = torch.zeros(img_n).long().to(device)
    for i in range(img_n):
        x = x_s[i]
        orientation = np.random.randint(0, 4)
        if orientation == 0:
            pass
        elif orientation == 1:
            x = x.rot90(1, (2, 3))
        elif orientation == 2:
            x = x.rot90(2, (2, 3))
        elif orientation == 3:
            x = x.rot90(3, (2, 3))
        x_aug[i] = x 
        x_rot[i] = orientation
    return x_aug, x_rot


def aug_rand(args, samples):
    img_n = samples.size()[0]
    x_aug = samples.detach().clone()
    for i in range(img_n):
        x_aug[i] = patch_rand_drop(args, x_aug[i])
        idx_rnd = randint(0, img_n)
        if idx_rnd != i:
            x_aug[i] = patch_rand_drop(args, x_aug[i], x_aug[idx_rnd])
    return x_aug

def monai_aug(args):
    transforms_list = []
    transforms_list+=[
            RandGaussianNoise(prob=0.1),
            RandRotate90(prob=0.1,max_k=1,spatial_axes=(0, 1)),
            RandRotate90(prob=0.1,max_k=1,spatial_axes=(1,2)),
            RandRotate90(prob=0.1,max_k=1,spatial_axes=(0,2)),
            RandShiftIntensity(prob=0.1,offsets=0.2),
            RandAdjustContrast(prob=0.1),
            RandSpatialCropSamples(
                roi_size=[args.roi_x,args.roi_y,args.roi_z],
                num_samples=1,
                random_center=True,
                random_size=False,
            ),
            SpatialPad(spatial_size=[args.roi_x,args.roi_y,args.roi_z], method='symmetric', mode='constant'),
            ToTensor()]
    img_transforms = Compose(transforms_list)
    return img_transforms

def img_monai_aug(img_transform,imgs):
    aug_x = []
    for i in range(imgs.shape[0]):
        aug_x.extend(img_transform(imgs[i,...]))
    return torch.stack(aug_x, dim=0)

def concat_image(imgs):
    output = []
    for img in imgs:
        img = img['image']
        output.append(img)
    output = torch.concatenate(output, dim=1)
    bs, sw_s, x, y, z = output.size()
    output = output.view(-1, 1, x, y, z)
    return output
