# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import scipy.ndimage as ndimage
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


# def dice(x, y): #output,label
#     intersect = np.sum(np.sum(np.sum(x * y)))
#     y_sum = np.sum(np.sum(np.sum(y)))
#     x_sum = np.sum(np.sum(np.sum(x)))
#     if x_sum == 0 and y_sum==0:
#         return None
#     if x_sum == 0 and y_sum != 0:
#         return 0.0
#     if y_sum == 0:
#         return 2 * intersect / (x_sum + y_sum)
#     if y_sum != 0:
#         return 2 * intersect / (x_sum + y_sum)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out

def draw_mask(image,mask,slice_idx,save_path):
    # slice_idx = image.shape[2] // 2  # 选择中间切片
    image_slice = image[:, :,slice_idx]
    mask_slice = mask[:,:,slice_idx]
    # print(np.unique(mask_slice))
    foreground_mask = mask_slice > 0
    # Step 3: 创建颜色映射
    # 定义掩码颜色（例如：红色）
    cmap = ListedColormap(['b', 'g','r','c','k','limegreen','teal','plum','xkcd:sky blue','y','m','w','xkcd:orange','saddlebrown'])

    # Step 4: 可视化叠加图像
    plt.figure(figsize=(10, 10))

    # 显示原始图像
    plt.imshow(image_slice, cmap="gray", interpolation="none")

    # 叠加分割掩码
    # plt.imshow(mask_slice, cmap=cmap, alpha=0.5, interpolation="none")  # alpha控制透明度

    plt.imshow(np.ma.masked_where(~foreground_mask, mask_slice), 
           cmap=cmap, alpha=0.2, interpolation="none")
    
    plt.imshow(np.ma.masked_where(mask_slice == 0, mask_slice), cmap=cmap, alpha=0.3, interpolation="none")

    # 添加标题和坐标轴
    plt.title("Image with Segmentation Mask")
    plt.axis("off")

    # 显示结果
    plt.show()
    plt.savefig(save_path)


def mask_max(mask):
    label_count_per_slice = []
    for slice_idx in range(mask.shape[2]):
        unique_labels = np.unique(mask[:, :, slice_idx])  
        label_count_per_slice.append(len(unique_labels))  
    max_labels = max(label_count_per_slice)  
    max_layer_idx = label_count_per_slice.index(max_labels)
    return max_layer_idx
