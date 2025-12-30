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

import argparse
import os
from functools import partial
import nibabel as nib
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from utils.data_utils import get_loader
from utils.utils import dice, resample_3d
from utils.utils import AverageMeter, distributed_all_gather

from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
import torch.nn.functional as F

def resize(img):
    size = 256
    b, _, c, h, w = img.size()
    new_img = []
    for i in range(b):
        im = img[i, :, :, :, :]
        im = F.interpolate(im, size=[size, size], mode='bilinear', align_corners=True)
        new_img.append(im.unsqueeze(0))
    new_img = torch.cat(new_img, dim=0)
    return new_img

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '28890'

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="logs", type=str, help="directory to save the tensorboard logs")
parser.add_argument(
    "--pretrained_dir", default="./runs/logs_384/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--csv_list", default="./csv/", type=str, help="csv directory")
parser.add_argument("--fold", default=0, type=int, help="fold")
parser.add_argument("--data_dir", default="/data/jiaxin/data/CC-CCII_public/data/", type=str, help="dataset directory")
parser.add_argument(
    "--pretrained_model_name",
    default="model.pt",
    type=str,
    help="pretrained model name",
)

parser.add_argument("--save_checkpoint", default=True, help="save checkpoint during training")
parser.add_argument("--max_epochs", default=100, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=4, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=5, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=4, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
# warmup is important !!!
parser.add_argument("--warmup_epochs", default=5, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")
parser.add_argument("--use_ssl_pretrained", default=True, help="use self-supervised pretrained weights")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")


def main():
    args = parser.parse_args()
    args.test_mode = True
    train_loader,val_loader = get_loader(args)

    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_pth = os.path.join(pretrained_dir, model_name)
    from model import Swin
    model = Swin(args)

    # model_dict = torch.load(pretrained_pth)["state_dict"]
    # model.load_state_dict(model_dict, strict=True)

    ####
    model_dict = torch.load(pretrained_pth)
    print(f"load pretrained model from {pretrained_pth}")
    if "state_dict" in model_dict.keys():
        state_dict = model_dict["state_dict"]
    else:
        state_dict = model_dict
    if "module." in list(state_dict.keys())[0]:
        print("Tag 'module.' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("module.", "")] = state_dict.pop(key)

    for key in list(state_dict.keys()):
        if "teacher" in key:
            state_dict.pop(key)
        else:
            if 'student' in key:
                state_dict[key.replace("student.", "")] = state_dict.pop(key)
            
    if "swin_vit" in list(state_dict.keys())[0]:
        print("Tag 'swin_vit' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
    # We now load model weights, setting param `strict` to False, i.e.:
    # this load the encoder weights (Swin-ViT, SSL pre-trained), but leaves
    # the decoder weights untouched (CNN UNet decoder).
    model.load_state_dict(state_dict, strict=False)
    print("Using pretrained self-supervised Swin UNETR backbone weights !")

    model.eval()
    model.to(device)
    labels = []
    features = []
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        loader = train_loader
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]

            data = resize(data)
            data, target = data.cuda(args.rank), target.cuda(args.rank)

            logits,feature = model(data,training=False)
            features.extend(feature.detach().cpu().numpy())
            labels.extend(target.detach().cpu().numpy())
            value = torch.eq(logits.argmax(dim=1), target)

            metric_count += len(value)
            num_correct += value.sum().item()

            metric = num_correct / metric_count
            print(
                "Val {}/{}".format(idx, len(loader)),
                "acc",
                metric,
            )
    features = np.stack(features)
    labels = np.stack(labels)
    np.save('feature_label/feature_ugco98k.npy',features)
    np.save('feature_label/labels_ugco98k.npy',labels)

def classifier(X_train,X_test,y_train,y_test):
    from sklearn import svm
    from sklearn.metrics import accuracy_score
    svm_classifier = svm.SVC(kernel='linear', C=1.0, random_state=42)
    svm_classifier.fit(X_train, y_train)

    # 对测试集进行预测
    y_pred = svm_classifier.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    import numpy as np
    from utils.visualization import SE_visual,pcatsne3d_visual,pcatsne2d_visual
    from monai.utils import set_determinism
    from sklearn.preprocessing import normalize
    set_determinism(seed=20)
    wolabel=1
    import random
    for model_name in ['ugco98k','voco','swinunetr']:
        print(model_name)
        X_train = normalize(np.load(f'feature_label/feature_{model_name}.npy'),norm='l2',axis=1)
        X_test = normalize(np.load(f'feature_label/feature_{model_name}_val.npy'),norm='l2',axis=1)
        y_train = np.load(f'feature_label/labels_{model_name}.npy')
        y_test = np.load(f'feature_label/labels_{model_name}_val.npy')
        sample_index = random.sample(range(0,len(X_train)), int(len(X_train)))
        classifier(X_train[sample_index],X_test,y_train[sample_index],y_test)
        # feature = normalize(np.load(f'feature_label/feature_{model_name}.npy'),norm='l2',axis=1)
        # label = np.load(f'feature_label/labels_{model_name}.npy')
        # pcatsne2d_visual(feature, label, f'feature_label/{model_name}_pca.png',wolabel=wolabel,method='pca')
        # pcatsne2d_visual(feature, label, f'feature_label/{model_name}_tsne.png',wolabel=wolabel,method='tsne')
        # pcatsne2d_visual(feature, label, f'feature_label/{model_name}_isomap.png',wolabel=wolabel,method='isomap')
        # pcatsne2d_visual(feature, label, f'feature_label/{model_name}_llm.png',wolabel=wolabel,method='llm')
        # pcatsne2d_visual(feature, label, f'feature_label/{model_name}_umap.png',wolabel=wolabel,method='umap')
        # SE_visual(feature, label, f'feature_label/{model_name}_se.png',wolabel=wolabel)
    # main()
