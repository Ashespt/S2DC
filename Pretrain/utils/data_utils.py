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

from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, SmartCacheDataset, load_decathlon_datalist,PersistentDataset
import monai.transforms as transforms
import monai
import torch
import math
from monai.transforms import *

import os
import numpy as np
import pickle


class MaskLabel():
    def __init__(self):
        return

    def __call__(self, x_in):
        image = x_in['image']
        image[image>0] = 1
        x_in['mask_label'] = image
        return x_in

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_infer_loader(args):
    splits1 = "/btcv.json"
    splits2 = "/dataset_TCIAcovid19_0.json"
    splits3 = "/dataset_LUNA16_0.json"
    
    list_dir = "./jsons_geco"
    jsonlist1 = list_dir + splits1
    jsonlist2 = list_dir + splits2
    jsonlist3 = list_dir + splits3
    
    datadir1 = "./data/BTCV"
    datadir2 = "./data/TCIAcovid19"
    datadir3 = "./data/Luna16-jx-part/"
    num_workers = 0
    datalist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
    print("Dataset 1 BTCV: number of data: {}".format(len(datalist1)))
    new_datalist1 = []
    for item in datalist1:
        item_dict = {"image": item["image"]}
        new_datalist1.append(item_dict)
    datalist2 = load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir2)
    print("Dataset 2 Covid 19: number of data: {}".format(len(datalist2)))
    datalist = new_datalist1 + datalist2
    print("Dataset all training: number of data: {}".format(len(datalist)))
    if args.token_head:
        from .data_check2 import GeoAugmentation
    else:
        from .data_check import GeoAugmentation
    
    train_transforms = Compose([LoadImaged(keys=["image"], image_only=True),
                                CropForegroundd(keys="image", source_key="image", select_fn=threshold_infer),
                                CenterSpatialCropd(
                                    keys=['image'],roi_size=(args.roi_large, args.roi_large, args.roi_z)
                                ),
                                Resized(keys=["image"], mode="trilinear", align_corners=True,
                                        spatial_size=(args.roi_large, args.roi_large, args.roi_z)),
                                GeoAugmentation(args,roi_large=args.roi_large,crop_size=args.roi_z)
                                ])

    train_ds = monai.data.Dataset(data=datalist, transform=train_transforms)

    if args.distributed:
        train_sampler = Sampler(train_ds) if args.distributed else None
    else:
        train_sampler = None
    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=0,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True
    )

    return train_loader

def get_1k_loader(args):
    splits1 = "/btcv.json"
    splits2 = "/dataset_TCIAcovid19_0.json"
    # splits3 = "/dataset_LUNA16_0.json" # for 1.6k data
    
    list_dir = "./jsons_geco"
    jsonlist1 = list_dir + splits1
    jsonlist2 = list_dir + splits2
    # jsonlist3 = list_dir + splits3
    
    datadir1 = "./data/BTCV"
    datadir2 = "./data/TCIAcovid19"
    # datadir3 = "./data/Luna16-jx-part/"
    num_workers = 0
    datalist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
    print("Dataset 1 BTCV: number of data: {}".format(len(datalist1)))
    new_datalist1 = []
    for item in datalist1:
        item_dict = {"image": item["image"]}
        new_datalist1.append(item_dict)
    datalist2 = load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir2)
    print("Dataset 2 Covid 19: number of data: {}".format(len(datalist2)))
    # datalist3 = load_decathlon_datalist(jsonlist3, False, "training", base_dir=datadir3)
    # print("Dataset 3 LUNA: number of data: {}".format(len(datalist3)))
    datalist = new_datalist1 + datalist2
    print("Dataset all training: number of data: {}".format(len(datalist)))
    if args.token_head:
        from .data_check2 import GeoAugmentation
    else:
        from .data_check import GeoAugmentation
    
    train_transforms = Compose([LoadImaged(keys=["image"], image_only=True),
                                AddChanneld(keys=["image"]),
                                Orientationd(keys=["image"], axcodes="RAS"),
                                ScaleIntensityRanged(
                                    keys=["image"], a_min=args.a_min, a_max=args.a_max,
                                    b_min=args.b_min, b_max=args.b_max, clip=True),
                                CropForegroundd(keys="image", source_key="image", select_fn=threshold),
                                RandSpatialCropSamplesd(
                                    keys=["image"],
                                    roi_size=[args.roi_large, args.roi_large, args.roi_z],
                                    num_samples=1,
                                    random_center=True,
                                    random_size=False,
                                ),
                                Resized(keys=["image"], mode="trilinear", align_corners=True,
                                        spatial_size=(args.roi_large, args.roi_large, args.roi_z)),
                                GeoAugmentation(args,roi_large=args.roi_large,crop_size=args.roi_z)
                                ])

    train_ds = monai.data.Dataset(data=datalist, transform=train_transforms)

    if args.distributed:
        # train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
        train_sampler = Sampler(train_ds) if args.distributed else None
    else:
        train_sampler = None

    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=0,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True
    )

    return train_loader

def get_10k_loader(args):
    splits1 = "/btcv.json"
    splits2 = "/dataset_TCIAcovid19_0.json"
    splits3 = "/dataset_LUNA16_0.json"
    splits4 = "/stoic21.json"
    splits5 = "/Totalsegmentator_dataset.json"
    splits6 = "/flare23.json"
    splits7 = "/HNSCC.json"
    splits8 = "/dataset_LIDC_0.json"
    
    list_dir = "./jsons_geco"
    jsonlist1 = list_dir + splits1
    jsonlist2 = list_dir + splits2
    jsonlist3 = list_dir + splits3
    jsonlist4 = list_dir + splits4
    jsonlist5 = list_dir + splits5
    jsonlist6 = list_dir + splits6
    jsonlist7 = list_dir + splits7
    jsonlist8 = list_dir + splits8
    
    datadir1 = "./data/BTCV"
    datadir2 = "./data/TCIAcovid19"
    datadir3 = "./data/Luna16-jx-part/"
    datadir4 = "./data/stoic21/"
    datadir5 = "./data/Totalsegmentator_dataset/"
    datadir6 = "./data/Flare23/"
    datadir7 = "./data/HNSCC_convert_v1/"
    datadir8 = "./data/LIDC_convert_v1/"

    num_workers = 0
    datalist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
    datalist2 = load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir2)
    datalist3 = load_decathlon_datalist(jsonlist3, False, "training", base_dir=datadir3)
    datalist4 = load_decathlon_datalist(jsonlist4, False, "training", base_dir=datadir4)
    datalist5 = load_decathlon_datalist(jsonlist5, False, "training", base_dir=datadir5)
    datalist6 = load_decathlon_datalist(jsonlist6, False, "training", base_dir=datadir6)
    datalist7 = load_decathlon_datalist(jsonlist7, False, "training", base_dir=datadir7)
    datalist8 = load_decathlon_datalist(jsonlist8, False, "training", base_dir=datadir8)
    
    datalist = datalist1 + datalist2 + datalist3 + datalist4 + datalist5 + datalist6 + datalist7 + datalist8
    print("Dataset all training: number of data: {}".format(len(datalist)))
    if args.token_head:
        from .data_check2 import GeoAugmentation
    else:
        from .data_check import GeoAugmentation
    
    train_transforms = Compose([LoadImaged(keys=["image"], image_only=True),
                                AddChanneld(keys=["image"]),
                                Orientationd(keys=["image"], axcodes="RAS"),
                                ScaleIntensityRanged(
                                    keys=["image"], a_min=args.a_min, a_max=args.a_max,
                                    b_min=args.b_min, b_max=args.b_max, clip=True),
                                CropForegroundd(keys="image", source_key="image", select_fn=threshold),
                                RandSpatialCropSamplesd(
                                    keys=["image"],
                                    roi_size=[args.roi_large, args.roi_large, args.roi_z],
                                    num_samples=1,
                                    random_center=True,
                                    random_size=False,
                                ),
                                Resized(keys=["image"], mode="trilinear", align_corners=True,
                                        spatial_size=(args.roi_large, args.roi_large, args.roi_z)),
                                GeoAugmentation(args,roi_large=args.roi_large,crop_size=args.roi_z)
                                ])

    train_ds = monai.data.Dataset(data=datalist, transform=train_transforms)

    if args.distributed:
        # train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
        train_sampler = Sampler(train_ds) if args.distributed else None
    else:
        train_sampler = None

    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=2,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True
    )

    return train_loader


def get_mri_loader(args,training=True):
    data_dir = './'
    datalist_json = './jsons/mri_oasis3_clean.json'
    all_keys = ['image']

    modal_keys = ['image']
    from .data_check2 import GeoAugmentation

    train_transform_list =[transforms.LoadImaged(keys=all_keys)]
    train_transform_list += [
        transforms.EnsureChannelFirstd(keys=['image']),
        Orientationd(keys=all_keys, axcodes="RAS"),
        transforms.CropForegroundd(keys='image',source_key='image'),
        transforms.NormalizeIntensityd(keys=["image"]),
        RandSpatialCropSamplesd(
            keys=["image"],
            roi_size=[args.roi_large, args.roi_large, args.roi_z],
            num_samples=1,
            random_center=True,
            random_size=False,
        ),
        Resized(keys=["image"], mode="trilinear", align_corners=True,
                spatial_size=(args.roi_large, args.roi_large, args.roi_z)),
        GeoAugmentation(args,roi_large=args.roi_large,crop_size=args.roi_z)
        ]
    train_transform = transforms.Compose(train_transform_list)

    val_transform_list = [transforms.LoadImaged(keys=all_keys)]

    val_transform_list += [
        transforms.EnsureChannelFirstd(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        transforms.CropForegroundd(keys=modal_keys,source_key='image'),
        transforms.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CenterSpatialCropd(
            keys=['image'],roi_size=(args.roi_large, args.roi_large, args.roi_z)
        ),
        Resized(keys=["image"], mode="trilinear", align_corners=True,
                spatial_size=(args.roi_large, args.roi_large, args.roi_z)),
        GeoAugmentation(args,roi_large=args.roi_large,crop_size=args.roi_z)]

    val_transform = transforms.Compose(val_transform_list)

    datalist = load_decathlon_datalist(datalist_json, False, "training", base_dir=data_dir)

    train_ds = monai.data.Dataset(data=datalist, transform=train_transform)

    if args.distributed:
        # train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
        train_sampler = Sampler(train_ds) if args.distributed else None
    else:
        train_sampler = None

    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=0,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True
    )
    vallist = load_decathlon_datalist(datalist_json, False, "training", base_dir=data_dir)
    new_vallist = []
    for item in vallist:
        item_name = ''.join(item['image']).split('/')[-1]
        item_dict = {'image': item['image']}
        new_vallist.append(item_dict)
    val_ds = monai.data.Dataset(data=new_vallist, transform=val_transform)
    val_sampler = None
    val_loader = monai.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2, sampler=val_sampler, pin_memory=True
    )
    if not training:
        return val_loader
    else:
        return train_loader


def get_pet_loader(args,training=True):
    data_dir = './jsons'
    datalist_json = os.path.join(data_dir, 'pet_pretrain_udpet_adni.json')
    from .data_check2 import GeoAugmentation


    if training:
        train_transforms_list=[
            LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstD(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            transforms.NormalizeIntensityd(keys=["image"]),# nonzero=True, channel_wise=True),
            # transforms.CropForegroundD(keys="image", source_key='image',select_fn=lambda x:x > 0.1),
            # transforms.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            # transforms.NormalizeIntensityd(keys=["image"],subtrahend=814,divisor=1960),#, nonzero=True, channel_wise=True),
            transforms.SpatialPadd(keys=["image"],spatial_size=[args.roi_large, args.roi_large, args.roi_z], method='symmetric', mode='constant'),
            RandSpatialCropSamplesd(
                                    keys=["image"],
                                    roi_size=[args.roi_large, args.roi_large, args.roi_z],
                                    num_samples=1,
                                    random_center=True,
                                    random_size=False,
                                ),
            Resized(keys=["image"], mode="trilinear", align_corners=True,
                    spatial_size=(args.roi_large, args.roi_large, args.roi_z)),
            GeoAugmentation(args,roi_large=args.roi_large,crop_size=args.roi_z)]
    else:
        train_transforms_list=[
            LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstD(keys=["image"]),
            Orientationd(keys=["image"], axcodes="SAR"),
            # transforms.NormalizeIntensityd(keys=["image"],subtrahend=814,divisor=1960),
            transforms.NormalizeIntensityd(keys=["image"]),# nonzero=True, channel_wise=True),
            # transforms.CropForegroundD(keys="image", source_key='image',select_fn=lambda x:x > 0.1),
            # transforms.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            # transforms.CropForegroundD(keys="image", source_key='image',select_fn=lambda x:x > 0.1),
            # transforms.NormalizeIntensityd(keys=["image"],subtrahend=814,divisor=1960),# nonzero=True, channel_wise=True),
            CenterSpatialCropd(keys=['image'],roi_size=(args.roi_large, args.roi_large, args.roi_z)),
            Resized(keys=["image"], mode="trilinear", align_corners=True,
                    spatial_size=(args.roi_large, args.roi_large, args.roi_z)),
            GeoAugmentation(args,roi_large=args.roi_large,crop_size=args.roi_z)]
    
    train_transforms = Compose(train_transforms_list)


    datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
    
    train_ds = monai.data.Dataset(data=datalist, transform=train_transforms)
    
    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=0,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True
    )
    
    return train_loader


def threshold_infer(x,td=0.3):
    # threshold at 0
    return x > td


def threshold(x,td=0):
    # threshold at 0
    return x > td