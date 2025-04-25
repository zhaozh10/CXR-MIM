# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
from PIL import Image
import os
import json
from sklearn.base import TransformerMixin
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class Reflacx(Dataset):
    def __init__(self, data_root: str, preload=False, transforms_dict=None) -> None:
        self.preload=preload
        self.data_root=data_root
        self.info=json.load(open(os.path.join(self.data_root,"reflacx.json")))
        self.gaze_dir=os.path.join(self.data_root,"attention")
        self.foreground_dir=os.path.join(self.data_root,'foreground')
        self.transforms=transforms

        self.flip = transforms_dict['flip']
        self.norm= transforms_dict['norm']
        self.resize_crop = transforms_dict['resized_crop']
        # self.local_crop = transforms.RandomResizedCrop(96, scale=self.local_crops_scale, interpolation=Image.BICUBIC,antialias=None)
        self.to_tensor = transforms_dict['to_tensor']  # Convert PIL to Tensor
        self.to_pil=transforms.ToPILImage()


    def __getitem__(self, index):
        image_path=self.info[index]['image_path']
        study_id=self.info[index]['study_id']
        reflacx_id=self.info[index]['reflacx_id']
        image_path=os.path.join(self.data_root,image_path)
        gaze_path=os.path.join(self.gaze_dir,study_id,f"{reflacx_id}.png")
        foreground_path=os.path.join(self.foreground_dir,study_id,f"{reflacx_id}.png")

        if self.preload==False:
            image = Image.open(image_path).convert('RGB')
            gaze=Image.open(gaze_path)

            # Convert image and gaze (both PIL) to tensors
            image_tensor = self.to_tensor(image)  # Convert 3-channel image to Tensor
            gaze_tensor = self.to_tensor(gaze)    # Convert 1-channel gaze to Tenso
            # Concatenate image and gaze into a 4-channel tensor
            combined = torch.cat((image_tensor, gaze_tensor), dim=0)


            combined_cropped = self.resize_crop(combined)
            combined_cropped = self.flip(combined_cropped)


            image_post = combined_cropped[:3, :, :]
            gaze_post = combined_cropped[3:, :, :]
            image_post = self.norm(image_post)

            return {"img":image_post,"attention":gaze_post}
        else:
            image = Image.open(image_path).convert('RGB')
            gaze=Image.open(gaze_path)
            foreground=Image.open(foreground_path)

            # Convert image and gaze (both PIL) to tensors
            image_tensor = self.to_tensor(image)  # Convert 3-channel image to Tensor
            foreground_tensor = self.to_tensor(foreground)    # Convert 1-channel foreground to Tensor
            gaze_tensor = self.to_tensor(gaze)    # Convert 1-channel gaze to Tensor
            # Concatenate image and gaze into a 4-channel tensor
            combined = torch.cat((image_tensor, foreground_tensor, gaze_tensor), dim=0)


            combined_cropped = self.resize_crop(combined)
            combined_cropped = self.flip(combined_cropped)


            image_post = combined_cropped[:3, :, :]
            # foreground_post = combined_cropped[3:4, :, :]
            # gaze_post = combined_cropped[4:, :, :]
            foreground_post = combined_cropped[3, :, :]
            gaze_post = combined_cropped[4, :, :]
            image_post = self.norm(image_post)

            # return {"img":image_post,"gaze":gaze_post,"foreground":foreground_post.squeeze()}
        return {"img":image_post,"attention":gaze_post,"foreground":foreground_post}
    

    def __len__(self):
        return len(self.info)

class MIMIC(Dataset):
    def __init__(self, data_root: str, dataset_size:int, preload=True, transforms_dict=None) -> None:
        
        self.data_root=data_root
        self.preload=preload
        self.dataset_size=dataset_size
        self.info=json.load(open(os.path.join(self.data_root,"mimic-cxr-jpg-cleanv4.json")))[:self.dataset_size]
        self.gaze_dir=os.path.join(self.data_root,"attention")
        self.foreground_dir=os.path.join(self.data_root,"foreground")
        self.transforms=transforms

        self.flip = transforms_dict['flip']
        self.norm= transforms_dict['norm']
        self.resize_crop = transforms_dict['resized_crop']
        # self.local_crop = transforms.RandomResizedCrop(96, scale=self.local_crops_scale, interpolation=Image.BICUBIC,antialias=None)
        self.to_tensor = transforms_dict['to_tensor']  # Convert PIL to Tensor
        self.to_pil=transforms.ToPILImage()


    def __getitem__(self, index):
        raw_image_path=self.info[index]['image_path']
        study_id=self.info[index]['study_id']
        reflacx_id=self.info[index]['subject_id']
        image_path=os.path.join(self.data_root,"files",raw_image_path)
        gaze_path=os.path.join(self.data_root,"attention",raw_image_path.replace('.jpg','.png'))
        foreground_path=os.path.join(self.data_root,"foreground",raw_image_path.replace('.jpg','.png'))
        # gaze_path=os.path.join(self.gaze_dir,study_id,f"{reflacx_id}.png")

        if self.preload==False:
            image = Image.open(image_path).convert('RGB')
            gaze=Image.open(gaze_path)

            # Convert image and gaze (both PIL) to tensors
            image_tensor = self.to_tensor(image)  # Convert 3-channel image to Tensor
            gaze_tensor = self.to_tensor(gaze)    # Convert 1-channel gaze to Tenso
            # Concatenate image and gaze into a 4-channel tensor
            combined = torch.cat((image_tensor, gaze_tensor), dim=0)


            combined_cropped = self.resize_crop(combined)
            combined_cropped = self.flip(combined_cropped)


            image_post = combined_cropped[:3, :, :]
            gaze_post = combined_cropped[3:, :, :]
            image_post = self.norm(image_post)

            return {"img":image_post,"attention":gaze_post}
        
        else:
            image = Image.open(image_path).convert('RGB')
            gaze=Image.open(gaze_path)
            foreground=Image.open(foreground_path)

            # Convert image and gaze (both PIL) to tensors
            image_tensor = self.to_tensor(image)  # Convert 3-channel image to Tensor
            foreground_tensor = self.to_tensor(foreground)    # Convert 1-channel foreground to Tensor
            gaze_tensor = self.to_tensor(gaze)    # Convert 1-channel gaze to Tensor
            # Concatenate image and gaze into a 4-channel tensor
            combined = torch.cat((image_tensor, foreground_tensor, gaze_tensor), dim=0)


            combined_cropped = self.resize_crop(combined)
            combined_cropped = self.flip(combined_cropped)


            image_post = combined_cropped[:3, :, :]
            foreground_post = combined_cropped[3, :, :]
            gaze_post = combined_cropped[4, :, :]
            image_post = self.norm(image_post)


        return {"img":image_post,"attention":gaze_post,"foreground":foreground_post}
        # return {"image":image}
        # return {"image":image,"gaze":gaze}
    

    def __len__(self):
        return len(self.info)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    print(dataset)
    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=transforms.InterpolationMode.BICUBIC,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std)
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224/256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC))  # to maintain same ratio w.r.t. 224 images
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)