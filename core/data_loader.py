"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
from itertools import chain
import os
import random

from munch import Munch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageMaskFolder(ImageFolder):
    def __init__(self, root, transform=None, mask_transform=None, use_mask=True):
        self.use_mask = use_mask
        super().__init__(root, transform)
        self.mask_transform = mask_transform

    def find_classes(self, directory):
        classes = [d.name for d in os.scandir(directory)
                   if d.is_dir() and not d.name.endswith('_mask') and not d.name.startswith('.')]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path).convert('RGB')
        mask = None
        if self.use_mask:
            class_name = self.classes[target]
            fname = os.path.basename(path)
            mask_path = os.path.join(self.root, class_name + '_mask', fname)
            mask = Image.open(mask_path).convert('L')

        if self.transform is not None:
            state = torch.get_rng_state()
            sample = self.transform(sample)
            if self.use_mask:
                torch.set_rng_state(state)
                if self.mask_transform is not None:
                    mask = self.mask_transform(mask)
                else:
                    mask = self.transform(mask)
            else:
                mask = torch.ones(1, sample.size(1), sample.size(2))
        return sample, mask, target


class ReferenceMaskDataset(data.Dataset):
    def __init__(self, root, transform=None, mask_transform=None, use_mask=True):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform
        self.mask_transform = mask_transform
        self.use_mask = use_mask
        self.root = root

    def _make_dataset(self, root):
        domains = [d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d))
                   and not d.endswith('_mask') and not d.startswith('.')]
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]

        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')

        if self.use_mask:
            domain = os.path.basename(os.path.dirname(fname))
            root_dir = os.path.dirname(os.path.dirname(fname))
            mask_path = os.path.join(root_dir, domain + '_mask', os.path.basename(fname))
            mask2_path = os.path.join(root_dir, domain + '_mask', os.path.basename(fname2))
            mask = Image.open(mask_path).convert('L')
            mask2 = Image.open(mask2_path).convert('L')

        if self.transform is not None:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            if self.use_mask:
                mask = self.mask_transform(mask) if self.mask_transform is not None else self.transform(mask)
            else:
                mask = torch.ones(1, img.size(1), img.size(2))

            state = torch.get_rng_state()
            img2 = self.transform(img2)
            torch.set_rng_state(state)
            if self.use_mask:
                mask2 = self.mask_transform(mask2) if self.mask_transform is not None else self.transform(mask2)
            else:
                mask2 = torch.ones(1, img2.size(1), img2.size(2))
        else:
            if self.use_mask:
                mask = self.mask_transform(mask) if self.mask_transform is not None else mask
                mask2 = self.mask_transform(mask2) if self.mask_transform is not None else mask2
            else:
                mask = torch.ones(1, img.size(1), img.size(2))
                mask2 = torch.ones(1, img2.size(1), img2.size(2))

        return img, mask, img2, mask2, label

    def __len__(self):
        return len(self.targets)


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames

# Center-crop to match the target aspect ratio, then resize
class CenterCropResize:
    def __init__(self, target_size):
        """target_size: (height, width)"""
        self.target_h, self.target_w = target_size
        self.target_ratio = self.target_w / self.target_h

    def __call__(self, img):
        w, h = img.size
        in_ratio = w / h
        if in_ratio > self.target_ratio:
            new_w = int(self.target_ratio * h)
            left = (w - new_w) // 2
            img = img.crop((left, 0, left + new_w, h))
        elif in_ratio < self.target_ratio:
            new_h = int(w / self.target_ratio)
            top = (h - new_h) // 2
            img = img.crop((0, top, w, top + new_h))
        return img.resize((self.target_w, self.target_h), Image.BILINEAR)


class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None, mask_transform=None, use_mask=True):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.mask_transform = mask_transform
        self.use_mask = use_mask
        self.root = root
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.use_mask:
            domain = os.path.basename(self.root)
            root_dir = os.path.dirname(self.root)
            mask_path = os.path.join(root_dir, domain + '_mask', os.path.basename(fname))
            mask = Image.open(mask_path).convert('L')
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
            else:
                mask = self.transform(mask)
        else:
            mask = torch.ones(1, img.size(1), img.size(2))
        return img, mask

    def __len__(self):
        return len(self.samples)


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_train_loader(root, which='source', img_size=256,
                     batch_size=8, prob=0.5, num_workers=4, use_mask=False):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    if isinstance(img_size, tuple):
        height, width = img_size
    else:
        height = width = img_size
    target_ratio = width / height

    crop = transforms.RandomResizedCrop(
        (height, width), scale=[0.8, 1.0], ratio=[0.9 * target_ratio, 1.1 * target_ratio])
    rand_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < prob else x)

    transform = transforms.Compose([
        rand_crop,
        CenterCropResize((height, width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    mask_transform = transforms.Compose([
        rand_crop,
        CenterCropResize((height, width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    if which == 'source':
        dataset = ImageMaskFolder(root, transform, mask_transform, use_mask=use_mask)
    elif which == 'reference':
        dataset = ReferenceMaskDataset(root, transform, mask_transform, use_mask=use_mask)
    else:
        raise NotImplementedError

    sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_eval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False, use_mask=False):
    print('Preparing DataLoader for the evaluation phase...')
    if isinstance(img_size, tuple):
        height, width = img_size
    else:
        height = width = img_size

    if imagenet_normalize:
        resize_h, resize_w = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        resize_h, resize_w = height, width
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        CenterCropResize((height, width)),
        transforms.Resize([resize_h, resize_w]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    mask_transform = transforms.Compose([
        CenterCropResize((height, width)),
        transforms.Resize([resize_h, resize_w]),
        transforms.ToTensor(),
    ])

    dataset = DefaultDataset(root, transform=transform, mask_transform=mask_transform, use_mask=use_mask)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_test_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4, use_mask=False):
    print('Preparing DataLoader for the generation phase...')
    if isinstance(img_size, tuple):
        height, width = img_size
    else:
        height = width = img_size

    transform = transforms.Compose([
        CenterCropResize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    mask_transform = transforms.Compose([
        CenterCropResize((height, width)),
        transforms.ToTensor(),
    ])

    dataset = ImageMaskFolder(root, transform, mask_transform, use_mask=use_mask)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, m, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, m, y = next(self.iter)
        return x, m, y

    def _fetch_refs(self):
        try:
            x, m, x2, m2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, m, x2, m2, y = next(self.iter_ref)
        return x, m, x2, m2, y

    def __next__(self):
        x, m, y = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, m_ref, x_ref2, m_ref2, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, m_src=m, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, m_ref=m_ref,
                           x_ref2=x_ref2, m_ref2=m_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, m_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, m_src=m, y_src=y,
                           x_ref=x_ref, m_ref=m_ref, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, m=m, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})
