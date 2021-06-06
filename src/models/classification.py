import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from models import BaseVAE
from models.types_ import *
from collections import OrderedDict

sys.path.append('../data/')
import pytorch_lightning as pl
from torchvision import transforms
from classification_dataset import CIFAR10
from torch.utils.data import DataLoader
from tiny_imagenet_dataset import Dataset

class VAEClassification(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEClassification, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.test_transform = transforms.RandomApply([transforms.RandomCrop(size=24)], p=0.5)
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        imgs, labels = batch
        real_img, transform_img = imgs[0], imgs[1]
        self.curr_device = real_img.device
        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*[results, labels],)

        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        # return train_loss
        tqdm_dict = {'train_loss': train_loss['loss'], 'acc1': train_loss['acc1'], 'acc5': train_loss['acc5']}
        output = OrderedDict({
            'loss': train_loss['loss'],
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        return output

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        imgs, labels = batch
        real_img, transform_img = imgs[0], imgs[1]
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*[results, labels],)

        output = OrderedDict({
            'val_loss': val_loss['loss'],
            'val_acc1': val_loss['acc1'],
            'val_acc5': val_loss['acc5'],
        })

        return output

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc1 = torch.stack([x['val_acc1'] for x in outputs]).mean()
        avg_acc5 = torch.stack([x['val_acc5'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc1': avg_acc1, 'val_acc5': avg_acc5}
        return {'progress_bar': tensorboard_logs, 'val_loss': avg_loss, 'val_acc1':avg_acc1, 'log': tensorboard_logs}

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    # @data_loader
    def train_dataloader(self):
        transform = self.data_transforms()
        target_transform = self.target_transforms()
        if self.params['dataset'] == 'celeba':
            dataset = CelebA(root = self.params['data_path'],
                                split='train',
                                transform=transform,
                                download=True)
        elif self.params['dataset'] == 'cifar10':
            dataset = CIFAR10(root = self.params['data_path'],
                                train=True,
                                transform = transform,
                                download = False,
                                img_size=self.params['img_size'])
        elif self.params['dataset'] == 'imagenet':
            dataset = Dataset(mode='train', 
                                transform=transform,
                                img_size=self.params['img_size'])
        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size= self.params['batch_size'],
                          shuffle = True,
                          drop_last=True,
                          num_workers=6)

    # @data_loader
    def val_dataloader(self):
        transform = self.val_data_transforms()

        if self.params['dataset'] == 'celeba':
            self.sample_dataloader =  DataLoader(CelebA(root = self.params['data_path'],
                                                        split='test',
                                                        transform=transform,
                                                        download=True,),
                                                 batch_size= self.params['batch_size'],
                                                 shuffle = True,
                                                 drop_last=True,
                                                 num_workers=4)
        elif self.params['dataset'] == 'cifar10':
            self.sample_dataloader =  DataLoader(CIFAR10(root=self.params['data_path'],
                                                        train=False,
                                                        transform=transform,
                                                        download=False,
                                                        img_size=self.params['img_size']),
                                                 batch_size= self.params['batch_size'],
                                                 shuffle=False,
                                                 num_workers=4,
                                                 drop_last=True
                                                )
        elif self.params['dataset'] == 'imagenet':
            self.sample_dataloader = DataLoader(Dataset(mode='valid', 
                                                 transform=transform,),
                                                 batch_size= self.params['batch_size'],
                                                 shuffle=False,
                                                 num_workers=4,
                                                 drop_last=True)
        else:
            raise ValueError('Undefined dataset type')
        self.num_val_imgs = len(self.sample_dataloader)
        return self.sample_dataloader

    def data_transforms(self):
        transform = transforms.Compose([
                                        # self.test_transform,
                                        transforms.Resize(self.params['img_size']),
                                        transforms.ToTensor(),])
        return transform
    
    def target_transforms(self):
        transform = transforms.Compose([
                                        transforms.Resize(self.params['img_size']),
                                        transforms.ToTensor(),])
        return transform
    
    def val_data_transforms(self):
        transform = transforms.Compose([
                                        # self.test_transform,
                                        transforms.Resize(self.params['img_size']),
                                        transforms.ToTensor(),])
        return transform