import sys
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from scipy.stats import norm


sys.path.append('../data/')
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from cifar_dataset import CIFAR10
from torch.utils.data import DataLoader
from tiny_imagenet_dataset import Dataset

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.test_transform = transforms.RandomAffine(degrees=60)
        #, translate=(0, 0.25), scale=(0.75, 1), shear=0.2)
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
        results = self.forward(transform_img, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['batch_size']/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx,
                                              real_img = real_img)

        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        imgs, labels = batch
        real_img, transform_img = imgs[0], imgs[1]
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = self.params['batch_size']/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx,
                                            real_img = real_img)

        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def sample_images(self):
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=8)
        vutils.save_image(test_input,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_input_{self.current_epoch}.png",
                          normalize=True,
                          nrow=8)
        try:
            samples = self.model.sample(self.params['batch_size'],
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                              f"{self.logger.name}.png",
                              normalize=True,
                              nrow=8)
        except:
            pass


        del test_input, recons #, samples


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
                                                 num_workers=6,
                                                 drop_last=True
                                                )
        elif self.params['dataset'] == 'imagenet':
            self.sample_dataloader = DataLoader(Dataset(mode='valid', 
                                                 transform=transform,
                                                 img_size=self.params['img_size']),
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
                                        self.test_transform,
                                        transforms.Resize(self.params['img_size']),
                                        transforms.ToTensor(),])
        return transform
    
    def val_data_transforms(self):
        transform = transforms.Compose([
                                        # self.test_transform,/
                                        transforms.Resize(self.params['img_size']),
                                        transforms.ToTensor(),])
        return transform