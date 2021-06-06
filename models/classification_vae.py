import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class ClassificationVAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(ClassificationVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_layer = nn.Sequential(
                                    nn.Linear(latent_dim, 32),
                                    nn.BatchNorm1d(num_features=32),
                                    nn.Tanh(),
                                    nn.Linear(32, 10),
                                    nn.BatchNorm1d(num_features=10),
                                    nn.ReLU(),
                                    )
        self.loss_criterion = nn.CrossEntropyLoss()
        self.load_model()
    
    def load_model(self):
        model_load_state_dict = torch.load('/home/chandramouli/kaggle/paper/src/visualization/cifar_transforms/random_crop/checkpoints/epoch=24.ckpt')['state_dict']
        self.encoder.load_state_dict({k.replace('model.encoder.',''):v for k,v in model_load_state_dict.items() if 'encoder' in k})
        self.fc_mu.load_state_dict({k.replace('model.fc_mu.',''):v for k,v in model_load_state_dict.items() if 'fc_mu' in k})
        self.fc_mu.load_state_dict({k.replace('model.fc_var.',''):v for k,v in model_load_state_dict.items() if 'fc_var' in k})

        
    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # with torch.no_grad():
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        # if self.training:
        #     gaussian_noise = torch.randn_like(std)
        #     return eps * std + mu + gaussian_noise * std
        # else:
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        with torch.no_grad():
            mu, log_var = self.encode(input)
            z = self.reparameterize(mu, log_var)
        output = self.fc_layer(z)
        return  output

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        pred = args[0]
        target = args[1]
        class_loss = self.loss_criterion(pred, target) 
        acc1, acc5 = self.__accuracy(pred, target, topk=(1, 5))
        return {'loss': class_loss, 'acc1': acc1, 'acc5': acc5}
    
    @classmethod
    def __accuracy(cls, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
