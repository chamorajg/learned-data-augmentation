import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
from .pytorch_ssim import SSIM


class VanillaVAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

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


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        self.decoder_input2 = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.ReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU())
            )



        self.decoder2 = nn.Sequential(*modules)

        self.final_layer2 = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.ReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())
        self.ssim_loss = SSIM()
        self.load_model()
        self.freeze_base_all()
    
    def load_model(self):
        model_state_dict = torch.load('/home/chandramouli/kaggle/paper/src/visualization/vae_cifar10.ckpt')["state_dict"]
        model_state_dict = {k.replace('model.',''):v for k,v in model_state_dict.items()}
        # self.encoder.load_state_dict(model_state_dict)
        encoder_state_dict = {k.replace('encoder.',''):v for k,v in model_state_dict.items() if 'encoder' in k}
        self.encoder.load_state_dict(encoder_state_dict)
        fc_mu_state_dict = {k.replace('fc_mu.',''):v for k,v in model_state_dict.items() if 'fc_mu' in k}
        self.fc_mu.load_state_dict(fc_mu_state_dict)
        fc_var_state_dict = {k.replace('fc_var.',''):v for k,v in model_state_dict.items() if 'fc_var' in k}
        self.fc_var.load_state_dict(fc_var_state_dict)
        decoder_input_state_dict = {k.replace('decoder_input.',''):v for k,v in model_state_dict.items() if 'decoder_input' in k}
        self.decoder_input.load_state_dict(decoder_input_state_dict)
        decoder_state_dict = {k.replace('decoder.',''):v for k,v in model_state_dict.items() if 'decoder' in k}
        decoder_state_dict = {k:v for k,v in decoder_state_dict.items() if not 'decoder_input' in k}
        self.decoder.load_state_dict(decoder_state_dict)
        final_layer_state_dict = {k.replace('final_layer.',''):v for k,v in model_state_dict.items() if 'final_layer' in k}
        self.final_layer.load_state_dict(final_layer_state_dict)
    
    def freeze_base_all(self):
        for child in list(self.encoder.children()):
            for _, param in child.named_parameters():
                param.requires_grad = False
        for child in list(self.fc_mu.children()):
            for _, param in child.named_parameters():
                param.requires_grad = False
        for child in list(self.fc_var.children()):
            for _, param in child.named_parameters():
                param.requires_grad = False
        for child in list(self.decoder_input.children()):
            for _, param in child.named_parameters():
                param.requires_grad = False
        for child in list(self.decoder.children()):
            for _, param in child.named_parameters():
                param.requires_grad = False
        for child in list(self.final_layer.children()):
            for _, param in child.named_parameters():
                param.requires_grad = False

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # self.encoder.eval()
        with torch.no_grad():
            result = self.encoder(input)
            result = torch.flatten(result, start_dim=1)
            # Split the result into mu and var components
            # of the latent Gaussian distribution
            mu = self.fc_mu(result)
            log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        with torch.no_grad():
            result = self.decoder_input(z)
            result = result.view(-1, 512, 2, 2)
            result = self.decoder(result)
            result = self.final_layer(result)
        return result
    
    def decode2(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input2(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder2(result)
        result = self.final_layer2(result)
        return result

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
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [[self.decode2(z), input, mu, log_var], [self.decode(z), input, mu, log_var]]

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
        recons = args[0][0]
        input = args[0][1]
        mu = args[0][2]
        log_var = args[0][3]
        recons_base = args[1][0]
        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = F.binary_cross_entropy(recons, recons_base, reduction='sum')
        # print(augment_loss)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode2(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        output = self.forward(x)
        return output[0][0], output[1][0]