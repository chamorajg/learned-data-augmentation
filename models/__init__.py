from .base import *
from .vanilla_vae import *
from .cvae import *
from .mssim_vae import MSSIMVAE
from .logcosh_vae import *
from .classification_vae import *
from .dual_decoder_vae import *


# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE
CVAE = ConditionalVAE

vae_models = {
              'MSSIMVAE':MSSIMVAE,
              'LogCoshVAE':LogCoshVAE,
              'VanillaVAE':VanillaVAE,
              'ConditionalVAE':ConditionalVAE,
              'ClassificationVAE':ClassificationVAE,
              'DualDecoderVAE':DualDecoderVAE,}
