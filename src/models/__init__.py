from .experiment import VAEXperiment
from .classification import VAEClassification
from .dual_decoder_experiment import DualDecoderVAEXperiment

vae_experiments = {
    'VAEXperiment': VAEXperiment,
    'VAEClassification': VAEClassification,
    'DualDecoderVAEXperiment': DualDecoderVAEXperiment,}