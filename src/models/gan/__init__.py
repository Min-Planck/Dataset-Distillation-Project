from .cgan import get_c_gan
from .acgan import get_ac_gan
import os 
import torch

def get_gan(gan_model_name: str, dataset_name: str, opt, use_pretrained: bool = True, pretrained_gan_path: str = None):
    """Return (generator, discriminator) pair for the requested GAN type and dataset."""
    
    gan_model_name = gan_model_name.lower()

    if gan_model_name == 'cgan':
        generator, discriminator =  get_c_gan(opt)
    elif gan_model_name == 'acgan':
        generator, discriminator = get_ac_gan(opt)
    else:
        raise ValueError(f"GAN model {gan_model_name} is not supported.")
    
    if use_pretrained: 
        if pretrained_gan_path is not None:
            if os.path.exists(pretrained_gan_path):
                try:
                    state = torch.load(pretrained_gan_path, map_location='cpu')
                    if isinstance(state, dict) and 'state_dict' in state:
                        state = state['state_dict']
                    generator.load_state_dict(state, strict=False)
                    print(f"Loaded pretrained generator from {pretrained_gan_path}")
                except Exception as e:
                    print(f"Failed to load pretrained generator from {pretrained_gan_path}: {e}")
            else:
                print(f"Pretrained GAN path {pretrained_gan_path} does not exist. Return random initialized gan.")

    return generator, discriminator