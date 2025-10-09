from .cgan import get_c_gan
import os 
import torch

def get_gan(gan_model_name: str, dataset_name: str, opt, use_pretrained: bool = True, pretrained_gan_path: str = None):
    """Return (generator, discriminator) pair for the requested GAN type and dataset."""
    
    gan_model_name = gan_model_name.lower()

    if gan_model_name == 'cgan':
        generator, discriminator =  get_c_gan(opt)
    else:
        raise ValueError(f"GAN model {gan_model_name} is not supported.")
    
    if use_pretrained: 
        if pretrained_gan_path is not None:
            if os.path.exists(pretrained_gan_path):
                try:
                    generator_path = os.path.join(pretrained_gan_path, 'generator.pt')
                    discriminator_path = os.path.join(pretrained_gan_path, 'discriminator.pt')

                    generator_state = torch.load(generator_path, map_location='cpu')
                    discriminator_state = torch.load(discriminator_path, map_location='cpu')

                    if isinstance(generator_state, dict) and 'state_dict' in generator_state:
                        generator_state = generator_state['state_dict']

                    if isinstance(discriminator_state, dict) and 'state_dict' in discriminator_state:
                        discriminator_state = discriminator_state['state_dict']

                    generator.load_state_dict(generator_state, strict=False)
                    discriminator.load_state_dict(discriminator_state, strict=False)
                    print(f"Loaded pretrained generator and discriminator from {pretrained_gan_path}")
                except Exception as e:
                    print(f"Failed to load pretrained generator and discriminator from {pretrained_gan_path}: {e}")
            else:
                print(f"Pretrained GAN path {pretrained_gan_path} does not exist. Return random initialized gan.")

    return generator, discriminator