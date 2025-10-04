import torch
from torch import nn 
import numpy as np


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt
        self.label_emb = nn.Embedding(opt['n_classes'], opt['latent_dim'])

        self.init_size = opt['img_size'] // 4  
        self.l1 = nn.Sequential(
            nn.Linear(opt['latent_dim'], 128 * self.init_size ** 2),  
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128), 
            
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, self.opt['channels'], kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.opt = opt

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(self.opt['channels'], 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # compute flattened feature size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, self.opt['channels'], self.opt['img_size'], self.opt['img_size'])
            out = self.conv_blocks(dummy)
            _, c, h, w = out.shape
            flattened_dim = c * h * w


        self.adv_layer = nn.Sequential(nn.Linear(flattened_dim, 1), nn.Sigmoid())
        self.aux_layer = nn.Linear(flattened_dim, self.opt['n_classes'])

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label_logits = self.aux_layer(out)
        return validity, label_logits
    
def get_ac_gan(opt): 
    gen_ = Generator(opt) 
    disc_ = Discriminator(opt)
    gen_.apply(weights_init_normal)
    disc_.apply(weights_init_normal)

    return gen_, disc_
