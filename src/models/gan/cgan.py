import torch
from torch import nn 
import numpy as np


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt 
        self.img_shape = (opt['channels'], opt['img_size'], opt['img_size'])
        self.label_emb = nn.Embedding(opt['n_classes'], opt['n_classes'])
        self.base_channel = opt['base_channel']
        
        self.fc1 = nn.Linear(opt['latent_dim'] + opt['n_classes'], self.base_channel * 4 * 4)
        self.bn0 = nn.BatchNorm1d(self.base_channel * 4 * 4)
        self.relu0 = nn.LeakyReLU(0.2, inplace=True)  # Thay ReLU bằng LeakyReLU

        if opt['channels'] == 1: 
            self.trans_conv1 = nn.ConvTranspose2d(self.base_channel, self.base_channel, kernel_size=3, stride=2, padding=1)
        else: 
            self.trans_conv1 = nn.ConvTranspose2d(self.base_channel, self.base_channel, kernel_size=4, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(self.base_channel)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        
        def block(in_feat, out_feat, normalize=True, transpose=False):
            if transpose: 
                layers = [nn.ConvTranspose2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1)]
            else:
                layers = [nn.Conv2d(in_feat, out_feat, 3, stride=1, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.base_channel, self.base_channel),
            *block(self.base_channel, self.base_channel),
            *block(self.base_channel, self.base_channel),
            *block(self.base_channel, self.base_channel, transpose=True),
            *block(self.base_channel, self.base_channel),
            *block(self.base_channel, self.base_channel, transpose=True),
        )
        
        self.conv8 = nn.Conv2d(self.base_channel, opt['channels'], kernel_size=3, stride=1, padding=1)  # Sửa lỗi chính tả
        self.tanh = nn.Tanh()

    def forward(self, noise, labels, print_size=False):
        assert noise.size(-1) == self.opt['latent_dim'], f"Expected noise dim {self.opt['latent_dim']}, got {noise.size(-1)}"
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        if print_size: 
            print("gen_input before fc1:", gen_input.size())
        gen_input = self.fc1(gen_input)
        gen_input = self.bn0(gen_input)
        gen_input = self.relu0(gen_input)
        if print_size: 
            print("gen_input after fc1:", gen_input.size())

        gen_input = gen_input.view(-1, self.base_channel, 4, 4) 
        if print_size: 
            print("gen_input after view:", gen_input.size())
        img = self.trans_conv1(gen_input)
        img = self.bn1(img)
        img = self.relu1(img)
        if print_size:
            print("img after trans_conv1:", img.size())
        img = self.model(img)
        if print_size:
            print("img after model:", img.size())
        img = self.conv8(img)
        img = self.tanh(img)
        if print_size:
            print("img after conv8 and tanh:", img.size())
        return img


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()   
        self.opt = opt
        self.base_channel = opt['base_channel']
        self.img_shape = (self.opt['channels'], self.opt['img_size'], self.opt['img_size'])
        self.label_embedding = nn.Embedding(opt['n_classes'], self.opt['img_size'] * self.opt['img_size'])
        self.conv1 = nn.Conv2d(self.opt['channels'] + 1, self.base_channel, kernel_size=3, stride=1, padding=1)
        self.ln1 = nn.LayerNorm([self.base_channel, self.opt['img_size'], self.opt['img_size']])  
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.model = nn.Sequential(
            nn.Conv2d(self.base_channel, self.base_channel, kernel_size=3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.BatchNorm2d(self.base_channel),
            nn.LeakyReLU(0.2, inplace=True),
        
            nn.Conv2d(self.base_channel, self.base_channel, kernel_size=3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.BatchNorm2d(self.base_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
        
            nn.Conv2d(self.base_channel, self.base_channel, kernel_size=3, stride=2, padding=1),  # 7x7 -> 4x4
            nn.BatchNorm2d(self.base_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
        
            nn.Conv2d(self.base_channel, self.base_channel, kernel_size=3, stride=1, padding=1),  # 4x4 -> 4x4
            nn.BatchNorm2d(self.base_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
        
            nn.Conv2d(self.base_channel, self.base_channel, kernel_size=3, stride=1, padding=1),  # 4x4 -> 4x4
            nn.BatchNorm2d(self.base_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
        
            nn.Conv2d(self.base_channel, self.base_channel, kernel_size=3, stride=1, padding=1),  # 4x4 -> 4x4
            nn.BatchNorm2d(self.base_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
        
            nn.Conv2d(self.base_channel, self.base_channel, kernel_size=3, stride=2, padding=1),  # 4x4 -> 2x2
            nn.BatchNorm2d(self.base_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
        
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 2x2 -> 1x1
            nn.Flatten(),
            nn.Linear(self.base_channel * 1 * 1, 1),  # Đầu ra nhị phân
            nn.Sigmoid(),
        )


    def forward(self, img, labels, print_shape=False):
        labels_emb = self.label_embedding(labels).view(-1, 1, self.opt['img_size'], self.opt['img_size'])
        d_in = torch.cat((img, labels_emb), dim=1)
        x = self.conv1(d_in)
        if print_shape:
            print("After conv1:", x.shape)
        x = self.ln1(x)
        x = self.relu1(x)
        for i, layer in enumerate(self.model):
            x = layer(x)
            if print_shape: 
                print(f"After layer {i}: {x.shape}")
        return x
    
def get_c_gan(opt): 
    return Generator(opt), Discriminator(opt)
