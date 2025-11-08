import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, num_channels=1):
        super().__init__()
        self.num_channels = num_channels
        self.embedding = nn.Embedding(10, 100)  

        if num_channels == 1:  # Fashion-MNIST (28x28)
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(200, 256, 7, 1, 0, bias=False),  # noise + embed concat → 200
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 14x14
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, num_channels, 4, 2, 1, bias=False),  # 28x28
                nn.Tanh()
            )
        else:  # RGB dataset (32x32)
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(200, 512, 4, 1, 0, bias=False),  # 4x4
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # 8x8
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 16x16
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, num_channels, 4, 2, 1, bias=False),  # 32x32
                nn.Tanh()
            )

    def forward(self, noise, labels):
        label_emb = self.embedding(labels)
        x = torch.cat((noise, label_emb), dim=1)  # N × 200
        x = x.view(-1, 200, 1, 1)
        return self.layers(x)

class Discriminator(nn.Module):
    def __init__(self, num_channels=1):
        super().__init__()
        self.num_channels = num_channels
        self.embedding = nn.Embedding(10, num_channels * 28 * 28 if num_channels == 1 else num_channels * 32 * 32)

        if num_channels == 1:  # 28x28 grayscale
            self.layers = nn.Sequential(
                nn.Conv2d(num_channels * 2, 64, 4, 2, 1, bias=False),  # 28→14
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 14→7
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 1, 7, 1, 0, bias=False),
                nn.Sigmoid()
            )
        else:  # 32x32 RGB
            self.layers = nn.Sequential(
                nn.Conv2d(num_channels * 2, 64, 4, 2, 1, bias=False),  # 32→16
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 16→8
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 8→4
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

    def forward(self, img, labels):
        batch_size = img.size(0)
        label_emb = self.embedding(labels)
        label_emb = label_emb.view(batch_size, self.num_channels, img.size(2), img.size(3))
        x = torch.cat((img, label_emb), dim=1)  
        return self.layers(x).view(-1)
    

def get_cgan(num_channels):
    gen = Generator(num_channels)
    disc = Discriminator(num_channels)
    return gen, disc