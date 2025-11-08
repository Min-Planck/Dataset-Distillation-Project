from torch import nn

class Generator(nn.Module):
    def __init__(self, num_channels=1):
        super().__init__()
        self.num_channels = num_channels
        self.embedding = nn.Embedding(10, 100)
        
        if num_channels == 1:  # Fashion-MNIST: 28x28
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(100, 256, 7, 1, 0, bias=False),  # 7x7
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 14x14
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, num_channels, 4, 2, 1, bias=False),  # 28x28
                nn.Tanh()
            )
        else:  # RGB dataset like CIFAR-10: 32x32
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),  # 4x4
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
        x = noise * label_emb
        x = x.view(-1, 100, 1, 1)
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self, num_channels=1):
        super().__init__()
        self.num_channels = num_channels
        
        if num_channels == 1:  # Fashion-MNIST 28x28
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(num_channels, 64, 4, 2, 1, bias=False),  # 28→14
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 14→7
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True)
            )
            kernel_size = 7
            in_channels = 128
        else:  # RGB 32x32
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(num_channels, 64, 4, 2, 1, bias=False),  # 32→16
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 16→8
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 8→4
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
            )
            kernel_size = 4
            in_channels = 256

        self.validity_layer = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.label_layer = nn.Sequential(
            nn.Conv2d(in_channels, 11, kernel_size, 1, 0, bias=False),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        validity = self.validity_layer(x).view(-1)
        label_pred = self.label_layer(x).view(-1, 11)
        return validity, label_pred
    
def get_acgan(num_channels): 
    gen = Generator(num_channels)
    disc = Discriminator(num_channels)

    return gen, disc
