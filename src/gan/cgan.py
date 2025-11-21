import torch
from torch import nn
import torch.optim as optim

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

def train_cgan(gen, disc, trainloader, epochs, lr, device):

    from src.utils import weights_init
    
    gen.to(device)
    gen.apply(weights_init)
    disc.to(device)
    disc.apply(weights_init)

    optimG = optim.Adam(gen.parameters(), lr, betas=(0.5, 0.999))
    optimD = optim.Adam(disc.parameters(), lr, betas=(0.5, 0.999))

    validity_loss = nn.BCELoss()

    real_labels = 0.7 + 0.3 * torch.rand(10, device=device)
    fake_labels = 0.3 * torch.rand(10, device=device)

    for epoch in range(1, epochs + 1):
        for idx, (images, labels) in enumerate(trainloader, 0):
            batch_size = images.size(0)
            images = images.to(device)
            labels = labels.to(device)

            real_label = real_labels[idx % 10]
            fake_label = fake_labels[idx % 10]

            if idx % 25 == 0:
                real_label, fake_label = fake_label, real_label

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimD.zero_grad()

            # Real images
            validity_label = torch.full((batch_size,), real_label, device=device)
            real_validity = disc(images, labels)
            errD_real = validity_loss(real_validity, validity_label)
            errD_real.backward()
            D_x = real_validity.mean().item()

            # Fake images
            noise = torch.randn(batch_size, 100, device=device)
            sample_labels = torch.randint(0, 10, (batch_size,), device=device, dtype=torch.long)
            fakes = gen(noise, sample_labels)

            validity_label.fill_(fake_label)
            fake_validity = disc(fakes.detach(), sample_labels)
            errD_fake = validity_loss(fake_validity, validity_label)
            errD_fake.backward()
            D_G_z1 = fake_validity.mean().item()

            errD = errD_real + errD_fake
            optimD.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimG.zero_grad()
            noise = torch.randn(batch_size, 100, device=device)
            sample_labels = torch.randint(0, 10, (batch_size,), device=device, dtype=torch.long)
            fakes = gen(noise, sample_labels)

            validity_label.fill_(1.0)
            fake_validity = disc(fakes, sample_labels)
            errG = validity_loss(fake_validity, validity_label)
            errG.backward()
            D_G_z2 = fake_validity.mean().item()
            optimG.step()

            # -----------------
            #  Logging
            # -----------------
            if idx % 200 == 0:
                print(
                    f"[{epoch}/{epochs}] [{idx}/{len(trainloader)}] "
                    f"D_x: {D_x:.4f} D_G: {D_G_z1:.4f}/{D_G_z2:.4f} "
                    f"G_loss: {errG:.4f} D_loss: {errD:.4f}"
                )
