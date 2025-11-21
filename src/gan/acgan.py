from torch import nn
import torch 
import torch.optim as optim
import torch.nn.functional as F

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
def train_acgan(gen, disc, trainloader, epochs, lr, device): 
    from utils import weights_init 

    gen.to(device)
    gen.apply(weights_init)
    disc.to(device)
    disc.apply(weights_init)

    optimG = optim.Adam(gen.parameters(), lr, betas = (0.5,0.999))
    optimD = optim.Adam(disc.parameters(), lr, betas = (0.5,0.999))

    validity_loss = nn.BCELoss()

    real_labels = 0.7 + 0.3 * torch.rand(10, device = device)
    fake_labels = 0.3 * torch.rand(10, device = device)

    for epoch in range(1,epochs+1):
        
        for idx, (images,labels) in enumerate(trainloader,0):
            
            batch_size = images.size(0)
            labels= labels.to(device)
            images = images.to(device)
            
            real_label = real_labels[idx % 10]
            fake_label = fake_labels[idx % 10]
            
            fake_class_labels = 10*torch.ones((batch_size,),dtype = torch.long,device = device)
            
            if idx % 25 == 0:
                real_label, fake_label = fake_label, real_label
            
            # ---------------------
            #         disc
            # ---------------------
            
            optimD.zero_grad()       
            
            # real
            validity_label = torch.full((batch_size,),real_label , device = device)
    
            pvalidity, plabels = disc(images)       
            
            errD_real_val = validity_loss(pvalidity, validity_label)            
            errD_real_label = F.nll_loss(plabels,labels)
            
            errD_real = errD_real_val + errD_real_label
            errD_real.backward()
            
            D_x = pvalidity.mean().item()        
            
            #fake 
            noise = torch.randn(batch_size,100,device = device)  
            sample_labels = torch.randint(0,10,(batch_size,),device = device, dtype = torch.long)
            
            fakes = gen(noise,sample_labels)
            
            validity_label.fill_(fake_label)
            
            pvalidity, plabels = disc(fakes.detach())       
            
            errD_fake_val = validity_loss(pvalidity, validity_label)
            errD_fake_label = F.nll_loss(plabels, fake_class_labels)
            
            errD_fake = errD_fake_val + errD_fake_label
            errD_fake.backward()
            
            D_G_z1 = pvalidity.mean().item()
            
            #finally update the params!
            errD = errD_real + errD_fake
            
            optimD.step()
        
            
            # ------------------------
            #      gen
            # ------------------------
            
            
            optimG.zero_grad()
            
            noise = torch.randn(batch_size,100,device = device)  
            sample_labels = torch.randint(0,10,(batch_size,),device = device, dtype = torch.long)
            
            validity_label.fill_(1)
            
            fakes = gen(noise,sample_labels)
            pvalidity,plabels = disc(fakes)
            
            errG_val = validity_loss(pvalidity, validity_label)        
            errG_label = F.nll_loss(plabels, sample_labels)
            
            errG = errG_val + errG_label
            errG.backward()
            
            D_G_z2 = pvalidity.mean().item()
            
            optimG.step()
            if idx % 200 == 0:
                print("[{}/{}] [{}/{}] D_x: [{:.4f}] D_G: [{:.4f}/{:.4f}] G_loss: [{:.4f}] D_loss: [{:.4f}] D_label: [{:.4f}] "
                .format(epoch,epochs, idx, len(trainloader),D_x, D_G_z1,D_G_z2,errG,errD,
                        errD_real_label + errD_fake_label + errG_label))