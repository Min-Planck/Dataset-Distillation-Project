import torch 
from torch import nn 
import torch.optim as optim
import os
import time
from torchvision.utils import make_grid

from src.utils import IDatasetDistillation
from src.gan import get_cgan

class LogitLoss(nn.Module):
    def __init__(self):
        super(LogitLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, logits_real, logits_syn):
        return self.criterion(logits_real, logits_syn)

class DiM(IDatasetDistillation):
    def __init__(self, trainloader, testloader, device, opt):
        self.d_lambda = opt['d_lambda']
        self.trainloader = trainloader
        self.testloader = testloader
        self.gen, self.disc = get_cgan(num_channels=opt['channel'])
        self.opt = opt
        self.device = device

        gan_path = 'pretrained/gan/cgan'
        self.gen.load_state_dict(torch.load(f'{gan_path}/gen_{opt['dataset_name']}.pth', map_location=device))
        self.disc.load_state_dict(torch.load(f'{gan_path}/disc_{opt['dataset_name']}.pth', map_location=device))

        self.gen.to(device)
        self.disc.to(device)
    def train_generator(self): 
        from src.utils import get_random_model_from_model_pool
        start_time = time.time()

        optimG = optim.SGD(self.gen.parameters(), lr=self.opt['lr'])
        optimD = optim.SGD(self.disc.parameters(), lr=self.opt['lr'])
        
        validity_loss = nn.BCELoss()
        logit_loss = LogitLoss()

        real_labels = 0.7 + 0.3 * torch.rand(10, device=self.device)
        fake_labels = 0.3 * torch.rand(10, device=self.device)

        for epoch in range(self.opt['num_distill_epochs']):
            for idx, (images, labels) in enumerate(self.trainloader, 0):
                batch_size = images.size(0)
                images = images.to(self.device)
                labels = labels.to(self.device)

                real_label = real_labels[idx % 10]
                fake_label = fake_labels[idx % 10]

                if idx % 25 == 0:
                    real_label, fake_label = fake_label, real_label

       
                optimD.zero_grad()

                # Real images
                validity_label = torch.full((batch_size,), real_label, device=self.device)
                real_validity = self.disc(images, labels)
                errD_real = validity_loss(real_validity, validity_label)
                errD_real.backward()

                # Fake images
                noise = torch.randn(batch_size, 100, device=self.device)
                sample_labels = torch.randint(0, 10, (batch_size,), device=self.device, dtype=torch.long)
                fakes = self.gen(noise, sample_labels)

                validity_label.fill_(fake_label)
                fake_validity = self.disc(fakes.detach(), sample_labels)
                errD_fake = validity_loss(fake_validity, validity_label)
                errD_fake.backward()

                errD = errD_real + errD_fake
                optimD.step()

                # -----------------
                #  Train Generator
                # -----------------
                optimG.zero_grad()
                noise = torch.randn(batch_size, 100, device=self.device)
                sample_labels = torch.randint(0, 10, (batch_size,), device=self.device, dtype=torch.long)
                fakes = self.gen(noise, sample_labels)

                validity_label.fill_(1.0)
                fake_validity = self.disc(fakes, sample_labels)
                rand_model = get_random_model_from_model_pool(self.opt).to(self.device)
                with torch.no_grad():
                    logits_real = rand_model(images)
                    logits_syn = rand_model(fakes)

                    errG_logit_loss = logit_loss(logits_real, logits_syn)

                errG = validity_loss(fake_validity, validity_label) + errG_logit_loss * self.d_lambda
                errG.backward()
                optimG.step()

                # -----------------
                #  Logging
                # -----------------
                if idx % 200 == 0:
                    print(
                        f"[{epoch}/{self.opt['num_distill_epochs']}] [{idx}/{len(self.trainloader)}] "
                        f"G_loss: {errG:.4f} D_loss: {errD:.4f}"
                    )
        end_time = time.time()
        os.makedirs('pretrained/dim', exist_ok=True)
        torch.save(self.gen.state_dict(), f'pretrained/dim/gen_{self.opt["dataset_name"]}.pth')
        
        return end_time - start_time
    
    def evaluate(self, model, ipc: int) -> float:
        from src.utils import evaluate_gen_distill_method
        trained_gen = torch.load(f'pretrained/dim/gen_{self.opt["dataset_name"]}.pth', map_location=self.device)
        if trained_gen is None:
            raise ValueError("Generator model not found.")
        self.gen.load_state_dict(trained_gen)
        accuracy = evaluate_gen_distill_method(self.gen, model, ipc, 300, self.testloader, self.opt, self.device)
        return accuracy

    def generate_sample(self, ipc: int):
        from src.utils import showImage
        noise = torch.randn(10, 100, device=self.device)
        labels = torch.arange(0, 10, dtype=torch.long, device=self.device)
        trained_gen = torch.load(f'pretrained/dim/gen_{self.opt["dataset_name"]}.pth', map_location=self.device)
        if trained_gen is None:
            raise ValueError("Generator model not found.")
        self.gen.load_state_dict(trained_gen)  
        gen_images = self.gen(noise, labels).detach()
        showImage(make_grid(gen_images), save_=True, algo_name='dim')

