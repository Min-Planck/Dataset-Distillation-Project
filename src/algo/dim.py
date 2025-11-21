import torch 
from torch import nn 
import torch.optim as optim
import os
import torch.nn.functional as F

import time
from torchvision.utils import make_grid

from src.utils import IDatasetDistillation
from src.gan import get_cgan, train_cgan

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


    def train(self, gen, disc, optimG, optimD, validity_loss, trainloader): 
        from src.utils import get_random_model_from_model_pool

        model = get_random_model_from_model_pool().to(self.device)
        model_optim = torch.optim.SGD(model.parameters(), lr=self.opt['lr_eval_net'], momentum=self.opt['momentum'], weight_decay=self.opt['weight_decay'])
        model.train()

        real_labels = torch.ones(10, device=self.device)
        fake_labels = torch.zeros(10, device=self.device)

        match_loss = LogitLoss()
        for idx, (images, labels) in enumerate(trainloader, 0):
            batch_size = images.size(0)
            images = images.to(self.device)
            labels = labels.to(self.device)

            real_label = real_labels[idx % 10]
            fake_label = fake_labels[idx % 10]

            optimD.zero_grad()

            # Real images
            validity_label = torch.full((batch_size,), real_label, device=self.device)
            real_validity = disc(images, labels)
            errD_real = validity_loss(real_validity, validity_label)
            errD_real.backward()

            # Fake images
            noise = torch.randn(batch_size, 100, device=self.device)
            sample_labels = torch.randint(0, 10, (batch_size,), device=self.device, dtype=torch.long)
            fakes = gen(noise, sample_labels)

            validity_label.fill_(fake_label)
            fake_validity = disc(fakes.detach(), sample_labels)
            errD_fake = validity_loss(fake_validity, validity_label)
            errD_fake.backward()
            optimD.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            optimG.zero_grad()
            noise = torch.randn(batch_size, 100, device=self.device)
            sample_labels = torch.randint(0, 10, (batch_size,), device=self.device, dtype=torch.long)
            fakes = gen(noise, sample_labels)

            validity_label.fill_(1.0)
            fake_validity = disc(fakes, sample_labels)
            errG = validity_loss(fake_validity, validity_label)

            self.train_match_model(model, self.opt['epochs_match_train'], model_optim, trainloader, nn.CrossEntropyLoss())
            output_real = F.log_softmax(model(images), dim=1)
            output_syn = F.log_softmax(model(fakes), dim=1)

            match_loss = match_loss(output_real, output_syn)
            errG += self.d_lambda * match_loss  
            errG.backward()
            optimG.step()
         
    @staticmethod
    def train_match_model(model, epochs_match_train, optim_model, trainloader, criterion):
        for batch_idx, (img, lab) in enumerate(trainloader):
            if batch_idx == epochs_match_train:
                break

            img = img.cuda()
            lab = lab.cuda()

            output = model(img)
            loss = criterion(output, lab)

            optim_model.zero_grad()
            loss.backward()
            optim_model.step()
    
    def train_generator(self): 
        start_time = time.time()
        optimG = optim.Adam(self.gen.parameters(), self.opt['lr'], betas=(0.5, 0.999))
        optimD = optim.Adam(self.disc.parameters(), self.opt['lr'], betas=(0.5, 0.999))
        validity_loss = nn.BCELoss()

        self.gen, self.disc = train_cgan(self.gen, self.disc, self.trainloader, self.opt['gan_epochs'], self.opt['gan_lr'], self.device)
        for epoch in range(1, self.opt['num_distill_epochs'] + 1): 
            self.train(self.gen, self.disc, optimG, optimD, validity_loss, self.trainloader)

        end_time = time.time()
        return end_time - start_time
    
    def evaluate(self, model, ipc: int) -> float:
        from src.utils import evaluate_gen_distill_method
        trained_gen = torch.load(f'pretrained/dim/gen_{self.opt["dataset_name"]}.pth', map_location=self.device)
        if trained_gen is None:
            raise ValueError("Generator model not found.")
        self.gen.load_state_dict(trained_gen)
        accuracy, distill_time, eval_time = evaluate_gen_distill_method(self.gen, model, ipc, 300, self.testloader, self.opt, self.device)
        return accuracy, distill_time, eval_time

    def generate_sample(self, ipc: int):
        from src.utils import showImage
        noise = torch.randn(10, 100, device=self.device)
        labels = torch.arange(0, 10, dtype=torch.long, device=self.device)
        trained_gen = torch.load(f'pretrained/dim/gen_{self.opt["dataset_name"]}.pth', map_location=self.device)
        if trained_gen is None:
            raise ValueError("Generator model not found.")
        self.gen.load_state_dict(trained_gen)  
        gen_images = self.gen(noise, labels).detach()
        showImage(make_grid(gen_images), save_=True, algo_name=f'dim_{ipc}')

