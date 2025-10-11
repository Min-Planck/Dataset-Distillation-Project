# DiM: Distilling Dataset into Generative Model - https://arxiv.org/pdf/2303.04707
import os
import torch
from ..models import get_gan, get_random_model_from_model_pool
from ..utils import train_acgan, train_cgan, LogitLossMSE, evaluate_dim_method, generate_sample_dim
from torch.autograd import Variable
from tqdm import tqdm
from torch import nn
import numpy as np 
from torchvision.utils import save_image
from .interface import IDatasetDistillation

class GDCL(IDatasetDistillation):

    def __init__(self, 
                 dataset_name: str, 
                 opt, 
                 trainloader, 
                 testloader, 
                 gan_model_name: str = 'acgan', 
                 pretrained_gan_path: str = None, 
                 use_pretrained: bool = True, 
                 device: str = 'cpu'):
        
        self.testloader = testloader
        self.device = device 
        self.opt = opt
        self.dataset_name = dataset_name

        gen, disc = get_gan(gan_model_name, dataset_name, opt, use_pretrained=use_pretrained, pretrained_gan_path=pretrained_gan_path)

        if not use_pretrained:
            if gan_model_name.lower() == 'cgan':
                print('Training cgan from scratch')
                train_cgan(gan_model_name, gen, disc, trainloader, opt, device)
            elif gan_model_name.lower() == 'acgan':
                print('Training acgan from scratch')
                train_acgan(gan_model_name, gen, disc, trainloader, opt, device)
            else: 
                raise ValueError(f"GAN model {gan_model_name} is not supported for training from scratch.")

        self.generator = gen.to(self.deviceF)
        self.discriminator = disc.to(self.device)

        self.distilled_model_path = self._get_distilled_model_path()

        self.logit_loss_fn = LogitLossMSE().to(self.device)

    def presentation(self):
        return 'DiM'
    
    def _get_distilled_model_path(self):
        model_dir = "distilled_models"
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, self.presentation(), f"{self.dataset_name}_distilled_generator.pth")
    
    def save_distilled_model(self):
        torch.save(self.generator.state_dict(), self.distilled_model_path)
        print(f"Distilled generator saved to: {self.distilled_model_path}")

    def load_distilled_model(self):
        if not os.path.exists(self.distilled_model_path):
            raise FileNotFoundError(f"Không tìm thấy mô hình tại: {self.distilled_model_path}. Hãy chạy `distillation()` trước.")

        state_dict = torch.load(self.distilled_model_path, map_location=self.device)
        self.generator.load_state_dict(state_dict)
        self.generator.to(self.device)
        self.generator.eval()

        print(f"Loaded distilled generator from: {self.distilled_model_path}")
            
    def distillation(self): 
        self.Q = self.opt['num_distill_epochs']
        lr = self.opt['lr']
        b1 = self.opt['b1'] 
        b2 = self.opt['b2']

        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        
        FloatTensor = torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if self.device == 'cuda' else torch.LongTensor

        print("Starting distillation with pretrained GAN...")

        for epoch in range(self.num_epochs):
            epoch_g_loss = 0.0
            epoch_m_loss = 0.0
            epoch_total_loss = 0.0
            num_batches = 0

            self.generator.train()
            self.discriminator.eval()  

            for i, (real_imgs, real_labels) in enumerate(tqdm(self.trainloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")):
                batch_size = real_imgs.size(0)
                real_imgs = real_imgs.to(self.device)
                real_labels = real_labels.to(self.device)

                valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

                optimizer_G.zero_grad()

                # --- Tính Lg (GAN Loss) ---
                z_for_gan_loss = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, self.opt['latent_dim']))))
                gen_labels_for_gan = Variable(LongTensor(np.random.randint(0, self.opt['n_classes'], batch_size)))
                gen_imgs_for_gan = self.generator(z_for_gan_loss, gen_labels_for_gan)

                validity_fake, _ = self.discriminator(gen_imgs_for_gan)
                Lg = nn.BCELoss()(validity_fake, valid) 

                # --- Tính Lm (Logits Matching Loss) ---
                classification_model = get_random_model_from_model_pool().to(self.device)
                classification_model.eval()

                z_for_matching_loss = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, self.opt['latent_dim']))))
                synthetic_imgs_for_matching = self.generator(z_for_matching_loss, real_labels)

                with torch.no_grad():
                    real_img_logits = classification_model(real_imgs)
                distilled_img_logits = classification_model(synthetic_imgs_for_matching)  

                Lm = self.logit_loss_fn(distilled_img_logits, real_img_logits)

                # --- Tính tổng loss và cập nhật Generator ---
                Ltotal = Lg + self.lambda_ * Lm
                Ltotal.backward()
                optimizer_G.step()

                epoch_g_loss += Lg.item()
                epoch_m_loss += Lm.item()
                epoch_total_loss += Ltotal.item()
                num_batches += 1

            if num_batches > 0:
                avg_g_loss = epoch_g_loss / num_batches
                avg_m_loss = epoch_m_loss / num_batches
                avg_total_loss = epoch_total_loss / num_batches
                print(f"[Epoch {epoch+1}/{self.num_epochs}] "
                      f"[Avg G Loss: {avg_g_loss:.4f}] "
                      f"[Avg M Loss: {avg_m_loss:.4f}] "
                      f"[Avg Total Loss: {avg_total_loss:.4f}]")

            if (epoch + 1) % 10 == 0 or epoch == self.num_epochs - 1:
                self.save_checkpoint(epoch + 1)

        self.save_distilled_model()
        print("Distillation completed and model saved.")

    def save_checkpoint(self, epoch):
        """Lưu trọng số của generator."""
        os.makedirs("checkpoints", exist_ok=True)
        save_path = f"checkpoints/{self.presentation()}/generator_epoch_{epoch}.pth"
        torch.save(self.generator.state_dict(), save_path)
        print(f"Checkpoint saved at epoch {epoch}: {save_path}")

    def generate_sample(self, ipc=10, save_root="generated_samples"):
        self.load_distilled_model()
        generate_sample_dim(self.generator, self.dataset_name, ipc, self.opt['latent_dim'], save_root, self.opt['n_classes'], self.device)

    def evaluate(self, eval_model, num_train: int = 100, ipc: int = 10):
        self.load_distilled_model()
        return evaluate_dim_method(self.generator, eval_model, self.testloader, num_train, self.opt['n_classes'], ipc, self.device)