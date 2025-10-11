# Data-to-Model Distillation: Data-Efficient Learning Framework  -https://arxiv.org/abs/2411.12841
import os
import torch
from ..models import get_gan, get_random_model_from_model_pool
from ..utils import train_cgan, LogitLossMSE, LogitLossKLDiv, evaluate_dim_method, generate_sample_dim
from torch.autograd import Variable
from tqdm import tqdm
from torch import nn
import numpy as np 
from .interface import IDatasetDistillation

class D2M(IDatasetDistillation):
    def __init__(self, 
                 dataset_name: str, 
                 opt, 
                 trainloader, 
                 testloader, 
                 gan_model_name: str = 'acgan', 
                 pretrained_gan_path: str = None, 
                 use_pretrained: bool = True, 
                 device: str = 'cpu'):

        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.opt = opt
        self.num_epochs = opt['num_distill_epochs']
        self.lambda_ = opt.get('lambda', 10.0)  # Default 10.0 as per paper
        self.temperature = opt.get('temperature', 4.0)  # Default 4.0
        self.dataset_name = dataset_name

        gen, disc = get_gan(gan_model_name, dataset_name, opt, use_pretrained=use_pretrained, pretrained_gan_path=pretrained_gan_path)

        if not use_pretrained:
            if gan_model_name.lower() == 'cgan':
                print('Training cgan from scratch')
                optim_g = torch.optim.Adam(gen.parameters(), lr=opt['lr'], betas=(opt['b1'], opt['b2']))
                optim_d = torch.optim.Adam(disc.parameters(), lr=opt['lr'], betas=(opt['b1'], opt['b2']))
                train_cgan(opt, self.opt['n_epochs'], gen, disc, optim_g, optim_d, trainloader, nn.CrossEntropyLoss(), device)
            else: 
                raise ValueError(f"GAN model {gan_model_name} is not supported for training from scratch.")

        self.generator = gen.to(self.device)
        self.discriminator = disc.to(self.device) 

        self.prediction_loss_fn = LogitLossKLDiv().to(self.device)  # KL Divergence cho LPM

        self.distilled_model_path = self._get_distilled_model_path()

    def presentation(self):
        return "D2M"

    def _get_distilled_model_path(self):
        model_dir = "distilled_models"
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, self.presentation(), f"{self.dataset_name}_distilled_generator.pth")

    def _compute_channel_attention(self, feature_map):
        return torch.mean(feature_map, dim=(2, 3))  # Global Average Pooling

    def distillation(self):
        lr = self.opt.get('lr', 0.0001)
        optimizer_G = torch.optim.SGD(self.generator.parameters(), lr=lr)  

        FloatTensor = torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor
        print("Starting D2M distillation with ACGAN...")

        for epoch in range(self.num_epochs):
            epoch_em_loss = 0.0
            epoch_pm_loss = 0.0
            epoch_total_loss = 0.0
            num_batches = 0

            self.generator.train()

            for i, (real_imgs, real_labels) in enumerate(tqdm(self.trainloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")):
                batch_size = real_imgs.size(0)
                real_imgs = real_imgs.to(self.device)
                real_labels = real_labels.to(self.device)

                optimizer_G.zero_grad()

                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, self.opt['latent_dim']))))
                synthetic_imgs = self.generator(z, real_labels)

                classification_model = get_random_model_from_model_pool(self.opt).to(self.device)
                classification_model.eval()

                try:
                    real_logits, real_features = classification_model.get_features_after_forward(real_imgs)
                    synth_logits, synth_features = classification_model.get_features_after_forward(synthetic_imgs)
                except AttributeError:
                    raise NotImplementedError("Mô hình phân loại cần có phương thức `get_features_after_forward`.")

                # --- Tính LEM (Embedding Matching Loss) ---
                LEM = 0.0
                num_layers = len(real_features)
                for l in range(num_layers):
                    if l < num_layers - 1:  
                        real_att = self._compute_channel_attention(real_features[l])
                        synth_att = self._compute_channel_attention(synth_features[l])
                        layer_loss = nn.MSELoss()(synth_att, real_att)
                    else:  
                        layer_loss = LogitLossMSE(synth_features[l], real_features[l])
                    LEM += layer_loss

                # --- Tính LPM (Prediction Matching Loss) ---
                real_probs = nn.functional.softmax(real_logits / self.temperature, dim=1)
                synth_log_probs = nn.functional.log_softmax(synth_logits / self.temperature, dim=1)
                LPM = nn.KLDivLoss(reduction='batchmean')(synth_log_probs, real_probs)

                # --- Tính tổng loss và cập nhật ---
                Ltotal = LEM + self.lambda_ * LPM
                Ltotal.backward()
                optimizer_G.step()

                epoch_em_loss += LEM.item()
                epoch_pm_loss += LPM.item()
                epoch_total_loss += Ltotal.item()
                num_batches += 1

            if num_batches > 0:
                avg_em_loss = epoch_em_loss / num_batches
                avg_pm_loss = epoch_pm_loss / num_batches
                avg_total_loss = epoch_total_loss / num_batches
                print(f"[Epoch {epoch+1}/{self.num_epochs}] "
                      f"[Avg EM Loss: {avg_em_loss:.4f}] "
                      f"[Avg PM Loss: {avg_pm_loss:.4f}] "
                      f"[Avg Total Loss: {avg_total_loss:.4f}]")

            if (epoch + 1) % 10 == 0:
                self.save_distilled_model()

        self.save_distilled_model()  
        print("D2M Distillation completed.")

    def save_distilled_model(self):
        torch.save(self.generator.state_dict(), self.distilled_model_path)
        print(f"D2M generator saved to: {self.distilled_model_path}")

    def load_distilled_model(self):
        if not os.path.exists(self.distilled_model_path):
            raise FileNotFoundError(f"Không tìm thấy mô hình D2M tại: {self.distilled_model_path}")

        state_dict = torch.load(self.distilled_model_path, map_location=self.device)
        self.generator.load_state_dict(state_dict)
        self.generator.to(self.device)
        self.generator.eval()
        print(f"Loaded D2M generator from: {self.distilled_model_path}")

    def generate_sample(self, ipc=10, save_root="generated_samples"):
        self.load_distilled_model()
        generate_sample_dim(self.generator, self.dataset_name, ipc, self.opt['latent_dim'], save_root, self.opt['n_classes'], self.device)

    def evaluate(self, eval_model, num_train: int = 100, ipc: int = 10):
        self.load_distilled_model()
        return evaluate_dim_method(self.generator, eval_model, self.testloader, num_train, self.opt['n_classes'], ipc, self.device)