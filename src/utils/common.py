import torch 
import numpy as np 
from models import get_model_by_name
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from torch import nn, optim
from torchvision.utils import save_image
import os

from ..algo.helper_class import SynThetic

def sample_image(generator, n_row, batches_done, device, latent_dim):
    os.makedirs("images", exist_ok=True)

    # Sample noise
    z = torch.randn(n_row ** 2, latent_dim, device=device)
    # labels: mỗi hàng là 1 class: [0,0,...,1,1,..., n_row-1,...]
    labels = torch.tensor([i for i in range(n_row) for _ in range(n_row)], dtype=torch.long, device=device)

    generator.eval()
    with torch.no_grad():
        gen_imgs = generator(z, labels)
    generator.train()

    save_image(gen_imgs, f"images/{batches_done}.png", nrow=n_row, normalize=True)



def get_images(indices_class, images_all, c, n): # get random n images from class c
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    return images_all[idx_shuffle]

def evaluate_dim_method(opt, generator, eval_model, testloader, num_train_epochs, n_classes, ipc, device): 
    FloatTensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if device == 'cuda' else torch.LongTensor

    print("Generating synthetic dataset for evaluation...")
    num_classes = n_classes

    synthetic_images = []
    synthetic_labels = []

    with torch.no_grad():
        for class_id in range(num_classes):
            for _ in tqdm(range(ipc), desc=f"Class {class_id}"):
                z = Variable(FloatTensor(np.random.normal(0, 1, (1, opt['latent_dim']))))
                label = Variable(LongTensor([class_id]))
                img = generator(z, label)
                synthetic_images.append(img.cpu())
                synthetic_labels.append(label.cpu())

    synthetic_images = torch.cat(synthetic_images, dim=0)
    synthetic_labels = torch.cat(synthetic_labels, dim=0).squeeze()
    synthetic_dataset = torch.utils.data.TensorDataset(synthetic_images, synthetic_labels)
    synthetic_loader = torch.utils.data.DataLoader(synthetic_dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(eval_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    print("Training evaluation model on synthetic data...")
    eval_model.train()
    for epoch in range(num_train_epochs):
        for imgs, labels in synthetic_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = eval_model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    eval_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = eval_model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"=== Evaluation Result ===")
    print(f"Accuracy on test set: {accuracy:.2f}%")
    return accuracy

def generate_sample_dim(generator, dataset_name, ipc, latent_dim, save_root, n_classes, device):
    FloatTensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if device == 'cuda' else torch.LongTensor

    base_dir = os.path.join(save_root, "dim", dataset_name)
    os.makedirs(base_dir, exist_ok=True)

    print(f"Generating {ipc} images per class for {n_classes} classes...")

    with torch.no_grad():
        for class_id in range(n_classes):
            class_dir = os.path.join(base_dir, f"class_{class_id}")
            os.makedirs(class_dir, exist_ok=True)

            for img_idx in tqdm(range(ipc), desc=f"Class {class_id}"):
                z = Variable(FloatTensor(np.random.normal(0, 1, (1, latent_dim))))
                label = Variable(LongTensor([class_id]))
                gen_img =generator(z, label)

                save_path = os.path.join(class_dir, f"img_{img_idx}.png")
                save_image(gen_img.data, save_path, normalize=True)

    print(f"Generated and Saved {n_classes * ipc} images to: {base_dir}")

def evaluate_dii_method(model_name, opt, synthetic_datas, testloader, batch_size, ipc, num_train_epochs, n_classes, device): 
    accuracies = []
    targets_syn = torch.tensor([np.ones(ipc) * i for i in range(n_classes)], dtype=torch.long, requires_grad=False,  device=device).view(-1)

    for data_syn in synthetic_datas:
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        net = get_model_by_name(model_name, opt).to(device)
        net.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        optimizer.zero_grad()

        syn_dataset = SynThetic(data_syn, targets_syn)
        trainloader = DataLoader(syn_dataset, batch_size=batch_size, shuffle=True)

        for it in range(num_train_epochs): 
                
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)

                prediction = net(images)
                loss = loss_fn(prediction, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        net.eval()
        with torch.inference_mode():
            correct = 0
            total = 0
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
                
        accuracies.append(accuracy)
            
    return sum(accuracies) / len(accuracies)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def train_cgan(model_name, generator, discriminator, opt, dataloader, device):
    lr = opt['lr']
    b1 = opt['b1']
    b2 = opt['b2']
    num_epochs = opt['num_epochs']
    n_classes = opt['n_classes']
    latent_dim = opt['latent_dim']
    sample_interval = opt['sample_interval']

    adversarial_loss = nn.MSELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    FloatTensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if device == 'cuda' else torch.LongTensor

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)

    for epoch in range(num_epochs):

        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0 ), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()
            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            if i % sample_interval == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, num_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )
            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                sample_image(generator, n_row=10, batches_done=batches_done, FloatTensor=FloatTensor, LongTensor=LongTensor, latent_dim=latent_dim)

    os.makedirs(f"pretrained_models/gan/cgan_{opt['dataset_name']}", exist_ok=True)
    torch.save(generator.state_dict(), f"pretrained_models/gan/cgan_{opt['dataset_name']}/{model_name}_generator.pth")

def train_acgan(model_name, generator, discriminator, dataloader, opt, device):
    lr = opt['lr']
    b1 = opt['b1']
    b2 = opt['b2']
    num_epochs = opt['n_epochs']
    n_classes = opt['n_classes']
    latent_dim = opt['latent_dim']
    sample_interval = opt['sample_interval']

    adversarial_loss = torch.nn.BCELoss()
    auxiliary_loss = torch.nn.CrossEntropyLoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    FloatTensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if device == 'cuda' else torch.LongTensor

    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)
    auxiliary_loss.to(device)

    flip_prob = 0.05
    
    for epoch in range(num_epochs):

        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            if torch.rand(1).item() < flip_prob:
                valid, fake = fake, valid
            
            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity, pred_label = discriminator(gen_imgs)
            g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))

            g_loss.backward()
            optimizer_G.step()
            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            real_pred, real_aux = discriminator(real_imgs)
            d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

            # Loss for fake images
            fake_pred, fake_aux = discriminator(gen_imgs.detach())
            d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            # Calculate discriminator accuracy
            pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
            gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
            d_acc = np.mean(np.argmax(pred, axis=1) == gt)

            d_loss.backward()
            optimizer_D.step()
            if i % sample_interval == 0:
                 print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, num_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )
            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                sample_image(generator, n_row=n_classes, batches_done=batches_done, device=device, latent_dim=latent_dim)

    os.makedirs(f"pretrained_models/gan/acgan_{opt['dataset_name']}", exist_ok=True)
    torch.save(generator.state_dict(), f"pretrained_models/gan/acgan_{opt['dataset_name']}/{model_name}_generator.pth")
