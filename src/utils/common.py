import torch 
import numpy as np 
from ..models import get_model_by_name
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from torch import nn, optim
from torchvision.utils import save_image, make_grid
import os

from ..algo.helper_class import Synthetic

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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

        syn_dataset = Synthetic(data_syn, targets_syn)
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

def generate_and_save_images_by_class(generator, step, num_classes, latent_dim, device, output_dir="./samples"):
    generator.eval()
    os.makedirs(output_dir, exist_ok=True)

    n_img_per_class = 10
    total = num_classes * n_img_per_class

    noise = torch.randn(total, latent_dim, device=device)
    labels = torch.arange(num_classes, device=device).repeat_interleave(n_img_per_class)

    onehot = torch.zeros(total, num_classes, device=device)
    onehot[torch.arange(total), labels] = 1
    noise[:, :num_classes] = onehot

    with torch.no_grad():
        fake_images = generator(noise)

    fake_images = (fake_images + 1) / 2

    grid = make_grid(fake_images, nrow=n_img_per_class, normalize=True)

    save_path = os.path.join(output_dir, f"step_{step:04d}.png")
    save_image(grid, save_path)
    print(f"[INFO] Saved generated images for {num_classes} classes at {save_path}")

    generator.train()


def calc_gradient_penalty(args, discriminator, img_real, img_syn, device):
    ''' Gradient penalty from Wasserstein GAN
    '''
    LAMBDA = 10
    n_size = img_real.shape[-1]
    batch_size = img_real.shape[0]
    n_channels = img_real.shape[1]

    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(img_real.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, n_channels, n_size, n_size)
    alpha = alpha.to(device)

    img_syn = img_syn.view(batch_size, n_channels, n_size, n_size)
    interpolates = alpha * img_real.detach() + ((1 - alpha) * img_syn.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates, _ = discriminator(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

def train_cgan(args, epochs, generator, discriminator, optim_g, optim_d, trainloader, criterion, device):
    '''The main training function for the generator
    '''
    for epoch in range(epochs): 
        generator.train()
        gen_losses = AverageMeter()
        disc_losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for batch_idx, (img_real, lab_real) in enumerate(trainloader):
            img_real = img_real.to(device)
            lab_real = lab_real.to(device)

            # train the generator
            discriminator.eval()
            optim_g.zero_grad()

            # obtain the noise with one-hot class labels
            current_batch_size = lab_real.size(0) # Get the actual batch size
            noise = torch.normal(0, 1, (current_batch_size, args['latent_dim']))
            lab_onehot = torch.zeros((current_batch_size, args['n_classes']))
            lab_onehot[torch.arange(current_batch_size), lab_real] = 1
            noise[torch.arange(current_batch_size), :args['n_classes']] = lab_onehot[torch.arange(current_batch_size)]
            noise = noise.to(device)

            img_syn = generator(noise)
            gen_source, gen_class = discriminator(img_syn)
            gen_source = gen_source.mean()
            gen_class = criterion(gen_class, lab_real)
            gen_loss = - gen_source + gen_class

            gen_loss.backward()
            optim_g.step()

            # train the discriminator
            discriminator.train()
            optim_d.zero_grad()
            lab_syn = torch.randint(args['n_classes'], (current_batch_size,)) # Use actual batch size
            noise = torch.normal(0, 1, (current_batch_size, args['latent_dim'])) # Use actual batch size
            lab_onehot = torch.zeros((current_batch_size, args['n_classes'])) # Use actual batch size
            lab_onehot[torch.arange(current_batch_size), lab_syn] = 1
            noise[torch.arange(current_batch_size), :args['n_classes']] = lab_onehot[torch.arange(current_batch_size)]
            noise = noise.to(device)
            lab_syn = lab_syn.to(device)

            with torch.no_grad():
                img_syn = generator(noise)

            disc_fake_source, disc_fake_class = discriminator(img_syn)
            disc_fake_source = disc_fake_source.mean()
            disc_fake_class = criterion(disc_fake_class, lab_syn)

            disc_real_source, disc_real_class = discriminator(img_real)
            acc1, acc5 = accuracy(disc_real_class.data, lab_real, topk=(1, 5))
            disc_real_source = disc_real_source.mean()
            disc_real_class = criterion(disc_real_class, lab_real)

            gradient_penalty = calc_gradient_penalty(args, discriminator, img_real, img_syn, device)

            disc_loss = disc_fake_source - disc_real_source + disc_fake_class + disc_real_class + gradient_penalty
            disc_loss.backward()
            optim_d.step()

            gen_losses.update(gen_loss.item())
            disc_losses.update(disc_loss.item())
            top1.update(acc1.item())
            top5.update(acc5.item())

            if (batch_idx + 1) % 100 == 0:
                print('[Train Epoch {} Iter {}] G Loss: {:.3f}({:.3f}) D Loss: {:.3f}({:.3f}) D Acc: {:.3f}({:.3f})'.format(
                    epoch, batch_idx + 1, gen_losses.val, gen_losses.avg, disc_losses.val, disc_losses.avg, top1.val, top1.avg)
                )

            batch_dones = epoch * len(trainloader) + batch_idx
            if batch_dones % args['sample_interval'] == 0:
                generate_and_save_images_by_class(generator, batch_dones, args['n_classes'], args['latent_dim'], device)
    
    save_path = f"./pretrained/gan/cgan_{args['dataset_name']}"
    os.makedirs(save_path, exist_ok=True)
    torch.save(generator.state_dict(), f"{save_path}/generator.pt")
    torch.save(discriminator.state_dict(), f"{save_path}/discriminator.pt")