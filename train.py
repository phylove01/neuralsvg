import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util import SVGPNGDataset, EarlyStopping
from torchvision import transforms
import tqdm
from dotenv import load_dotenv
from vqgan_pytorch.training_vqgan import TrainVQGAN

from config import Config
import numpy as np
import torch.nn.functional as F
from torchvision import utils as vutils
import warnings


config = Config()
args = config.args
device = config.device
load_dotenv()
warnings.filterwarnings("ignore")

print('datapath: ', config.path)
print('device: ', config.device)

epochs = int(os.getenv('epochs'))
lr = float(os.getenv('lr'))
patience = int(os.getenv('patience'))
batch_size = int(os.getenv('batch_size'))
image_size = int(os.getenv('image_size'))
num_item = int(os.getenv('num_item'))
max_length_limit = int(os.getenv('max_length_limit'))

mean = [0.8664]
std = [0.0376]

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

dataset = SVGPNGDataset(num_items=num_item, max_length_limit=max_length_limit, transform=transform)
vocab_size = max(dataset.get_max_svg_length(), dataset.max_length_limit)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("vocab_size: ", vocab_size)
print(len(train_dataset), len(test_dataset))

train = TrainVQGAN(args)
model = train.vqgan

criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)
early_stopping = EarlyStopping(patience=patience, delta=0)
best_loss = float('infinity')
best_model = None

for epoch in range(epochs):
    train_loss, train_vq_loss, train_gan_loss = 0, 0, 0
    model.train()

    idx = 0
    for batch in tqdm.tqdm(train_loader, desc='train at epoch'+str(epoch)):
        target = batch['png_tensor'].to(device)
        decoded_images, _, q_loss, _ = model(target)

        disc_real = train.discriminator(target)
        disc_fake = train.discriminator(decoded_images)
        disc_factor = model.adopt_weight(args.disc_factor, epoch * len(train_loader) + idx,
                                                          threshold=args.disc_start)

        perceptual_loss = train.perceptual_loss(target, decoded_images)
        rec_loss = torch.abs(target - decoded_images)
        perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss

        perceptual_rec_loss = perceptual_rec_loss.mean()
        g_loss = -torch.mean(disc_fake)
        λ = model.calculate_lambda(perceptual_rec_loss, g_loss)
        vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

        d_loss_real = torch.mean(F.relu(1. - disc_real))
        d_loss_fake = torch.mean(F.relu(1. + disc_fake))
        gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

        train.opt_vq.zero_grad()
        vq_loss.backward(retain_graph=True)

        train.opt_disc.zero_grad()
        gan_loss.backward()

        train.opt_vq.step()
        train.opt_disc.step()

        real_fake_images = torch.cat((target[:4], decoded_images.add(1).mul(0.5)[:4]))
        if idx % 10 ==0:
            vutils.save_image(real_fake_images, os.path.join(config.save_path, f"train_{epoch}_{idx}.jpg"), nrow=4)

        VQ_Loss = np.round(vq_loss.cpu().detach().numpy(), 5).item()
        GAN_Loss = np.round(gan_loss.cpu().detach().numpy(), 5).item()

        idx += 1
        train_vq_loss += VQ_Loss
        train_gan_loss += GAN_Loss

    train_vq_loss /= len(train_loader)
    train_gan_loss /= len(train_loader)
    train_loss = train_vq_loss + train_gan_loss
    scheduler.step(train_loss)

    print(
        f"Epoch [{epoch + 1}/{epochs}], Train VQ Loss: {train_vq_loss:.4f}, Train GAN Loss: {train_gan_loss:.4f}, Train Loss: {train_loss:.4f}")

    if train_loss < best_loss:
        best_loss = train_loss

    if early_stopping(train_loss, model):
        print("Early stopping")
        break

