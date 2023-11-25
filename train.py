import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util import SVGPNGDataset, EarlyStopping
from torchvision import transforms
import tqdm
from dotenv import load_dotenv
from network import Model
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

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
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

model = Model(os.getenv('model'), device=device)
vg = TrainVQGAN(args)

criterion = torch.nn.MSELoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)
early_stopping = EarlyStopping(patience=patience, delta=0)
best_loss = float('infinity')
best_model = None

for epoch in range(epochs):
    train_loss, test_loss = 0, 0
    model.train()

    idx = 0
    for batch in tqdm.tqdm(train_loader, desc='train'):
        target = batch['png_tensor'].to(device)
        if model.form == 'diffusion':
            loss = model(target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elif model.form == 'vqgan':
            decoded_images, _, q_loss = vg.vqgan(target)

            disc_real = vg.discriminator(target)
            disc_fake = vg.discriminator(decoded_images)
            disc_factor = vg.vqgan.adopt_weight(args.disc_factor, epoch * len(target) + idx,
                                                              threshold=args.disc_start)

            perceptual_loss = vg.perceptual_loss(target, decoded_images)
            rec_loss = torch.abs(target - decoded_images)
            perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
            perceptual_rec_loss = perceptual_rec_loss.mean()
            g_loss = -torch.mean(disc_fake)

            λ = vg.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
            vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss
            d_loss_real = torch.mean(F.relu(1. - disc_real))
            d_loss_fake = torch.mean(F.relu(1. + disc_fake))
            gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

            vg.opt_vq.zero_grad()
            vq_loss.backward(retain_graph=True)

            vg.opt_disc.zero_grad()
            gan_loss.backward()

            vg.opt_vq.step()
            vg.opt_disc.step()

            real_fake_images = torch.cat((target[:4], target[:4]))#decoded_images.add(1).mul(0.5)[:6]))
            vutils.save_image(real_fake_images, os.path.join(config.save_path, f"train_{epoch}_{idx}.jpg"), nrow=4)

            VQ_Loss = np.round(vq_loss.cpu().detach().numpy().item(), 5),
            GAN_Loss = np.round(gan_loss.cpu().detach().numpy().item(), 5)
            #print('VQ_Loss: ', VQ_Loss, 'GAN_Loss: ', GAN_Loss)

            loss = VQ_Loss + GAN_Loss
            idx += 1

        train_loss += loss
    train_loss /= len(train_loader)

    model.eval()
    for batch in train_loader:
        target = batch['png_tensor'].to(device)
        if model.form == 'diffusion':
            loss = model(target)
            test_loss /= len(test_loader)
        elif model.form == 'vqgan':
            with torch.no_grad():
                decoded_images, _, q_loss = vg.vqgan(target)
                real_fake_images = torch.cat((target[:4], decoded_images.add(1).mul(0.5)[:4]))
                vutils.save_image(real_fake_images, os.path.join(config.save_path, f"test_{epoch}_{idx}.jpg"), nrow=4)
    if model.form == 'vqgan':
        test_loss = train_loss

    scheduler.step(test_loss)

    if test_loss < best_loss:
        best_loss = test_loss
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    if early_stopping(test_loss, model):
        print("Early stopping")
        break

# model = SVGPNGModel(vocab_size=vocab_size, init_rff=True, device=device)

# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)
#
# early_stopping = EarlyStopping(patience=2 * patience, delta=0)
# best_loss = 10000
#
# for epoch in range(epochs):
#     model.train()
#     train_loss = 0.0
#     for data in train_loader:
#         svg_images, png_images = data['svg_tensor'], data['png_tensor']
#         svg_images, png_images = svg_images.to(device), png_images.to(device)  # 이미지를 device로 이동
#
#         svg_preds, png_preds, zs, zp = model(svg_images, png_images)
#         loss1_1 = criterion_svg_cmd(svg_preds[0,:], svg_images[0,:])
#         loss1_2 = criterion_svg_coordinate(svg_preds[1,:], svg_images[1,:])
#         loss1_3 = criterion_svg_coordinate(svg_preds[2,:], svg_images[2,:])
#
#         loss2 = criterion_png(png_preds, png_images)
#         loss3 = criterion_z(zs, zp)
#         loss = loss1_1 + loss1_2 + loss1_3 + loss2 + loss3
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#
#     train_loss /= len(train_loader)
#
#     model.eval()
#     test_loss = 0.0
#     with torch.no_grad():
#         for data in test_loader:
#             model.eval()
#             svg_images, png_images = data['svg_tensor'], data['png_tensor']
#             svg_images, png_images = svg_images.to(device), png_images.to(device)  # 이미지를 device로 이동
#
#             svg_preds, png_preds, zs, zp = model(svg_images, png_images)
#             loss1_1 = criterion_svg_cmd(svg_preds[0, :], svg_images[0, :])
#             loss1_2 = criterion_svg_coordinate(svg_preds[1, :], svg_images[1, :])
#             loss1_3 = criterion_svg_coordinate(svg_preds[2, :], svg_images[2, :])
#
#             loss2 = criterion_png(png_preds, png_images)
#             loss3 = criterion_z(zs, zp)
#             loss = loss1_1 + loss1_2 + loss1_3 + loss2 + loss3
#
#             test_loss += loss.item()
#
#     test_loss /= len(test_loader)
#     print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
#
#     scheduler.step(test_loss)
#
#     if test_loss < best_loss:
#         best_loss = test_loss
#         torch.save(model.state_dict(),  os.path.join(dataset.path,'best_encoder.pth'))
#
#
#     if early_stopping(test_loss, model, os.path.join(dataset.path,'best_model.pth')):
#         print("Early stopping")
#         break
