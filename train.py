import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import Config
from util import SVGPNGDataset, EarlyStopping
from network import SVGEncoder, SVGDecoder, SVGPNGModel
from torchvision import transforms
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

epochs = 100
lr = 0.001
patience = 5
batch_size = 16
image_size = 24
num_epochs = 10

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

dataset = SVGPNGDataset(num_items=10000, max_length_limit=1024, transform=transform)
vocab_size = max(dataset.get_max_svg_length(), dataset.max_length_limit)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("vocab_size: ", vocab_size)
print(len(train_dataset), len(test_dataset))

criterion_svg_cmd = nn.CrossEntropyLoss().to(device)
criterion_svg_coordinate = nn.L1Loss().to(device)

criterion_png = nn.MSELoss().to(device)
criterion_z = nn.L1Loss().to(device)

model = Unet(dim=64,
             dim_mults=(1, 2, 4, 8),
             flash_attn=True).to(device)

diffusion = GaussianDiffusion(model,
                              image_size=image_size,
                              timesteps=1000).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)

early_stopping = EarlyStopping(patience=2 * patience, delta=0)
best_loss = 10000

# 학습 루프
for epoch in range(num_epochs):
    train_loss, test_loss = 0, 0

    diffusion.train()
    for batch in tqdm.tqdm(train_loader, desc='train'):
        loss = diffusion(batch['png_tensor'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    diffusion.eval()
    for batch in train_loader:
        loss = diffusion(batch['png_tensor'])
        test_loss += loss.item()
    test_loss /= len(test_loader)

    print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    scheduler.step(test_loss)

    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(diffusion.state_dict(), os.path.join(dataset.path, 'best_encoder.pth'))

    #if early_stopping(test_loss, diffusion, os.path.join(dataset.path, 'best_encoder.pth')):
    #    print("Early stopping")
    #    break

sampled_images = diffusion.sample(batch_size=4)

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
