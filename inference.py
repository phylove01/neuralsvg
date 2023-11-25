import seaborn as sns
import torch
from dotenv import load_dotenv
import os
from network import Model
from torchvision import transforms
from util import SVGPNGDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

load_dotenv()
epochs = int(os.getenv('epochs'))
lr = float(os.getenv('lr'))
patience = int(os.getenv('patience'))
batch_size = int(os.getenv('batch_size'))
image_size = int(os.getenv('image_size'))
num_item = int(os.getenv('num_item'))
max_length_limit = int(os.getenv('max_length_limit'))
noise_level= float(os.getenv('noise_level'))


transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

dataset = SVGPNGDataset(num_items=num_item, max_length_limit=max_length_limit, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size)
device = dataset.device

model = Model(os.getenv('model'), device=device)
model.load_state_dict(torch.load('best_'+model.form+'.pth'))
model.eval()

if model.form == "diffusion":
    with torch.no_grad():
        reconstructed_images = model.model.sample(batch_size=4)
        for reconstructed_image in reconstructed_images:
            sns.heatmap(reconstructed_image.mean(dim=0).detach().cpu().numpy(), cmap='gray')
            plt.show()
elif model.form == "vqgan"
    for batch in data_loader:
        with torch.no_grad():
            original_image = batch['png_tensor'].to(device)
            noisy_image = original_image + torch.randn_like(original_image) * noise_level
            noisy_image = noisy_image.to(device)

        for idx in range(len(original_image)):
            sns.heatmap(original_image[idx])
            sns.heatmap(reconstructed_image[idx])
            plt.show()

