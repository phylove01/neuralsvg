import torch.nn as nn
import numpy as np
import torch
from dotenv import load_dotenv
import os
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from vqgan_pytorch.vqgan import VQGAN
from config import Config


load_dotenv()
PADDING_VALUE = -1

epochs = int(os.getenv('epochs'))
lr = float(os.getenv('lr'))
patience = int(os.getenv('patience'))
batch_size = int(os.getenv('batch_size'))
image_size = int(os.getenv('image_size'))


class Model(nn.Module):
    def __init__(self, form, device='cpu'):
        super(Model, self).__init__()
        self.form = form
        config = Config()
        if form == 'diffusion':
            backbone_model = Unet(dim=64,
                                  dim_mults=(1, 2, 4, 8),
                                  flash_attn=True).to(device)

            self.model = GaussianDiffusion(backbone_model,
                                           image_size=image_size,
                                           timesteps=1000).to(device)
        elif form == 'vqgan':
            self.model = VQGAN(config.args)
            self.model = self.model.to(device)

    def forward(self, x):
        return self.model(x)

class SVGEncoder(nn.Module):
    def __init__(self, vocab_size=1024, embed_size=256, num_layers=6, heads=8, dropout=0.1,
                 max_len=1024, num_freq_bands=4, max_freq=10., rff_path="best_rff.pt", init_rff=False, device='cpu'):
        super(SVGEncoder, self).__init__()

        self.device = device
        self.command_embedding = nn.Embedding(6 + 1, embed_size)  # Change here

        # Positional Encoding
        self.positions = torch.arange(max_len).unsqueeze(1).to(self.device)
        self.positional_encoding = nn.Embedding(max_len, embed_size)

        self.freq_bands = torch.linspace(0., max_freq, steps=num_freq_bands).unsqueeze(-1).to(self.device)
        self.scale = 2 * np.pi
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if init_rff:
            self.initialize_rff()
        else:
            self.load_best_rff(path=rff_path)

    def initialize_rff(self):
        self.D = self.embed_size  # Fourier encoding의 결과 차원 (설정에 따라 바뀔 수 있습니다.)
        self.W = torch.randn(4, self.D, requires_grad=False, device=self.device)
        self.b = 2 * np.pi * torch.rand(self.D, requires_grad=False, device=self.device)

    def save_best_rff(self, path):
        torch.save({'W': self.W, 'b': self.b}, path)

    def load_best_rff(self, path):
        checkpoint = torch.load(path)
        self.W = checkpoint['W'].to(self.device)
        self.b = checkpoint['b'].to(self.device)

    def fourier_encode(self, x):
        batch_size, vocab_size, _ = x.shape

        # x를 [batch_size*vocab_size, 4] 형태로 재구성합니다.
        x_reshaped = x.view(-1, 4).float()  # .float()를 추가하여 타입 변환

        x_proj = torch.mm(x_reshaped, self.W) + self.b
        x_proj = torch.cos(x_proj)

        # 결과를 원래의 [batch_size, vocab_size, ...] 형태로 되돌립니다.
        x_proj = x_proj.view(batch_size, vocab_size, -1)

        return x_proj

    def forward(self, svg_data):
        commands = svg_data[:, :, 0].long()
        commands[commands == PADDING_VALUE] = 6
        coordinates = svg_data[:, :, 1:]

        command_embeds = self.command_embedding(commands)
        encoded_coords = self.fourier_encode(coordinates)
        x = command_embeds + encoded_coords
        positions = self.positional_encoding(self.positions[:x.size(1)])
        positions = positions.squeeze(1).expand(x.size(0), -1, x.size(2))

        x += positions
        x = self.transformer_encoder(x)
        return x


class SVGDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size=512, num_layers=6, heads=8, dropout=0.1,
                 max_len=1000, latent_dim=512, device='cpu'):
        super(SVGDecoder, self).__init__()

        self.device = device

        # Latent Embedding
        self.latent_embedding = nn.Linear(latent_dim, embed_size)

        # 명령어(ASCII 값)에 대한 Embedding
        self.command_embedding = nn.Embedding(vocab_size, embed_size)

        # Positional Encoding
        self.positions = torch.arange(max_len).unsqueeze(1).to(self.device)
        self.positional_encoding = nn.Embedding(max_len, embed_size)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=heads, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, z, prev_data):
        z_embed = self.latent_embedding(z).unsqueeze(1)

        # 명령어와 좌표값들을 분리
        prev_commands = prev_data[:, :, 0].long()
        prev_coordinates = prev_data[:, :, 1:]

        command_embeds = self.command_embedding(prev_commands)

        # 명령어와 좌표값들의 Encoding을 연결
        x = torch.cat([command_embeds, prev_coordinates], dim=-1)

        # Positional Encoding 추가
        positions = self.positional_encoding(self.positions[:x.size(1)]).expand(x.size(0), x.size(1), -1)
        x += positions

        # Pass through the Transformer Decoder with latent as memory
        x = self.transformer_decoder(x, z_embed)
        return x


class DDPMFeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, timesteps):
        noisy_images = self.model.q_sample(images, timesteps)
        features = self.model.denoise_fn(noisy_images, timesteps)
        return features


class LanguageModelToDDPM(torch.nn.Module):
    def __init__(self, language_model, ddpm_model):
        super().__init__()
        self.language_model = language_model
        self.ddpm_model = ddpm_model
        self.mapping_layer = torch.nn.Linear(language_model_output_dim, image_size * image_size * channels)

    def forward(self, text_input):
        # 언어 모델을 통해 feature 추출
        language_features = self.language_model(text_input)

        # 언어 모델의 출력을 이미지 형식으로 매핑
        mapped_features = self.mapping_layer(language_features)
        mapped_features = mapped_features.view(-1, channels, image_size, image_size)

        # DDPM을 사용하여 이미지 생성
        generated_images = self.ddpm_model.p_sample_loop(mapped_features, model.num_timesteps)
        return generated_images


class SVGPNGModel(nn.Module):
    def __init__(self, vocab_size, init_rff=False, device='cpu'):
        super(SVGPNGModel, self).__init__()
        self.svgencoder = SVGEncoder(vocab_size=vocab_size, init_rff=init_rff, device=device).to(device)
        self.svgdecoder = SVGDecoder(vocab_size=vocab_size, device=device).to(device)

        self.pngencoder = PNGEncoder(device=device).to(device)
        self.pngdecoder = PNGDecoder(device=device).to(device)

    def forward(self, svg, png):
        zs = self.svgencoder(svg)
        zp = self.pngencoder(png)

        reconstructed_svg = self.svgdecoder(zs)
        reconstructed_png = self.pngdecoder(zp)

        return reconstructed_svg, reconstructed_png, zs, zp
