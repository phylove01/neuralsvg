import os
from dotenv import load_dotenv
import torch
import argparse

CHUNK = 50000
MAX_SVG_LENGTH = 1024


class Config:
    def __init__(self):
        load_dotenv(encoding='utf-8')
        path = os.getenv('data_path')
        if '/' in path:
            self.path = os.path.join(*path.split('/'))
        elif '\\' in path:
            self.path = '\\'.join(path.split('\\'))

        self.dataset_folder = os.path.join(self.path, "dataset")
        self.chunk = CHUNK
        self.max_svg_length = MAX_SVG_LENGTH
        self.svg_root_folder = os.path.join(self.path, "svg")
        self.png_root_folder = os.path.join(self.path, "png")
        self.pickle_save_folder = self.dataset_folder
        self.error_log_path = os.path.join(self.path, "dataset.err")
        self.save_path = os.path.join(self.path, 'save')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gen_pickle_save_folder()


        parser = argparse.ArgumentParser(description="VQGAN")
        parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
        parser.add_argument('--image-size', type=int, default=int(os.getenv('image_size')), help='Image height and width (default: '+os.getenv('image_size')+')')
        parser.add_argument('--num-codebook-vectors', type=int, default=1024,
                            help='Number of codebook vectors (default: 256)')
        parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
        parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images (default: 1)')
        parser.add_argument('--dataset-path', type=str, default= self.dataset_folder, help='Path to data (default:'+self.dataset_folder+' )')
        parser.add_argument('--device', type=str, default=self.device, help='Which device the training is on')
        parser.add_argument('--batch-size', type=int, default=int(os.getenv('batch_size')), help='Input batch size for training (default: 6)')
        parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
        parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
        parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
        parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
        parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
        parser.add_argument('--disc-factor', type=float, default=1., help='')
        parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
        parser.add_argument('--perceptual-loss-factor', type=float, default=1.,
                            help='Weighting factor for perceptual loss.')

        self.args = parser.parse_args()

    def gen_pickle_save_folder(self):
        if not os.path.exists(self.pickle_save_folder):
            os.makedirs(self.pickle_save_folder)

