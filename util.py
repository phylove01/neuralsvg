import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import io
from config import Config


class SVGPNGDataset(Dataset, Config):
    def __init__(self, num_items=None, max_length_limit=4096, transform=None):
        super().__init__()
        self.folder_path = self.pickle_save_folder
        self.file_list = sorted(
            [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith('.pkl')])

        if num_items is not None:
            # 필요한 파일의 수 계산
            num_files_needed = -(-num_items // self.chunk)  # ceiling division
            self.file_list = self.file_list[:num_files_needed]
            self.num_items = num_items
        else:
            self.num_items = len(self.file_list) * self.chunk

        self.loaded_data = {}
        self.max_length_limit = max_length_limit
        self.original_num_items = self.num_items
        self.skipped_items = 0
        self.cmd_history = []
        self.char_dict = {'M':0,'L':1,'Q':2,'V':3,'H':4,'Z':5}
        self.transform = transform

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        file_idx = idx // self.chunk
        inner_idx = idx % self.chunk

        if file_idx not in self.loaded_data:
            self.loaded_data[file_idx] = pd.read_pickle(self.file_list[file_idx])

        row = self.loaded_data[file_idx].iloc[inner_idx]

        font = row['font']
        character = chr(int(row['ord_num']))
        svg_data = row['svg_content']

        svg_data = self.trim_svg_data(svg_data)
        svg_tensor = self.svg_to_tensor(svg_data)

        if svg_tensor.shape[0] > self.max_length_limit:
            print(f"Warning: SVG data at index {idx} exceeds max length of {self.max_length_limit}. Skipping...")
            return self.__getitem__(idx + 1)

        # SVG 데이터 패딩
        padding_length = self.max_length_limit - svg_tensor.shape[0]
        if padding_length > 0:
            svg_tensor = torch.cat([svg_tensor, torch.full((padding_length, 5), -1, dtype=torch.long)], dim=0)

        png_tensor = self.png_to_tensor(row['png_content'])
        return {
            'font': font,
            'character': character,
            'svg_tensor': svg_tensor,
            'png_tensor': png_tensor
        }

    def svg_to_tensor(self, svg_data, padding_value=-1, max_length=5):
        tensor_data = []
        for item in svg_data:
            if item[0] not in self.cmd_history:
                self.cmd_history.append(item[0])
            command = [self.char_dict[item[0]]]
            if item[0] == "M":
                coords = item[-2:]
                padding = [-1, -1]
            else:
                coords = item[1:]
                padding = [-1 for _ in range(max_length - len(command) - len(coords))]
            tensor_data.append(command + coords + padding)
        return torch.tensor(tensor_data).long()

    def png_to_tensor(self, png_data):
        png_image = Image.open(io.BytesIO(png_data)).convert("L")  # Convert to grayscale
        png_tensor = self.transform(png_image)
        png_tensor = png_tensor.float() / 255.0
        return png_tensor.repeat(3, 1, 1)

    def get_max_svg_length(self):
        max_length = 0
        count_exceeding = 0

        for file in self.file_list:
            data_chunk = pd.read_pickle(file)
            for _, row in data_chunk.iterrows():
                svg_data = row['svg_content']
                length = sum([len(item) for item in svg_data])
                if length <= self.max_length_limit:
                    max_length = max(max_length, length)
                else:
                    count_exceeding += 1

        if count_exceeding > 0:
            print(f"Warning: {count_exceeding} items exceed the max_length_limit of {self.max_length_limit}.")

        return max_length

    def trim_svg_data(self, svg_data):
        total_length = sum([len(item) for item in svg_data])

        if total_length <= self.max_svg_length:
            return svg_data

        trimmed_data = []
        accumulated_length = 0
        for item in svg_data:
            if accumulated_length + len(item) > self.max_svg_length:
                break
            accumulated_length += len(item)
            trimmed_data.append(item)
        return trimmed_data


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
        return False

    def save_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(), path)
        model.svgencoder.save_best_rff(path="best_rff.pt")
