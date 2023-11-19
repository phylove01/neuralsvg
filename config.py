import os

path = r"C:\Users\이수용\Desktop\fonts-main"
CHUNK = 50000
MAX_SVG_LENGTH = 1024


class Config:
    def __init__(self):
        self.path = path
        self.dataset_folder = os.path.join(path, "dataset")
        self.chunk = CHUNK
        self.max_svg_length = MAX_SVG_LENGTH
        self.svg_root_folder = os.path.join(path, "output_svg")
        self.png_root_folder = os.path.join(path, "output_png")
        self.pickle_save_folder = os.path.join(path, 'dataset')
        self.error_log_path = os.path.join(path, "error_dataset_log.txt")

        self.gen_pickle_save_folder()

    def gen_pickle_save_folder(self):
        if not os.path.exists(self.pickle_save_folder):
            os.makedirs(self.pickle_save_folder)