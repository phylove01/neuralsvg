import os
import re
import pandas as pd
import pickle
from tqdm import tqdm
from config import Config

config = Config()


def extract_svg_path(svg_file):
    with open(svg_file, 'r', encoding='utf-8') as f:
        content = f.read()
        matches = re.findall(r'd=["\'](.*?)["\']', content)
        if matches:
            return parse_path(''.join(matches))
        else:
            return []


def generate_save_name(idx):
    total_chunks = total_files // config.chunk + 1
    total_digits = len(str(total_chunks))
    return f"save{idx:0{total_digits}d}.pkl"


def parse_path(path_str):
    commands = re.findall("[A-Za-z][^A-Za-z]*", path_str)

    parsed_commands = []
    for command in commands:
        letter = command[0]
        numbers = list(map(float, re.findall("[-+]?\d*\.?\d+", command[1:])))
        parsed_commands.append([letter] + numbers)

    return parsed_commands


total_files = 0
for root, _, files in os.walk(config.svg_root_folder):
    total_files += sum(1 for file in files if file.endswith('.svg'))

data_list = []
index_num = 1

progress_bar = tqdm(total=total_files, desc="Processing files")
font_folders = os.listdir(config.svg_root_folder)

for font_folder in font_folders:
    font_svg_folder = os.path.join(config.svg_root_folder, font_folder)
    if os.path.isdir(font_svg_folder):
        files = os.listdir(font_svg_folder)
        for file in files:
            if file.endswith('.svg'):
                svg_file_path = os.path.join(config.svg_root_folder, font_folder, file)
                png_file_path = os.path.join(config.png_root_folder, font_folder, file.replace('.svg', '.png'))
                try:
                    svg_content = extract_svg_path(svg_file_path)
                    with open(png_file_path, 'rb') as f:
                        png_byte_data = f.read()
                    data_list.append({
                        'index_num': index_num,
                        'font': font_folder,
                        'ord_num': os.path.splitext(file)[0],
                        'svg_content': svg_content,
                        'png_content': png_byte_data
                    })

                except Exception as e:  # 에러 발생 시
                    with open(config.error_log_path, 'a', encoding="utf-8") as error_log:  # 에러 로그에 기록
                        error_log.write(
                            f"Error processing {svg_file_path} and {png_file_path}. Error: {str(e)}\n")
                    continue
                finally:
                    index_num += 1
                    progress_bar.update(1)

                if len(data_list) == config.chunk:
                    df = pd.DataFrame(data_list)
                    file_name = os.path.join(config.pickle_save_folder, generate_save_name(index_num // config.chunk))
                    with open(file_name, 'wb') as f:
                        pickle.dump(df, f)
                    data_list = []

if data_list:
    df = pd.DataFrame(data_list)
    file_name = os.path.join(config.pickle_save_folder, generate_save_name((index_num - 1) // config.chunk + 1))
    with open(file_name, 'wb') as f:
        pickle.dump(df, f)

progress_bar.close()
