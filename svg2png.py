import os
from wand.image import Image
from wand.color import Color
import pathlib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Input and output folders
input_folder_path = r'C:\Users\이수용\Desktop\fonts-main\output_svg'
output_folder_path = r'C:\Users\이수용\Desktop\fonts-main\output_png'

# Ensure the output folder exists
pathlib.Path(output_folder_path).mkdir(parents=True, exist_ok=True)


# SVG to PNG conversion function
def convert_svg_to_png(svg_path, png_path):
    # Check if PNG already exists, if so, skip conversion
    if os.path.exists(png_path):
        return
    try:
        with Image(filename=svg_path, background=Color('transparent')) as svg_image:
            svg_image.format = "png"
            svg_image.resize(1000, 1000)  # Resize the image to 1000x1000
            svg_image.save(filename=png_path)
    except Exception as e:
        with open('error_svg2png.log', 'a') as log_file:
            log_file.write(f"Error converting {svg_path} to PNG: {str(e)}\n")


# List to store conversion tasks
conversion_tasks = []

for svg_path in pathlib.Path(input_folder_path).rglob('*.svg'):
    relative_path = os.path.relpath(svg_path, input_folder_path)
    png_path = os.path.join(output_folder_path, os.path.splitext(relative_path)[0] + '.png')
    pathlib.Path(os.path.dirname(png_path)).mkdir(parents=True, exist_ok=True)
    conversion_tasks.append((str(svg_path), png_path))

# Filtering the tasks to only include SVGs that haven't been converted yet
conversion_tasks = [(svg, png) for svg, png in conversion_tasks if not os.path.exists(png)]

# Count total SVG files for progress bar
total_svg_count = len(conversion_tasks)

# Using ThreadPoolExecutor to process images in parallel
with ThreadPoolExecutor() as executor, tqdm(total=total_svg_count, unit="file", desc="Converting") as pbar:
    def task_wrapper(args):
        convert_svg_to_png(*args)
        pbar.update(1)


    list(executor.map(task_wrapper, conversion_tasks))
