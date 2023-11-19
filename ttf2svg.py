import sys
import time
import os
import cairosvg
import pathlib
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from tqdm import tqdm

# Input and output folders
input_folder_path = r'C:\Users\이수용\Desktop\fonts-main\output_svg'
output_folder_path = r'C:\Users\이수용\Desktop\fonts-main\output_png'

# Ensure the output folder exists
pathlib.Path(output_folder_path).mkdir(parents=True, exist_ok=True)

# SVG to PNG conversion function
def convert_svg_to_png(svg_path, png_path, pbar):
    try:
        cairosvg.svg2png(url=svg_path, write_to=png_path)
        pbar.update(1)  # Update the progress bar
    except Exception as e:
        # Log error messages to a file instead of printing them
        with open('error_svg2png.log', 'a') as log_file:
            log_file.write(f"Error converting {svg_path} to PNG: {str(e)}\n")


# Watchdog event handler
class SVGHandler(PatternMatchingEventHandler):
    patterns = ["*.svg"]

    def process(self, event):
        _, file_extension = os.path.splitext(event.src_path)
        if file_extension.lower() == ".svg":
            relative_path = os.path.relpath(event.src_path, input_folder_path)
            png_path = os.path.join(output_folder_path, os.path.splitext(relative_path)[0] + '.png')
            pathlib.Path(os.path.dirname(png_path)).mkdir(parents=True, exist_ok=True)
            convert_svg_to_png(event.src_path, png_path, pbar)

    def on_modified(self, event):
        self.process(event)

    def on_created(self, event):
        self.process(event)


# Count total SVG files for progress bar
total_svg_count = sum(1 for _ in pathlib.Path(input_folder_path).rglob('*.svg'))

# Watchdog setup
observer = Observer()
observer.schedule(SVGHandler(), path=input_folder_path, recursive=True)

# Start the observer
observer.start()
try:
    print("Monitoring for SVG changes. Press Ctrl+C to stop.")
    with tqdm(total=total_svg_count, unit="file", desc="Converting") as pbar:
        while True:
            time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
