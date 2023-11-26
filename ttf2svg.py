import os
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.boundsPen import BoundsPen
import tqdm
import logging

import re

BOUNDING_SIZE = 1000
MARGIN = 0.1  # 10%

def flip_y_axis(commands, height):
    """
    Flip the y-axis of the SVG commands.
    The height parameter should be the height of the SVG viewport to adjust the flipped coordinates.
    """
    command_pattern = re.compile(r"([MLQVHCZ])([\d\.,\s\-]*)")
    new_commands = []

    for command_match in command_pattern.finditer(commands):
        command, coordinates = command_match.groups()
        coord_values = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+", coordinates)]

        new_coords = []
        if command.upper() == "V":  # No x coordinate, just flip y
            new_coords = [height - v for v in coord_values]
        elif command.upper() == "H":  # No y coordinate, leave x as is
            new_coords = coord_values
        else:  # Flip y coordinate for each point
            for i in range(0, len(coord_values), 2):
                x, y = coord_values[i:i + 2]
                new_coords.extend([x, height - y])

        new_commands.append(f"{command}{' '.join(map(str, new_coords))}")

    return " ".join(new_commands)


def find_extrema(p0, p1, p2):
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2

    tx = (x0 - x1) / (x0 - 2 * x1 + x2) if x0 != 2 * x1 - x2 else 0.0
    ty = (y0 - y1) / (y0 - 2 * y1 + y2) if y0 != 2 * y1 - y2 else 0.0

    x_values = [x0, x2] + [x0 * (1 - tx) ** 2 + 2 * x1 * tx * (1 - tx) + x2 * tx ** 2 for tx in [tx] if 0 <= tx <= 1]
    y_values = [y0, y2] + [y0 * (1 - ty) ** 2 + 2 * y1 * ty * (1 - ty) + y2 * ty ** 2 for ty in [ty] if 0 <= ty <= 1]

    return min(x_values), min(y_values), max(x_values), max(y_values)


def find_rectangle(p0, p1):
    x_values = [p0[0], p1[0]]
    y_values = [p0[1], p1[1]]
    return min(x_values), min(y_values), max(x_values), max(y_values)


def get_glyph_bounds_from_boundspen(glyph, glyph_set):
    bounds_pen = BoundsPen(glyph_set)
    glyph.draw(bounds_pen)
    bounds = bounds_pen.bounds  # (xMin, yMin, xMax, yMax)
    return bounds


def get_glyph_bounding_box_from_command(commands):
    command_pattern = re.compile(r"([MLQVHCZ])([\d\.,\s\-]*)")
    points = []
    boxes = []

    for command_match in command_pattern.finditer(commands):
        command, coordinates = command_match.groups()
        coord_values = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+", coordinates)]

        if command == "M" and len(coord_values) >= 2:
            points.append(tuple(coord_values[-2:]))
        elif command == "L":
            points.append(tuple(coord_values[0: 2]))
            boxes.append(find_rectangle(points[-1], points[-2]))
        elif command == "H":
            points.extend([(x, points[-1][1]) for x in coord_values])
            boxes.append(find_rectangle(points[-1], points[-2]))
        elif command == "V":
            points.extend([(points[-1][0], y) for y in coord_values])
            boxes.append(find_rectangle(points[-1], points[-2]))
        elif command == "Q":
            for i in range(0, len(coord_values), 4):
                p0 = points[-1]
                p1 = tuple(coord_values[i:i + 2])
                p2 = tuple(coord_values[i + 2:i + 4])
                points.append(p2)
                boxes.append(find_extrema(p0, p1, p2))
        elif command == "Z":
            pass

    # If there are no boxes calculated from curves, use points for the bounding box
    if not boxes and not points:
        return None

    min_x = min(box[0] for box in boxes)
    max_x = max(box[2] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_y = max(box[3] for box in boxes)
    return min_x, min_y, max_x, max_y


def normalize_commands(commands, bounds, max_scale=1):
    # Calculating the scaling factor considering margin and keeping aspect ratio
    x_min, y_min, x_max, y_max = bounds

    original_width = x_max - x_min
    original_height = y_max - y_min
    scale = min(
        (BOUNDING_SIZE * (1 - 2 * MARGIN)) / original_width,
        (BOUNDING_SIZE * (1 - 2 * MARGIN)) / original_height,
        max_scale
    )

    # Calculating the translation values
    x_translate = x_min + original_width / 2
    y_translate = y_min + original_height / 2

    # Calculating the new center to place the glyph nicely in the middle of the viewBox
    x_center = BOUNDING_SIZE / 2
    y_center = BOUNDING_SIZE / 2

    # Regular expression to extract command and coordinates from the path data
    command_pattern = re.compile(r"([MLQVHCZ])([\d\.,\s\-]*)")

    new_commands = []
    for command_match in command_pattern.finditer(commands):
        command, coordinates = command_match.groups()
        coord_values = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+", coordinates)]

        # Adjusting the coordinates according to new bounding box
        new_coords = []
        if command.upper() == "V":
            new_coords = [-(v - y_translate) * scale + y_center for v in coord_values]
        elif command.upper() == "H":
            new_coords = [(v - x_translate) * scale + x_center for v in coord_values]
        else:
            for i in range(0, len(coord_values), 2):
                x, y = coord_values[i:i + 2]
                new_coords.extend([(x - x_translate) * scale + x_center, -(y - y_translate) * scale + y_center])

        new_commands.append(f"{command}{' '.join(map(str, new_coords))}")

    return " ".join(new_commands)


def glyph_to_svg(font, glyph_name, output_path, option=None, normalization=True):
    try:
        glyph_set = font.getGlyphSet()
        glyph = glyph_set[glyph_name]
        pen = SVGPathPen(glyph_set)
        glyph.draw(pen)
        commands = pen.getCommands()

        # Get the bounding box
        if option == 'filled_bounds':
            bounds = get_glyph_bounding_box_from_command(commands)
        else:
            bounds = get_glyph_bounds_from_boundspen(glyph, glyph_set)


        # Normalize commands to fit into the BOUNDING_SIZE with margin
        if normalization:
            commands = normalize_commands(commands, bounds, max_scale = 0.5)

        svg_content = f"""<svg width='{BOUNDING_SIZE}' height='{BOUNDING_SIZE}' version='1.1' xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {BOUNDING_SIZE} {BOUNDING_SIZE}'>
            <rect width='100%' height='100%' fill='white'/> 
            <path d='{commands}' fill='black'/> 
        </svg>"""

        with open(output_path, 'w') as svg_file:
            svg_file.write(svg_content)

    except Exception as e:
        with open(log_file, 'a') as log:
            log.write(f"Failed to convert {glyph_name} at {font}: {e}\n")
        return False


def create_svg_from_font(ttf_path, output_folder, log_file):
    trial, successful = 0, 0
    font = TTFont(ttf_path)
    unicode_map = font.getBestCmap()

    if unicode_map is not None:
        for unicode_val, glyph_name in unicode_map.items():
            svg_filename = os.path.join(output_folder, f"{unicode_val}.svg")

            try:
                if not os.path.exists(svg_filename):
                    glyph_to_svg(font, glyph_name, svg_filename)
                trial += 1
                successful +=1
            except Exception as e:
                trial += 1
                logging.error(f"An error occurred: {e}")
    return trial, successful

def main(root_folders, output_root_folder, log_file):
    candidates = []
    for root_folder in root_folders:
        for folder_name, subfolders, filenames in os.walk(root_folder):
            for filename in filenames:
                if filename.endswith('.ttf'):
                    candidates.append((folder_name, filename))

    trial_conversions, successful_conversions = 0, 0
    for folder_name, filename in tqdm.tqdm(candidates, desc='ttf2svg'):
        ttf_file = os.path.join(folder_name, filename)
        font_name = filename.replace('.ttf', '')
        output_folder = os.path.join(output_root_folder, font_name)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        trial, successful = create_svg_from_font(ttf_file, output_folder, log_file)
        trial_conversions += trial
        successful_conversions += successful

    with open(log_file, 'a') as log:
        log.write(f"\nTotal files: {trial_conversions}\n")
        log.write(f"Successful conversions: {successful_conversions}\n")
        log.write(f"Failed conversions: {trial_conversions - successful_conversions}\n")
        success_rate = (successful_conversions / trial_conversions) * 100 if trial_conversions else 0
        log.write(f"Success rate: {success_rate:.2f}%\n")


if __name__ == "__main__":
    root_folders = [
        "C:\\Windows\\Fonts\\",
        "C:\\Users\\이수용\\Desktop\\fonts-main\\ttf\\",
        "C:\\Users\\이수용\\Downloads\\"
    ]
    output_root_folder = 'C:\\Users\\이수용\\Desktop\\fonts-main\\svg\\'
    log_file = 'C:\\Users\\이수용\\Desktop\\fonts-main\\ttf2svg.err'
    if os.path.exists(log_file):
        os.remove(log_file)
    logging.basicConfig(filename=log_file, level=logging.ERROR)
    main(root_folders, output_root_folder, log_file)
