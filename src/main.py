import os
from image_loader import load_images_from_folder, load_single_image
from visualization import show_image_processing_steps

from feature_extractor import (
    rgb_to_grayscale,
    grayscale_to_binary,
    calculate_area,
    calculate_perimeter,
    calculate_bounding_box,
    calculate_extent,
    calculate_circularity
)
from utils import write_to_csv

def process_single_image(file_directory):
    image_path = file_directory
    output_csv = "fitur_gambar_tunggal.csv"
    image = load_single_image(image_path)

    if not image:
        print("Gambar tidak dapat diproses.")
        return

    grayscale = rgb_to_grayscale(image)
    binary = grayscale_to_binary(grayscale)
    area = calculate_area(binary)
    perimeter = calculate_perimeter(binary)
    bb_width, bb_height, bb_area, aspect_ratio = calculate_bounding_box(binary)
    extent = calculate_extent(area, bb_area)
    circularity = calculate_circularity(area, perimeter)

    headers = ['filename', 'area', 'perimeter', 'aspect_ratio', 'extent', 'circularity']
    result = [[
        os.path.basename(image_path),
        area,
        perimeter,
        round(aspect_ratio, 2),
        round(extent, 2),
        round(circularity, 2)
    ]]
    filename = os.path.basename(image_path)
    write_to_csv(headers, result, output_csv)
    show_image_processing_steps(image, title=filename)
    print(f"Hasil fitur dari {image_path} disimpan ke {output_csv}")

def process_batch_images(directory):
    input_folder = directory
    output_csv = "fitur_biskuit.csv"
    headers = ['filename', 'area', 'perimeter', 'aspect_ratio', 'extent', 'circularity', 'label']
    results = []

    images = load_images_from_folder(input_folder)

    for image, filename, label in images:
        grayscale = rgb_to_grayscale(image)
        binary = grayscale_to_binary(grayscale)
        area = calculate_area(binary)
        perimeter = calculate_perimeter(binary)
        bb_width, bb_height, bb_area, aspect_ratio = calculate_bounding_box(binary)
        extent = calculate_extent(area, bb_area)
        circularity = calculate_circularity(area, perimeter)

        results.append([
            filename,
            area,
            perimeter,
            round(aspect_ratio, 2),
            round(extent, 2),
            round(circularity, 2),
            label
        ])
        image.close()

    write_to_csv(headers, results, output_csv)
    print(f"Hasil ekstraksi fitur disimpan di {output_csv}")

if __name__ == "__main__":
    input_folder = "dataset"
    input_file = "dataset/bulat/01.jpg"
    
    # process_batch_images(input_folder)
    process_single_image(input_file)
