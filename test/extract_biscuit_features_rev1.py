from PIL import Image
import os
import csv
import math

# --- Fungsi-fungsi utility untuk fitur ---
def rgb_to_grayscale(image):
    width, height = image.size
    grayscale = [[0] * width for _ in range(height)]
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            grayscale[y][x] = gray
    return grayscale

def grayscale_to_binary(grayscale, threshold=128):
    height = len(grayscale)
    width = len(grayscale[0])
    binary = [[0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            binary[y][x] = 1 if grayscale[y][x] < threshold else 0
    return binary

def calculate_area(binary):
    return sum(sum(row) for row in binary)

def calculate_perimeter(binary):
    height = len(binary)
    width = len(binary[0])
    perimeter = 0
    for y in range(height):
        for x in range(width):
            if binary[y][x] == 1:
                if (x == 0 or x == width - 1 or y == 0 or y == height - 1 or
                    (x > 0 and binary[y][x-1] == 0) or
                    (x < width - 1 and binary[y][x+1] == 0) or
                    (y > 0 and binary[y-1][x] == 0) or
                    (y < height - 1 and binary[y+1][x] == 0)):
                    perimeter += 1
    return perimeter

def calculate_bounding_box(binary):
    height = len(binary)
    width = len(binary[0])
    min_x, min_y = width, height
    max_x, max_y = -1, -1
    for y in range(height):
        for x in range(width):
            if binary[y][x] == 1:
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
    bb_width = max_x - min_x + 1
    bb_height = max_y - min_y + 1
    aspect_ratio = bb_width / bb_height if bb_height > 0 else 1.0
    bb_area = bb_width * bb_height
    return bb_width, bb_height, bb_area, aspect_ratio

def calculate_extent(area, bb_area):
    return area / bb_area if bb_area > 0 else 0.0

def calculate_circularity(area, perimeter):
    return (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

# --- Fungsi untuk load image dari folder ---
def load_images_from_folder(folder_path):
    image_data = []
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith(('.jpg', '.png')):
                    image_path = os.path.join(subfolder_path, filename)
                    image = Image.open(image_path).convert('RGB')
                    image_data.append((image, filename, subfolder))
    return image_data

def load_single_image(image_path):
    """Membuka 1 gambar dan mengembalikannya dalam mode RGB"""
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        print(f"Gagal membuka {image_path}: {e}")
        return None

def single_image_process():
    image_path = "dataset/bulat/01.jpg"
    output_csv = "fitur_gambar_tunggal.csv"

    image = load_single_image(image_path)
    if not image:
        print("Gambar tidak dapat diproses.")
        return

    # Ekstraksi fitur
    grayscale = rgb_to_grayscale(image)
    binary = grayscale_to_binary(grayscale)

    area = calculate_area(binary)
    perimeter = calculate_perimeter(binary)
    bb_width, bb_height, bb_area, aspect_ratio = calculate_bounding_box(binary)
    extent = calculate_extent(area, bb_area)
    circularity = calculate_circularity(area, perimeter)

    # Tulis hasil ke CSV
    headers = ['filename', 'area', 'perimeter', 'aspect_ratio', 'extent', 'circularity']
    result = [[
        os.path.basename(image_path),
        area,
        perimeter,
        round(aspect_ratio, 2),
        round(extent, 2),
        round(circularity, 2)
    ]]

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(result)

    print(f"Hasil fitur dari {image_path} disimpan ke {output_csv}")
# --- Main ---
def main():
    input_folder = "dataset"
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

    # Simpan ke CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)

    print(f"Hasil ekstraksi fitur disimpan di {output_csv}")

if __name__ == "__main__":
    # main()
    single_image_process()