from PIL import Image
import os
import csv
import math

# Fungsi untuk konversi RGB ke Grayscale
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

# Fungsi untuk konversi Grayscale ke Biner
def grayscale_to_binary(grayscale, threshold=128):
    height = len(grayscale)
    width = len(grayscale[0])
    binary = [[0] * width for _ in range(height)]
    
    for y in range(height):
        for x in range(width):
            binary[y][x] = 1 if grayscale[y][x] < threshold else 0
    return binary

# Fungsi untuk menghitung area
def calculate_area(binary):
    area = 0
    for row in binary:
        area += sum(row)
    return area

# Fungsi untuk menghitung perimeter
def calculate_perimeter(binary):
    height = len(binary)
    width = len(binary[0])
    perimeter = 0
    
    for y in range(height):
        for x in range(width):
            if binary[y][x] == 1:
                is_boundary = False
                if (x == 0 or x == width - 1 or y == 0 or y == height - 1 or
                    (x > 0 and binary[y][x-1] == 0) or
                    (x < width - 1 and binary[y][x+1] == 0) or
                    (y > 0 and binary[y-1][x] == 0) or
                    (y < height - 1 and binary[y+1][x] == 0)):
                    is_boundary = True
                if is_boundary:
                    perimeter += 1
    return perimeter

# Fungsi untuk menghitung bounding box dan aspect ratio
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

# Fungsi untuk menghitung extent
def calculate_extent(area, bb_area):
    return area / bb_area if bb_area > 0 else 0.0

# Fungsi untuk menghitung circularity
def calculate_circularity(area, perimeter):
    if perimeter == 0:
        return 0.0
    return (4 * math.pi * area) / (perimeter ** 2)

# Pipeline utama untuk memproses folder gambar
def process_images(input_folder, output_csv):
    headers = ['filename', 'area', 'perimeter', 'aspect_ratio', 'extent', 'circularity', 'label']
    results = []
    
    # Iterasi melalui subfolder di dalam dataset
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith(('.jpg', '.png')):
                    image_path = os.path.join(subfolder_path, filename)
                    image = Image.open(image_path).convert('RGB')
                    
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
                        subfolder  # Label diambil dari nama subfolder
                    ])
                    
                    image.close()
    
    # Tulis ke CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)

if __name__ == "__main__":
    input_folder = "datauji"
    output_csv = "uji_fitur_biskuit.csv"
    process_images(input_folder, output_csv)
    print(f"Hasil ekstraksi fitur disimpan di {output_csv}")