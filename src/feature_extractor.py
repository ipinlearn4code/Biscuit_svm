import math

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

def grayscale_to_binary(grayscale, threshold=180):
    height = len(grayscale)
    width = len(grayscale[0])
    binary = [[0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            binary[y][x] = 1 if grayscale[y][x] >= threshold else 0
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
