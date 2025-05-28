import matplotlib.pyplot as plt
import matplotlib.patches as patches
from feature_extractor import rgb_to_grayscale, grayscale_to_binary, calculate_bounding_box

def show_image_processing_steps(image, title="Visualisasi Proses"):
    grayscale = rgb_to_grayscale(image)
    binary = grayscale_to_binary(grayscale)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(title, fontsize=14)

    # 1. Original
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # 2. Grayscale
    axes[1].imshow(grayscale, cmap='gray')
    axes[1].set_title("Grayscale")
    axes[1].axis("off")

    # 3. Binary
    axes[2].imshow(binary, cmap='gray')
    axes[2].set_title("Binary")
    axes[2].axis("off")

    # 4. Binary + Bounding Box
    axes[3].imshow(binary, cmap='gray')
    axes[3].set_title("Bounding Box")
    axes[3].axis("off")

    bb_width, bb_height, _, _ = calculate_bounding_box(binary)

    min_x, min_y = find_min_position(binary)
    if bb_width > 0 and bb_height > 0:
        rect = patches.Rectangle((min_x, min_y), bb_width, bb_height, linewidth=2, edgecolor='r', facecolor='none')
        axes[3].add_patch(rect)

    plt.tight_layout()
    plt.show()


def find_min_position(binary):
    """Mencari posisi kiri atas dari bounding box."""
    height = len(binary)
    width = len(binary[0])
    min_x, min_y = width, height
    for y in range(height):
        for x in range(width):
            if binary[y][x] == 1:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
    return min_x, min_y
