from PIL import Image
import os

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