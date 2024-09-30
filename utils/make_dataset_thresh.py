from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
    

def filter_mamonas(source_dir, patch_size, upper_threshold, lower_threshold):
    target_dir = f"/home/juan/Documents/tcc/datasets/mamonas_{patch_size}x{patch_size}_{lower_threshold}_to_{upper_threshold}"

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        os.makedirs(target_dir + '/labels')
        os.makedirs(target_dir + '/mamonas')
        os.makedirs(target_dir + '/background')

    img_names = os.listdir(source_dir)
    with tqdm(img_names) as pbar:
        for img_name in pbar:
            image_path = os.path.join(source_dir, img_name)
            image = Image.open(image_path)
            image_np = np.array(image)
            image_np = np.sum(image_np, axis=2)
            mamona_pixels = np.sum(image_np == 255)
            total_pixels = image_np.shape[0] * image_np.shape[1]
            pct = mamona_pixels / total_pixels

            if (pct > lower_threshold) and (pct <= upper_threshold):
                image.save(os.path.join(target_dir, "labels", img_name)) 
                os.system(f"cp {image_path.replace('labels', 'rgb').replace('.png', '.jpg')} {target_dir}/mamonas/{img_name.replace('.png', '.jpg')}")

            elif np.sum(image_np) == 0:
                os.system(f"cp {image_path.replace('labels', 'rgb').replace('.png', '.jpg')} {target_dir}/background/{img_name.replace('.png', '.jpg')}")
                

if __name__ == "__main__":
    # source_dir = "/home/juan/Documents/tcc/datasets/mamonas_64x64/labels"
    # patch_size = 64
    # lower_threshold = 0.01
    # upper_threshold = 1

    # filter_mamonas(source_dir, patch_size, upper_threshold, lower_threshold)

    source_dir = "/home/juan/Documents/tcc/datasets/mamonas_32x32/labels"
    patch_size = 32
    lower_threshold = 0.01
    upper_threshold = 1

    filter_mamonas(source_dir, patch_size, upper_threshold, lower_threshold)

    source_dir = "/home/juan/Documents/tcc/datasets/mamonas_64x64/labels"
    patch_size = 64
    lower_threshold = 0.1
    upper_threshold = 1

    filter_mamonas(source_dir, patch_size, upper_threshold, lower_threshold)

    source_dir = "/home/juan/Documents/tcc/datasets/mamonas_32x32/labels"
    patch_size = 32
    lower_threshold = 0.1
    upper_threshold = 1

    filter_mamonas(source_dir, patch_size, upper_threshold, lower_threshold)




