import os
import cv2
import tqdm

def subdivide_image(source_dir, patch_size):
    img_type =  source_dir.split("/")[-1]
    target_dir = f"/home/juan/Documents/tcc/datasets/mamonas_{patch_size}x{patch_size}/{img_type}"

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    img_names = os.listdir(source_dir)

    print(f"Processing {len(img_names)} images from {source_dir}")
    with tqdm.tqdm(img_names) as pbar:
        for img_name in pbar:
            image_path = os.path.join(source_dir, img_name)
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            for i in range(0, height, patch_size):
                for j in range(0, width, patch_size):
                    pbar.set_description(f"Image: {img_name}, Patch: {i}, {j}")
                    patch = image[i:i+patch_size, j:j+patch_size]
                    if img_type == "labels":
                        patch_name = img_name.replace(".png", f"_patch_{i}_{j}.png")
                    else:
                        patch_name = img_name.replace(".jpg", f"_patch_{i}_{j}.jpg")
                    cv2.imwrite(os.path.join(target_dir, patch_name), patch)

if __name__ == "__main__":
    patch_size = 64
    source_dir = "/home/juan/Documents/tcc/30k_mamona/30k_mamona/labels"
    subdivide_image(source_dir, patch_size)

    patch_size = 64
    source_dir = "/home/juan/Documents/tcc/30k_mamona/30k_mamona/rgb"
    subdivide_image(source_dir, patch_size)

    patch_size = 32
    source_dir = "/home/juan/Documents/tcc/30k_mamona/30k_mamona/labels"
    subdivide_image(source_dir, patch_size)

    patch_size = 32
    source_dir = "/home/juan/Documents/tcc/30k_mamona/30k_mamona/rgb"
    subdivide_image(source_dir, patch_size)