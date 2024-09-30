import os
import shutil
import random
from tqdm import tqdm

def split_dataset(whole_dataset_dir, split_dataset_dir, split_ratio):
    if not os.path.exists(split_dataset_dir):
        os.makedirs(os.path.join(split_dataset_dir, "train"))
        os.makedirs(os.path.join(split_dataset_dir, "test"))
        

    for class_dir in os.listdir(whole_dataset_dir):
        if class_dir == "labels":
            continue

        if not os.path.exists(os.path.join(split_dataset_dir, "train", class_dir)):
            os.makedirs(os.path.join(split_dataset_dir, "train", class_dir))

        if not os.path.exists(os.path.join(split_dataset_dir, "test", class_dir)):
            os.makedirs(os.path.join(split_dataset_dir, "test", class_dir))
        
        print(f"Splitting class {class_dir}")

        class_path = os.path.join(whole_dataset_dir, class_dir)
        images = os.listdir(class_path)
        random.shuffle(images)
        split_index = int(len(images) * split_ratio)

        train_images = images[:split_index]
        test_images = images[split_index:]

        for image in tqdm(train_images):
            image_path = os.path.join(class_path, image)
            new_image_path = os.path.join(split_dataset_dir, "train", class_dir, image)
            shutil.copyfile(image_path, new_image_path)

        for image in tqdm(test_images):
            image_path = os.path.join(class_path, image)
            new_image_path = os.path.join(split_dataset_dir, "test", class_dir, image)
            shutil.copyfile(image_path, new_image_path)

        


if __name__ == "__main__":
    random.seed(42)
    whole_dataset_dir = "/home/juan/Documents/tcc/datasets/mamonas_32x32_0.1_to_1"
    split_dataset_dir = "/home/juan/Documents/tcc/datasets/mamonas_32x32_0.1_to_1_split"

    split_dataset(whole_dataset_dir, split_dataset_dir, 0.9)