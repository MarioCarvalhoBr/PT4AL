import numpy as np
import os
from tqdm import tqdm

if __name__ == '__main__':
    batches_dir = "/home/juan/Documents/tcc/code/PT4AL/loss"

    # for file in os.listdir(batches_dir):
        # if file.startswith("batch"):
            # replace /home/aroeira/Desktop/CARVALHO/doutorado/datasets/mamonas_32x32_0.1_to_1_split with /home/juan/Documents/tcc/datasets/mamonas_32x32_0.1_to_1_split
    file = "/home/juan/Documents/tcc/code/PT4AL/rotation_loss.txt"
    new_lines = [] 
    with open(os.path.join(batches_dir, file), "r") as f:
        lines = f.readlines()
        
    for line in tqdm(lines):
        new_lines.append(line.replace("/home/aroeira/Desktop/CARVALHO/doutorado/datasets/mamonas_32x32_0.1_to_1_split", "/home/juan/Documents/tcc/datasets/mamonas_32x32_0.1_to_1_split"))

    with open(os.path.join(batches_dir, file), "w") as f:
        f.writelines(new_lines)