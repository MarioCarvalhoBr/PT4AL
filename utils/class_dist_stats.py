import numpy as np

dist_path = "/home/juan/Documents/tcc/code/PT4AL/random_run/class_dist.txt"

dists = []
with open(dist_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(": ")
        dist = line[-1].strip("[]").split()
        dists.append(np.array(dist).astype(float))

dists = np.array(dists)
print(dists.shape)
print("Class distribution")
for i in range(dists.shape[1]):
    mean = np.mean(dists[:, i])
    std = np.std(dists[:, i])
    print(f"Class {i} - Mean: {mean:.2f}, Std: {std:.2f}")
