import matplotlib.pyplot as plt
import numpy as np

metrics_path = "/home/juan/Documents/tcc/code/PT4AL/pt4al_run0/metrics.txt"
precisions = []
recalls = []
f1_scores = []

with open(metrics_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        
        if len(line) == 0 or line[1] == "precision":
            continue

        if line[0] == "0":
            precisions.append(float(line[1]))
            recalls.append(float(line[2]))
            f1_scores.append(float(line[3]))
    
# print stats
precisions = np.array(precisions)
recalls = np.array(recalls)
f1_scores = np.array(f1_scores)
print("Precision - Mean: {:.2f}, Std: {:.2f}".format(np.mean(precisions), np.std(precisions)))
print("Recall - Mean: {:.2f}, Std: {:.2f}".format(np.mean(recalls), np.std(recalls)))
print("F1-score - Mean: {:.2f}, Std: {:.2f}".format(np.mean(f1_scores), np.std(f1_scores)))

# Plot the values
plt.figure(figsize=(10, 6))
plt.plot(precisions, label="Precision", linestyle=":")
# --
plt.plot(recalls, label="Recall")
# ..
plt.plot(f1_scores, label="F1-score", linestyle="--")
plt.xlabel("Training Cycle")
plt.ylabel("Score")
plt.title("PT4AL Metrics")
plt.legend()
plt.show()