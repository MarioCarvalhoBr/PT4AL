from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def calculate_mamona_percentage(image_np):
    total_pixels = image_np.shape[0] * image_np.shape[1]
    # sum the color channels
    image_np = np.sum(image_np, axis=2)
    mamona_pixels = np.sum(image_np == 255)
    return mamona_pixels / total_pixels

if __name__ == "__main__":
    data_dir = "/home/juan/Documents/tcc/datasets/mamonas_64x64/labels"

    mamona_percentages = []

    for img_name in tqdm(os.listdir(data_dir)):
        img = np.array(Image.open(os.path.join(data_dir, img_name)))

        mamona_percentage = calculate_mamona_percentage(img)
        mamona_percentages.append(mamona_percentage)

    mamona_percentages = np.array(mamona_percentages)
    mamona_percentages_1_perc = mamona_percentages[mamona_percentages > 0.01]
    mamona_percentages_10_perc = mamona_percentages[mamona_percentages > 0.1]

    with open('mamona_over_image_pixels.txt', 'w') as f:
        for perc in mamona_percentages:
            f.write(str(perc) + '\n')

    plt.hist(mamona_percentages_1_perc, bins=100)
    plt.xlabel('Mamona over image pixels')
    plt.ylabel('Frequency')
    plt.title('Mamona over image pixels histogram (over 1 percent)')
    plt.savefig('mamona_over_image_pixels_histogram.png')
    plt.clf()

    plt.hist(mamona_percentages_10_perc, bins=100)
    plt.xlabel('Mamona over image pixels')
    plt.ylabel('Frequency')
    plt.title('Mamona over image pixels histogram (over 10 percent)')
    plt.savefig('mamona_over_image_pixels_histogram_10_perc.png')

    print('Quantidade de imagens acima de 1%:', len(mamona_percentages_1_perc))
    print('Quantidade de imagens acima de 10%:', len(mamona_percentages_10_perc))


        