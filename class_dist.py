import numpy as np

def get_class_dist(img_list, batch_num, num_classes, out_file):
    class_dist = np.zeros(num_classes)
    for img_path in img_list:
        class_index = int(img_path.split('/')[-2])
        class_dist[class_index] +=1

    with open(out_file, 'a') as f:
        f.write(f'{batch_num} Class Distribution: {class_dist}\n')
        