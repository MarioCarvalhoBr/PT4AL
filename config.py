import os
from math import ceil

class Config:
    def __init__(self, data_dir="./DATA", img_size=32, unlabeled_batch_size=100, unlabeled_batch_percentage_to_label=0.5):
        self._data_dir = data_dir
        train_dir = os.path.join(data_dir, 'train')
        self._num_classes = len(os.listdir(train_dir))
        self._train_set_size = sum([len(files) for _, _, files in os.walk(train_dir)])

        self._img_size = img_size

        self._num_unlabeled_batches = ceil(self._train_set_size / unlabeled_batch_size)
        self._unlabeled_batch_percentage_to_label = unlabeled_batch_percentage_to_label
        self._unlabeled_batch_size = unlabeled_batch_size
        self._labeled_set_increase = int(self._unlabeled_batch_size * self._unlabeled_batch_percentage_to_label)

    @property
    def data_dir(self):
        return self._data_dir
    
    @property
    def num_classes(self):
        return self._num_classes
    
    @property
    def train_set_size(self):
        return self._train_set_size
    
    @property
    def img_size(self):
        return self._img_size
    
    @property
    def num_unlabeled_batches(self):
        return self._num_unlabeled_batches
    
    @property
    def unlabeled_batch_percentage_to_label(self):
        return self._unlabeled_batch_percentage_to_label
    
    @property
    def unlabeled_batch_size(self):
        return self._unlabeled_batch_size
    
    @property
    def labeled_set_increase(self):
        return self._labeled_set_increase

        

if __name__ == '__main__':
    config = Config(data_dir='/home/juan/Documents/tcc/code/PT4AL/DATA')
    print(config.data_dir)
    print(config.num_classes)
    print(config.train_set_size)
    print(config.unlabeled_batch_size)
    print(config.labeled_set_increase)


