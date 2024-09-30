# [PT4AL: Using Self-Supervised Pretext Tasks for Active Learning (ECCV2022)](https://arxiv.org/abs/2201.07459) - Official Pytorch Implementation
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/using-self-supervised-pretext-tasks-for/active-learning-on-cifar10-10000)](https://paperswithcode.com/sota/active-learning-on-cifar10-10000?p=using-self-supervised-pretext-tasks-for)

# Update Note

- We solved all problems. The issue is that the epoch of the rotation prediction task was supposed to run only 15 epochs, but it was written incorrectly as 120 epochs. Sorry for the inconvenience. [2023.01.02]
- Add Cold Start Experiments

```
[solved problem]
We are redoing the CIFAR10 experiment.

The current reproduction result is the performance of 91 to 93.

We will re-tune the code again for stable performance in the near future.

The rest of the experiments confirmed that there was no problem with reproduction.
```
Sorry for the inconvenience.
## Experiment Setting:
- CIFAR10 (downloaded and saved in ```./DATA```
- Rotation prediction for pretext task

## Prerequisites:
Python >= 3.7

CUDA = 11.0

PyTorch = 1.7.1

numpy >= 1.16.0

## Running the Code

Change the test parameters in ```config.py```

To train the rotation predition task on the unlabeled set:
```
python rotation.py
```

To extract pretext task losses and create batches:
```
python make_batches.py
```

To evaluate on active learning task:
```
python main.py
```
![image](https://user-images.githubusercontent.com/33244972/210195074-8cc85f97-8a20-4aac-b61b-034e91694788.png)


To mask cold start experiments (PT4AL)
```
python main_pt4al.py
```
![image](https://user-images.githubusercontent.com/33244972/210192180-6158a4ea-052b-4313-baf9-0048aaa5746f.png)


To get the metrics for a specific run, run ```utils/plot_curves.py```, changing the path of the metrics file

To compare two runs (e.g. Random vs PT4AL), run ```utils/generate_acc_curves.py```

To evaluate a classifier in the whole dataset, without active learning, run ```whole_dataset.py```



## Citation
If you use our code in your research, or find our work helpful, please consider citing us with the bibtex below:
```
@inproceedings{yi2022using,
  title = {Using Self-Supervised Pretext Tasks for Active Learning},
  author = {Yi, John Seon Keun and Seo, Minseok and Park, Jongchan and Choi, Dong-Geol},
  booktitle = {Proc. ECCV},
  year = {2022},
}
```
