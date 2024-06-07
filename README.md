# Skip-GANomaly++: Skip Connections and Residual Blocks for Anomaly Detection


This repository contains the implementation of the following paper: Skip-GANomaly++: Skip Connections and Residual Blocks for Anomaly Detection 

## 1. Table of Contents
- [Skip-GANomaly](#skip-ganomaly)
  - [1. Table of Contents](#1-table-of-contents)
  - [2. Installation](#2-installation)
  - [3. Experiment](#3-experiment)
  - [4. Training](#4-training)
    - [4.1. Train on CIFAR10](#41-training-on-cifar10)
    - [4.2. Train on Custom Dataset](#42-train-on-custom-dataset)
  - [5. Citing Skip-GANomaly](#5-citing-skip-ganomaly)
  - [6. Reference](#6-reference)

## 2. Installation
1. First clone the repository
   ```
   git clone https://github.com/crinex/SkipGANomalyPP.git
   ```
2. Create the virtual environment via conda
    ```
    conda create -n skipganomalyPP python=3.7
    ```
3. Activate the virtual environment.
    ```
    conda activate skipganomalyPP
    ```
4. Install the dependencies.
   ```
   pip install --user --requirement requirements.txt
   ```

## 3. Experiment
To replicate the results in the paper for CIFAR10  dataset, run the following commands:

``` shell
# CIFAR
sh experiments/run_cifar.sh
```

## 4. Training
To list the arguments, run the following command:
```
python train.py -h
```

### 4.1. Training on CIFAR10
To train the model on CIFAR10 dataset for a given anomaly class, run the following:

``` 
python train.py \
    --dataset cifar10                                                             \
    --niter <number-of-epochs>                                                    \
    --abnormal_class                                                              \
        <airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck>    \
    --model skipganomalypp                                                        \
    --save_test_images                                                            \
```

### 4.2. Train on Custom Dataset
To train the model on a custom dataset, the dataset should be copied into `./data` directory, and should have the following directory & file structure:

```
Custom Dataset
├── test
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_n.png
│   ├── 1.abnormal
│   │   └── abnormal_tst_img_0.png
│   │   └── abnormal_tst_img_1.png
│   │   ...
│   │   └── abnormal_tst_img_m.png
├── train
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_t.png

```

Then model training is the same as the training explained above.

```
python train.py                     \
    --dataset <name-of-the-data>    \
    --isize <image-size>            \
    --niter <number-of-epochs>      \
    --model skipganomalypp          \
    --save_test_images              \
```

For more training options, run `python train.py -h`.

## 5. Citing Skip-GANomaly++
If you use this repository or would like to refer the paper, please use the following BibTeX entry
```
@INPROCEEDINGS {Akcay2019SkipGANomaly,
    author    = "June-Young Park, Jae-Ryung Hong, Min-Hye Kim and Tae-Joon Kim",
    title     = "Skip-GANomaly++: Skip Connections and Residual Blocks for Anomaly Detection",
    booktitle = "Proceedings of the 38th AAAI Conference on Artificial Intelligence, AAAI 2024",
    year      = "2024",
    pages     = "1-8",
    month     = "feb",
    publisher = "AAAI"
}
```
