## LGAN
Source codes for ["A Lightweight Local-Global Attention Network for Single Image Super-Resolution"](). It is accepted by [ACCV2022](https://accv2022.org/).

## Step 1: Install environment
The required environment is listed in 'env.yaml', please install the environment before training.

## Step 2: Edit config file
Settings can be edited in 'configs/config.yml'.

## Step 3: Prepare the dataset
You can download the dataset from [here](https://drive.google.com/file/d/1m4dZ1CARTohwu6tOznHtS-XjC4hnwCgm/view?usp=sharing).
```
|/SR_datasets
|-|/DIV2K
|-|-|/DIV2K_train_HR/
|-|-|/DIV2K_train_LR_bicubic/
|-|/benchmark
|-|-|/B100
|-|-|-|/HR
|-|-|-|/LR_bicubic
|-|-|/Manga109
|-|-|-|/HR
|-|-|-|/LR_bicubic
|-|-|/Set14
|-|-|-|/HR
|-|-|-|/LR_bicubic
|-|-|/Set5
|-|-|-|/HR
|-|-|-|/LR_bicubic
|-|-|/Urban100
|-|-|-|/HR
|-|-|-|/LR_bicubic
```

## Step 4: Train the network
Run
```
sh train.sh
```
or you can download [pre-trained models](https://drive.google.com/file/d/141eRsRVGYDe-zzxF_bzYi8IyZzNP9-fd/view?usp=sharing, https://drive.google.com/file/d/15GS6tapZC57Hr9OnBfJLcXvP8WkNACSz/view?usp=sharing, https://drive.google.com/file/d/1F3lasAWstsEVfaQefX1HiRAprOk5txvQ/view?usp=sharing)

## Step 5: Evaluate the network
To evaluate the network, you should specify the parameter 'pretrain' in 'configs/config.yaml' first.
And run
```
sh test.sh
```
The visual result produced by LGAN can be obtained [here](https://drive.google.com/file/d/1WUxKDC3n07UIK0QFolREgTdwABwBWobU/view?usp=sharing).

## Thanks
The codes are implemented based on [ELAN](https://github.com/xindongzhang/ELAN).

A neat network framework is necessary for SISR and allows us to focus more on improving the structure of network.
We are grateful for [ELAN](https://github.com/xindongzhang/ELAN).

