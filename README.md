# Adversarial Example Games
![alt text](https://github.com/joeybose/Adversarial-Example-Games/blob/main/aeg_log_reg.gif "AEG Logistic Regression")

This repo contains code for the NeurIPS 2020 paper ["Adversarial Example Games"](https://arxiv.org/abs/2007.00720)

If this repository is helpful in your research, please consider citing us.
```
@article{bose2020adversarial,
  title={Adversarial Example Games},
  author={Bose, Avishek Joey and Gidel, Gauthier and Berard, Hugo and Cianflone, Andre and Vincent, Pascal and Lacoste-Julien, Simon and Hamilton, William L},
  journal={Thirty-fourth Conference on Neural Information Processing Systems},
  year={2020}
}
```

The current codebase is Research code and is subject to change.

## Requirements
Assuming you are in a clean Python 3.6 environment, to install all required packages and download required data, simply run:
```
bash setup.sh
```
Requirements. Checkout the environment.yml file for a comprehensive list taken from pip freeze.
```
wandb (latest version) \
pytorch==1.4 \
torchvision \
cudatoolkit==10.1\
PIL \
numpy \
json \
wandb \
tqdm \
matplotlib \
image \
ipdb \
[advertorch](https://github.com/BorealisAI/advertorch) \
```

## Sample Commands
Please refer to the [attacks sub-directory](attacks/) to find example commands while the
defenses subdirectory contains various defenses.

## Directory Structure
To reproduce our results and to ensure that the commands provided in other README's please ensure that you download all our models
(or re-train them) and each following subdirectory has the following structure. Some of this will be automated when you run the setup scripts.
Each of the following trees are subdirectories with trained models. You may have to either copy them or use `mkdir` command to create these.
### Data Directory
```
.
├── cifar-10-batches-py
│   ├── batches.meta
│   ├── data_batch_1
│   ├── data_batch_2
│   ├── data_batch_3
│   ├── data_batch_4
│   ├── data_batch_5
│   ├── readme.html
│   └── test_batch
├── cifar-10-python.tar.gz
├── cifar-split
│   ├── split_6
│   │   └── data.pt
│   └── split_7
│       └── data.pt
└── MNIST
    ├── processed
    │   ├── test.pt
    │   └── training.pt
    ├── raw
    │   ├── t10k-images-idx3-ubyte
    │   ├── t10k-images-idx3-ubyte.gz
    │   ├── t10k-labels-idx1-ubyte
    │   ├── t10k-labels-idx1-ubyte.gz
    │   ├── train-images-idx3-ubyte
    │   ├── train-images-idx3-ubyte.gz
    │   ├── train-labels-idx1-ubyte
    │   └── train-labels-idx1-ubyte.gz
    └── split_7
        └── data.pt
```
### Split-Classifiers
```
.
├── cifar
│   ├── res18
│   │   ├── split_0
│   │   │   └── model_0.pt
│   │   ├── split_1
│   │   │   └── model_0.pt
│   │   ├── split_2
│   │   │   └── model_0.pt
│   │   ├── split_3
│   │   │   └── model_0.pt
│   │   ├── split_4
│   │   │   └── model_0.pt
│   │   └── split_5
│   │       └── model_0.pt
│   ├── VGG16
│   │   ├── model_0.pt
│   │   ├── split_0
│   │   │   └── model_0.pt
│   │   ├── split_1
│   │   │   └── model_0.pt
│   │   ├── split_2
│   │   │   └── model_0.pt
│   │   ├── split_3
│   │   │   └── model_0.pt
│   │   ├── split_4
│   │   │   └── model_0.pt
│   │   └── split_5
│   │       └── model_0.pt
│   └── wide_resnet
│       ├── split_0
│       │   └── model_0.pt
│       ├── split_1
│       │   └── model_0.pt
│       ├── split_2
│       │   └── model_0.pt
│       ├── split_3
│       │   └── model_0.pt
│       ├── split_4
│       │   └── model_0.pt
│       └── split_5
│           └── model_0.pt
├── mnist
│   ├── natural
│   │   ├── split_0
│   │   │   └── model_0.pt
│   │   ├── split_1
│   │   │   └── model_0.pt
│   │   ├── split_2
│   │   │   └── model_0.pt
│   │   ├── split_3
│   │   │   └── model_0.pt
│   │   ├── split_4
│   │   │   └── model_0.pt
│   │   ├── split_5
│   │   │   └── model_0.pt
│   │   └── split_6
│   │       └── model_0.pt
```
### pretrained_classifiers
We have supressed the repeated instances of models but the downloaded models will contain multiple trained versions of each.
```
├── cifar
│   ├── dense121
│   │   ├── model_0.pt
│   ├── ensemble_adv_trained
│   │   ├── dense121_ens
│   │   │   ├── 0.pt
│   │   ├── dense121_ens.pt
│   │   ├── googlenet_ens
│   │   │   ├── 0.pt
│   │   ├── googlenet_ens.pt
│   │   ├── res18_ens
│   │   │   ├── 0.pt
│   │   ├── wide_resnet_ens
│   │   │   ├── 0.pt
│   │   └── wide_resnet_ens.pt
│   ├── googlenet
│   │   ├── model_0.pt
│   ├── lenet
│   │   ├── model_0.pt
│   ├── res18
│   │   ├── model_0.pt
│   ├── VGG16
│   │   ├── model_0.pt
│   └── wide_resnet
│       ├── model_0.pt
├── mnist
│   ├── adv_trained
│   │   ├── model_0.pt
│   │   └── model_1.pt
│   ├── ensemble_adv_trained
│   │   ├── mnist
│   │   │   ├── modelA_adv.pt
│   │   │   ├── modelA_ens.pt
│   │   │   ├── modelA.pt
│   │   │   ├── modelB_ens.pt
│   │   │   ├── modelB.pt
│   │   │   ├── modelC_ens.pt
│   │   │   ├── modelC.pt
│   │   │   ├── modelD_ens.pt
│   │   │   └── modelD.pt
│   │   ├── modelA_adv.pt
│   │   ├── modelA.pt
│   │   ├── modelB_ens.pt
│   │   ├── modelB.pt
│   │   ├── modelC_ens.pt
│   │   ├── modelC.pt
│   │   ├── modelD_ens.pt
│   │   └── modelD.pt
│   └── natural
│       ├── model_0.pt
├── mnist_cnn_0.pt
```

### madry_challenge_models
```
.
├── cifar
│   ├── adv_trained
│   │   ├── checkpoint
│   │   ├── checkpoint-70000.data-00000-of-00001
│   │   ├── checkpoint-70000.index
│   │   ├── checkpoint-70000.meta
│   │   └── config.json
│   ├── natural
│   │   ├── checkpoint
│   │   ├── checkpoint-70000.data-00000-of-00001
│   │   ├── checkpoint-70000.index
│   │   ├── checkpoint-70000.meta
│   │   └── config.json
│   ├── naturally_trained
│   │   ├── checkpoint
│   │   ├── checkpoint-70000.data-00000-of-00001
│   │   ├── checkpoint-70000.index
│   │   ├── checkpoint-70000.meta
│   │   └── config.json
│   └── secret
│       ├── checkpoint
│       ├── checkpoint-70000.data-00000-of-00001
│       ├── checkpoint-70000.index
│       ├── checkpoint-70000.meta
│       └── config.json
└── mnist
    ├── adv_trained
    │   ├── checkpoint
    │   ├── checkpoint-99900.data-00000-of-00001
    │   ├── checkpoint-99900.index
    │   ├── checkpoint-99900.meta
    │   ├── config.json
    │   ├── mnist_lenet5_advtrained.pt
    │   └── mnist_test_lenet5_advtrained.pt
    ├── natural
    │   ├── checkpoint
    │   ├── checkpoint-24900.data-00000-of-00001
    │   ├── checkpoint-24900.index
    │   ├── checkpoint-24900.meta
    │   ├── config.json
    │   └── mnist_lenet5_clntrained.pt
    └── secret
        ├── checkpoint
        ├── checkpoint-99900.data-00000-of-00001
        ├── checkpoint-99900.index
        ├── checkpoint-99900.meta
        └── config.json
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)


