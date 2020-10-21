# NoBoxAttack

This repo contains code for the NeurIPS 2020 paper ["Adversarial Example Games"](https://arxiv.org/abs/2007.00720)

If this repository is helpful in your research, please consider citing us.
```
@article{bose2020adversarial,
  title={Adversarial Example Games},
  author={Bose, Avishek Joey and Gidel, Gauthier and Berard, Hugo and Cianflone, Andre and Vincent, Pascal and Lacoste-Julien, Simon and Hamilton, William L},
  journal={arXiv preprint arXiv:2007.00720},
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
Please refer to the attacks sub-directory to find example commands while the
defenses subdirectory contains various defenses.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)


