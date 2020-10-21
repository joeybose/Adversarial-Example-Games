from typing import Any
import argparse
from .ae_models import MnistVAE, CifarVAE, Mnistautoencoder
from .generators import Bottleneck, Generator, ConvGenerator, DCGAN, Cond_DCGAN, MnistGenerator, ResnetGenerator
from torch import nn
from .dcgan28 import DiscriminatorCNN28, GeneratorCNN28

__all__ = [
    "MnistVAE",
    "CifarVAE",
    "Mnistautoencoder",
    "Generator",
    "ConvGenerator",
    "DCGAN",
    "Cond_DCGAN",
    "ResNetGenerator"
]

def create_generator(arg_parse: argparse.Namespace, model_type: str,
                     deterministic: bool, dataset_type: str, *args: Any,
                     **kwargs: Any):

    if dataset_type == 'mnist':
        if model_type == 'DC_GAN':
            # G = ResNet32Generator(arg_parse.dev, epsilon=arg_parse.epsilon,
            #                       norm=arg_parse.attack_ball)
            G = GeneratorCNN28(arg_parse.dev, epsilon=arg_parse.epsilon,
                               norm=arg_parse.attack_ball, img_channels=1,
                               img_dim=784).to(arg_parse.dev)
        elif model_type == 'CondGen':
            G = MnistGenerator(norm=arg_parse.attack_ball).to(arg_parse.dev)
        else:
            G = Generator(input_size=784).to(arg_parse.dev)
            G = nn.DataParallel(G)
    elif dataset_type =='cifar':
        if model_type == 'DC_GAN':
            raise("DC GAN not for CIFAR")
            G = ResnetGenerator(arg_parse.dev, input_nc=3, output_nc=3,
                                epsilon=arg_parse.epsilon,
                                norm=arg_parse.attack_ball).to(arg_parse.dev)
        elif model_type == 'CondGen':
            G = ConvGenerator(3,Bottleneck, [6,12,24,16], deterministic, *args,
                              **kwargs, norm=arg_parse.attack_ball).to(arg_parse.dev)
        elif model_type == "Resnet":
            G = ResnetGenerator(arg_parse.dev,3,3, epsilon=arg_parse.epsilon,
                               norm=arg_parse.attack_ball).to(arg_parse.dev)
            # G = ConvGenerator(3,Bottleneck, [6,12,24,16], deterministic, *args,
            #                   **kwargs).to(arg_parse.dev)
            G = nn.DataParallel(G)
        else:
            ValueError(f"model {model_type} not recognized")
    else:
        raise ValueError(f"Unknown dataset type: '{dataset_type}'.")
    return G
