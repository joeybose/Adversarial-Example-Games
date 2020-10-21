from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.attacks import LinfPGDAttack, MomentumIterativeAttack
from advertorch.attacks import LinfMomentumIterativeAttack, L2MomentumIterativeAttack

import argparse
import os
import ipdb
import json
import sys
from __init__ import data_and_model_setup, load_data, eval
sys.path.append("..")  # Adds higher directory to python modules path.
from utils.utils import create_loaders, load_unk_model, test_classifier
from classifiers import load_all_classifiers, load_list_classifiers, load_dict_classifiers
from cnn_models.mnist_ensemble_adv_train_models import *
from cnn_models import LeNet as Net
from cnn_models import ResNet18
from eval import baseline_transfer, baseline_eval_classifier
from defenses.ensemble_adver_train_mnist import *

# Code Taken from: https://github.com/csdongxian/skip-connections-matter/
def backward_hook(gamma):
    # implement SGM through grad through ReLU
    def _backward_hook(module, grad_in, grad_out):
        if isinstance(module, nn.ReLU):
            return (gamma * grad_in[0],)
    return _backward_hook


def backward_hook_norm(module, grad_in, grad_out):
    # normalize the gradient to avoid gradient explosion or vanish
    std = torch.std(grad_in[0])
    return (grad_in[0] / std,)


def register_hook_for_resnet(model, arch, gamma):
    # There is only 1 ReLU in Conv module of ResNet-18/34
    # and 2 ReLU in Conv module ResNet-50/101/152
    if arch in ['res50', 'res101', 'res152']:
        gamma = np.power(gamma, 0.5)
    backward_hook_sgm = backward_hook(gamma)

    for name, module in model.named_modules():
        if 'relu' in name and not '0.relu' in name:
            module.register_backward_hook(backward_hook_sgm)

        # e.g., 1.layer1.1, 1.layer4.2, ...
        # if len(name.split('.')) == 3:
            # module.register_backward_hook(backward_hook_norm)


def register_hook_for_densenet(model, arch, gamma):
    # There are 2 ReLU in Conv module of DenseNet-121/169/201.
    gamma = np.power(gamma, 0.5)
    backward_hook_sgm = backward_hook(gamma)
    for name, module in model.named_modules():
        if 'relu' in name and not 'transition' in name:
            module.register_backward_hook(backward_hook_sgm)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--n_iter', type=int, default=1000)
    parser.add_argument('--query_step', type=int, default=1)
    parser.add_argument('--transfer', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument("--wandb", action="store_true", default=False, help='Use wandb for logging')
    parser.add_argument('--ensemble_adv_trained', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=256, metavar='S')
    parser.add_argument('--test_batch_size', type=int, default=128, metavar='S')
    parser.add_argument('--train_set', default='test',
                        choices=['train_and_test','test','train'],
                        help='add the test set in the training set')
    parser.add_argument('--modelIn', type=str,
                        default='../pretrained_classifiers/cifar/res18/model_0.pt')
    parser.add_argument('--robust_model_path', type=str,
                        default="../madry_challenge_models/mnist/adv_trained/mnist_lenet5_advtrained.pt")
    parser.add_argument('--dir_test_models', type=str,
                        default="../",
                        help="The path to the directory containing the classifier models for evaluation.")
    parser.add_argument("--max_test_model", type=int, default=2,
                    help="The maximum number of pretrained classifiers to use for testing.")
    parser.add_argument('--train_on_madry', default=False, action='store_true',
                        help='Train using Madry tf grad')
    parser.add_argument('--momentum', default=0.0, type=float)
    parser.add_argument('--train_on_list', default=False, action='store_true',
                        help='train on a list of classifiers')
    parser.add_argument('--attack_ball', type=str, default="Linf",
                        choices= ['L2','Linf'])
    parser.add_argument('--source_arch', default="res18",
                        help="The architecture we want to attack on CIFAR.")
    parser.add_argument('--adv_models', nargs='*', help='path to adv model(s)')
    parser.add_argument('--target_arch', default=None,
                        help="The architecture we want to blackbox transfer to on CIFAR.")
    parser.add_argument('--epsilon', type=float, default=0.03125, metavar='M',
                        help='Epsilon for Delta (default: 0.1)')
    parser.add_argument('--num_test_samples', default=None, type=int,
                        help="The number of samples used to train and test the attacker.")
    parser.add_argument('--split', type=int, default=None,
                        help="Which subsplit to use.")
    parser.add_argument('--step-size', default=2, type=float, help='perturb step size')
    parser.add_argument('--train_with_critic_path', type=str, default=None,
                        help='Train generator with saved critic model')
    parser.add_argument('--namestr', type=str, default='SGM', \
            help='additional info in output filename to describe experiments')

    args = parser.parse_args()
    args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, split_train_loader, split_test_loader = create_loaders(args,
            root='../data', split=args.split)
    if os.path.isfile("../settings.json"):
        with open('../settings.json') as f:
            data = json.load(f)
        args.wandb_apikey = data.get("wandbapikey")

    if args.wandb:
        os.environ['WANDB_API_KEY'] = args.wandb_apikey
        wandb.init(project='NoBox-sweeps', name='AutoAttack-{}'.format(args.dataset))

    model, adv_models, l_test_classif_paths, model_type = data_and_model_setup(args)
    model.to(args.dev)
    model.eval()

    if args.step_size < 0:
        step_size = args.epsilon / args.n_iter
    else:
        step_size = args.step_size / 255.0

    # using our method - Skip Gradient Method (SGM)
    if args.gamma < 1.0:
        if args.source_arch in ['res18', 'res34', 'res50', 'res101', 'res152',
                                'wide_resnet']:
            register_hook_for_resnet(model, arch=args.source_arch, gamma=args.gamma)
        elif args.source_arch in ['dense121', 'dens169', 'dense201']:
            register_hook_for_densenet(model, arch=args.source_arch, gamma=args.gamma)
        else:
            raise ValueError('Current code only supports resnet/densenet. '
                             'You can extend this code to other architectures.')

    if args.momentum > 0.0:
        print('using PGD attack with momentum = {}'.format(args.momentum))
        attacker = MomentumIterativeAttack(predict=model,
                                           loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                           eps=args.epsilon,
                                           nb_iter=args.n_iter,
                                           eps_iter=step_size,
                                           decay_factor=args.momentum,
                                           clip_min=0.0, clip_max=1.0,
                                           targeted=False)
    else:
        print('using Linf PGD attack')
        attacker = LinfPGDAttack(predict=model,
                                 loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                 eps=args.epsilon, nb_iter=args.n_iter,
                                 eps_iter=step_size, rand_init=False,
                                 clip_min=0.0, clip_max=1.0, targeted=False)


    eval_helpers = [model, model_type, adv_models, l_test_classif_paths, test_loader]
    total_fool_rate = eval(args, attacker, "SGM-Attack", eval_helpers)

if __name__ == '__main__':
    main()
