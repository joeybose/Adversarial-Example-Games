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
from cnn_models import LeNet as Net
from cnn_models import ResNet18
from cnn_models.mnist_ensemble_adv_train_models import *
from eval import baseline_transfer, baseline_eval_classifier
from defenses.ensemble_adver_train_mnist import *

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

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
    parser.add_argument('--noise', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=256, metavar='S')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='S')
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
    parser.add_argument('--train_on_list', default=False, action='store_true',
                        help='train on a list of classifiers')
    parser.add_argument('--attack_ball', type=str, default="Linf",
                        choices= ['L2','Linf'])
    parser.add_argument('--source_arch', default="res18",
                        help="The architecture we want to attack on CIFAR.")
    parser.add_argument('--target_arch', default=None,
                        help="The architecture we want to blackbox transfer to on CIFAR.")
    parser.add_argument('--split', type=int, default=None,
                        help="Which subsplit to use.")
    parser.add_argument('--epsilon', type=float, default=0.1, metavar='M',
                        help='Epsilon for Delta (default: 0.1)')
    parser.add_argument('--num_test_samples', default=None, type=int,
                        help="The number of samples used to train and test the attacker.")
    parser.add_argument('--train_with_critic_path', type=str, default=None,
                        help='Train generator with saved critic model')
    parser.add_argument('--model', help='path to model')
    parser.add_argument('--adv_models', nargs='*', help='path to adv model(s)')
    parser.add_argument('--type', type=int, default=0, help='Model type (default: 0)')
    parser.add_argument('--namestr', type=str, default='NoBox', \
            help='additional info in output filename to describe experiments')
    args = parser.parse_args()
    args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, split_train_loader, split_test_loader = create_loaders(args,
            root='../data', split=args.split)

    if args.split is not None:
        train_loader = split_train_loader
        test_loader = split_test_loader

    if os.path.isfile("../settings.json"):
        with open('../settings.json') as f:
            data = json.load(f)
        args.wandb_apikey = data.get("wandbapikey")

    if args.wandb:
        os.environ['WANDB_API_KEY'] = args.wandb_apikey
        wandb.init(project='NoBox-sweeps', name='MI-Attack-{}'.format(args.dataset))

    model, adv_models, l_test_classif_paths, model_type = data_and_model_setup(args)
    model.to(args.dev)
    model.eval()

    print("Testing on %d Test Classifiers with Source Model %s" %(len(l_test_classif_paths), args.source_arch))

    if args.attack_ball == 'Linf':
        attacker = LinfMomentumIterativeAttack(model,
                                               loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                               eps=args.epsilon,
                                               nb_iter=args.n_iter,
                                               decay_factor=1., eps_iter=0.01,
                                               clip_min=0., clip_max=1.,
                                               targeted=False)
    elif args.attack_ball == 'L2':
        attacker = L2MomentumIterativeAttack(model,
                                             loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                             eps=args.epsilon,
                                             nb_iter=args.n_iter,
                                             decay_factor=1., eps_iter=0.01,
                                             clip_min=0., clip_max=1.,
                                             targeted=False)
    else:
        raise NotImplementedError

    eval_helpers = [model, model_type, adv_models, l_test_classif_paths, test_loader]
    total_fool_rate = eval(args, attacker, "MI-Attack", eval_helpers)

if __name__ == '__main__':
    main()
