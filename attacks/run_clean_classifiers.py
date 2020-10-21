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
sys.path.append("..")  # Adds higher directory to python modules path.
from utils.utils import create_loaders, load_unk_model, test_classifier
from classifiers import load_all_classifiers, load_list_classifiers, load_dict_classifiers
from cnn_models import LeNet as Net
from cnn_models import ResNet18
from cnn_models.mnist_ensemble_adv_train_models import *
from eval import baseline_transfer, baseline_eval_classifier
from defenses.ensemble_adver_train_mnist import *

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument("--wandb", action="store_true", default=False, help='Use wandb for logging')
    parser.add_argument('--batch_size', type=int, default=256, metavar='S')
    parser.add_argument('--test_batch_size', type=int, default=512, metavar='S')
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
    parser.add_argument('--train_with_critic_path', type=str, default=None,
                        help='Train generator with saved critic model')
    parser.add_argument('--model', help='path to model')
    parser.add_argument('--namestr', type=str, default='NoBox', \
            help='additional info in output filename to describe experiments')
    args = parser.parse_args()
    args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = create_loaders(args, root='../data')
    if os.path.isfile("../settings.json"):
        with open('../settings.json') as f:
            data = json.load(f)
        args.wandb_apikey = data.get("wandbapikey")

    if args.wandb:
        os.environ['WANDB_API_KEY'] = args.wandb_apikey
        wandb.init(project='NoBox-sweeps', name='MI-Attack-{}'.format(args.dataset))

    adv_models = None
    attacker = None
    if args.dataset == 'cifar':
        args.nc, args.h, args.w = 3, 32, 32
        model, l_test_classif_paths = load_all_classifiers(args, load_archs=[args.source_arch])
        model_type = args.source_arch
    elif args.dataset == 'mnist':
        if args.source_arch == 'natural':
            model, l_test_classif_paths = load_all_classifiers(args, load_archs=["natural"])
            model_type = 'natural'
        elif args.source_arch == 'ens_adv':
            adv_model_names = args.adv_models
            adv_models = [None] * len(adv_model_names)
            for i in range(len(adv_model_names)):
                type = get_model_type(adv_model_names[i])
                adv_models[i] = load_model(args, adv_model_names[i], type=type).to(args.dev)

            path = os.path.join(args.dir_test_models, "pretrained_classifiers",
                                args.dataset, "ensemble_adv_trained", args.model)
            model = load_model(args, args.model, type=args.type)
            l_test_classif_paths = [path]
            model_type = 'Ensemble Adversarial'

    model.to(args.dev)
    model.eval()
    print("Testing on %d Test Classifiers with Source Model %s" %(len(l_test_classif_paths), args.source_arch))
    l = [x.unsqueeze(0) for (x, y) in test_loader.dataset]
    x_test = torch.cat(l, 0).to(args.dev)
    l = [y for (x, y) in test_loader.dataset]
    y_test = torch.Tensor(l).long().to(args.dev)
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print("CUDA Device Count is %d, Error might happen. Use export CUDA_VISIBLE_DEVICES=0" %(device_count))

    test_img_list = []
    x_orig = x_test[:args.batch_size]
    y_orig = y_test[:args.batch_size]
    for i in range(0, len(x_orig)):
        test_img_list.append([x_orig[i].unsqueeze(0), y_orig[i]])

    # Free memory
    del model
    torch.cuda.empty_cache()
    baseline_transfer(args, attacker, "Clean", model_type,
                      test_img_list, l_test_classif_paths, adv_models)

if __name__ == '__main__':
    main()
