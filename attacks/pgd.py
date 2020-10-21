from __future__ import print_function

import os
import argparse
import ipdb
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from utils.utils import load_mnist
from cnn_models import LeNet as Net

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.test_utils import LeNet5
from advertorch_examples.utils import get_mnist_train_loader
from advertorch_examples.utils import get_mnist_test_loader
from advertorch.attacks import Attack
from advertorch.attacks import LinfPGDAttack, L2PGDAttack

class PGDAttack(Attack):
    def __init__(self, args, model, nb_iter,
                 loss_fn=nn.CrossEntropyLoss(reduction="sum")):
        super(PGDAttack, self).__init__(args, model, nb_iter, loss_fn)
        self.args = args
        self.model = model
        if args.attack_ball == 'Linf':
            self.adversary = LinfPGDAttack(self.model, loss_fn=loss_fn,
                                           eps=args.epsilon, nb_iter=nb_iter,
                                           eps_iter=0.01, rand_init=True,
                                           clip_min=args.clip_min,clip_max=args.clip_max,
                                           targeted=False)
        elif args.attack_ball == 'L2':
            self.adversary = L2PGDAttack(self.model, loss_fn=loss_fn,
                                         eps=args.epsilon, nb_iter=nb_iter,
                                         eps_iter=0.01, rand_init=True,
                                         clip_min=args.clip_min,clip_max=args.clip_max,
                                         targeted=False)
        else:
            raise NotImplementedError

    def train(self, train_loader, test_loader, l_test_classifiers, l_train_classif=None):
        pass

    def perturb(self, x, target):
        advcorrect, clncorrect, test_clnloss, test_advloss = 0, 0, 0, 0
        x = x.to(self.args.dev)
        target = target.to(self.args.dev)
        with torch.no_grad():
            output = self.model(x)
        test_clnloss += F.cross_entropy(
            output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        clncorrect += pred.eq(target.view_as(pred)).sum().item()
        advdata = self.adversary.perturb(x, target)
        with torch.no_grad():
            output = self.model(advdata)
        test_advloss += F.cross_entropy(
            output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        advcorrect += pred.eq(target.view_as(pred)).sum().item()
        print('Clean loss: {:.4f},'
              'Adv acc: {}/{} ({:.2f}%)\n'.format(test_clnloss, clncorrect,
                                                  len(x), 100. * clncorrect /
                                                  len(x)))
        print('Adv loss: {:.4f},'
              'Adv acc: {}/{} ({:.2f}%)\n'.format( test_advloss, advcorrect,
                                                  len(x), 100. * advcorrect /
                                                  len(x)))
