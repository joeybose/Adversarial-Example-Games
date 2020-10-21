from __future__ import print_function

import os
import argparse
import ipdb
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from utils.utils import load_mnist
from cnn_models import LeNet as Net
from cnn_models import MadryLeNet as MadryNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.test_utils import LeNet5
from advertorch_examples.utils import get_mnist_train_loader
from advertorch_examples.utils import get_mnist_test_loader

# Training script taken from:
# https://github.com/BorealisAI/advertorch/blob/master/advertorch_examples/tutorial_train_mnist.py

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Train MNIST')
        parser.add_argument('--save_path', default='../madry_challenge_models/mnist/')
        parser.add_argument('--seed', default=0, type=int)
        parser.add_argument('--mode', default="cln", help="cln | adv")
        parser.add_argument('--batch_size', default=50, type=int)
        parser.add_argument('--test_batch_size', default=1000, type=int)
        parser.add_argument('--log_interval', default=200, type=int)
        parser.add_argument('--train_set', default=None, help="train | test | train_and_test")
        parser.add_argument('--architecture', default='LeNet', type = str)
        self.parser = parser
         
    def parse_args(self):
        parser = self.parser
        args = parser.parse_args()

        return args


def run(args):
    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.mode == "cln":
        flag_advtrain = False
        nb_epoch = 10
        model_filename = "mnist_lenet5_clntrained_%i.pt"%args.seed
    elif args.mode == "adv":
        flag_advtrain = True
        nb_epoch = 90
        model_filename = "mnist_lenet5_advtrained_%i.pt"%args.seed
    else:
        raise

    train_loader, test_loader = load_mnist(args, augment=False, root='../data/')
    if args.architecture == 'LeNet':
        model = Net(1, 28, 28).to(device)
    elif args.architecture == 'MadryLeNet':
        model = MadryNet(1,28,28).to(device)
        print(device)
    else:
        raise(f'Architecture {args.architecture} not implemented')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if flag_advtrain:
        from advertorch.attacks import LinfPGDAttack
        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
            nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
            clip_max=1.0, targeted=False)
        save_path = os.path.join(args.save_path, "adv_trained")
    else:
        save_path = os.path.join(args.save_path, "natural")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path = os.path.join(save_path, model_filename)

    for epoch in range(nb_epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            ori = data
            if flag_advtrain:
                # when performing attack, the model needs to be in eval mode
                # also the parameters should NOT be accumulating gradients
                with ctx_noparamgrad_and_eval(model):
                    data = adversary.perturb(data, target)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(
                output, target, reduction='mean')
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx *
                    len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        model.eval()
        test_clnloss = 0
        clncorrect = 0

        if flag_advtrain:
            test_advloss = 0
            advcorrect = 0

        for clndata, target in test_loader:
            clndata, target = clndata.to(device), target.to(device)
            with torch.no_grad():
                output = model(clndata)
            test_clnloss += F.cross_entropy(
                output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            clncorrect += pred.eq(target.view_as(pred)).sum().item()

            if flag_advtrain:
                advdata = adversary.perturb(clndata, target)
                with torch.no_grad():
                    output = model(advdata)
                test_advloss += F.cross_entropy(
                    output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                advcorrect += pred.eq(target.view_as(pred)).sum().item()

        test_clnloss /= len(test_loader.dataset)
        print('\nTest set: avg cln loss: {:.4f},'
              ' cln acc: {}/{} ({:.0f}%)\n'.format(
                  test_clnloss, clncorrect, len(test_loader.dataset),
                  100. * clncorrect / len(test_loader.dataset)))
        if flag_advtrain:
            test_advloss /= len(test_loader.dataset)
            print('Test set: avg adv loss: {:.4f},'
                  ' adv acc: {}/{} ({:.0f}%)\n'.format(
                      test_advloss, advcorrect, len(test_loader.dataset),
                      100. * advcorrect / len(test_loader.dataset)))

    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    args = Config().parse_args()
    run(args)