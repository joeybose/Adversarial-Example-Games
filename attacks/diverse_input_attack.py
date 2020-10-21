from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.bernoulli import Bernoulli
import torchvision.transforms as transforms
from advertorch.attacks import LinfMomentumIterativeAttack, L2MomentumIterativeAttack
from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import clamp_by_pnorm
from advertorch.utils import clamp
from advertorch.utils import batch_clamp
from advertorch.utils import batch_multiply

import argparse
import os
import ipdb
import json
import sys
sys.path.insert(0, "..")  # Adds higher directory to python modules path.
from attacks import data_and_model_setup, load_data, eval
from attacks.iterative_attacks import BIM
from utils.utils import create_loaders, load_unk_model
from classifiers import load_all_classifiers, load_list_classifiers, load_dict_classifiers
from cnn_models import LeNet as Net
from cnn_models import ResNet18
from cnn_models.mnist_ensemble_adv_train_models import *
from eval import baseline_transfer, baseline_eval_classifier
from defenses.ensemble_adver_train_mnist import *

class DIM(LinfMomentumIterativeAttack):
    def __init__(self, args, model,
                 loss_fn=nn.CrossEntropyLoss(reduction="sum"), decay_factor=1.,
                 attack_ball='Linf', eps=0.3, eps_iter=0.01, n_iter=50,
                 clip_max=1., clip_min=-0.):
        super(DIM, self).__init__(model, loss_fn=loss_fn, eps=eps,
                                  nb_iter=n_iter, decay_factor=decay_factor,
                                  eps_iter=eps_iter, clip_min=clip_min,
                                  clip_max=clip_max)
        self.model = model
        self.eps = eps
        self.eps_iter = eps_iter
        self.n_iter = n_iter
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.attack_ball = attack_ball
        self.momentum = args.momentum
        self.transform_prob = args.transform_prob
        self.apply_transform = Bernoulli(torch.tensor([self.transform_prob]))
        self.resize_factor = args.resize_factor
        self.args = args

    def input_diversity(self, input_tensor):
        _, c, h, w = input_tensor.size()
        image_resize = int(self.resize_factor* w)
        rnd = torch.randint(h, image_resize, [1])
        h_rem = image_resize - rnd
        w_rem = image_resize - rnd
        self.pad_top = torch.randint(low=0, high=h_rem.item(), size=[1])
        self.pad_bottom = h_rem - self.pad_top
        self.pad_left = torch.randint(0, w_rem.item(), [1])
        self.pad_right = w_rem - self.pad_left
        device = input_tensor[0].device
        apply_prob = self.apply_transform.sample()
        if apply_prob:
            inp = F.interpolate(input_tensor, size=(rnd.item(), rnd.item()),
                                mode='bilinear')
            out = F.pad(inp, pad=(self.pad_left, self.pad_right,
                                  self.pad_top, self.pad_bottom))
        else:
            out = input_tensor
        return out.to(device)

    def perturb(self, x, y):
        x, y = self._verify_and_process_inputs(x, y)
        delta = torch.zeros_like(x)
        g = torch.zeros_like(x)
        delta = nn.Parameter(delta)

        for i in range(self.nb_iter):
            if delta.grad is not None:
                delta.grad.detach_()
                delta.grad.zero_()

            imgadv = x + delta
            diverse_x = self.input_diversity(imgadv)
            outputs = self.predict(diverse_x)
            loss = self.loss_fn(outputs, y)
            if self.targeted:
                loss = -loss
            loss.backward()

            g = self.decay_factor * g + normalize_by_pnorm(
                delta.grad.data, p=1)
            # according to the paper it should be .sum(), but in their
            #   implementations (both cleverhans and the link from the paper)
            #   it is .mean(), but actually it shouldn't matter
            if self.attack_ball == 'Linf':
                delta.data += self.eps_iter * torch.sign(g)
                delta.data = clamp(
                    delta.data, min=-self.eps, max=self.eps)
                delta.data = clamp(
                    x + delta.data, min=self.clip_min, max=self.clip_max) - x
            elif self.attack_ball == 'L2':
                delta.data += self.eps_iter * normalize_by_pnorm(g, p=2)
                delta.data *= clamp(
                    (self.eps * normalize_by_pnorm(delta.data, p=2) /
                        delta.data),
                    max=1.)
                delta.data = clamp(
                    x + delta.data, min=self.clip_min, max=self.clip_max) - x
            else:
                error = "Only ord = inf and ord = 2 have been implemented"
                raise NotImplementedError(error)

        rval = x + delta.data
        return rval

def main(args):
    args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.isfile("../settings.json"):
        with open('../settings.json') as f:
            data = json.load(f)
        args.wandb_apikey = data.get("wandbapikey")

    if args.wandb:
        os.environ['WANDB_API_KEY'] = args.wandb_apikey
        wandb.init(project='NoBox-sweeps', name='AutoAttack-{}'.format(args.dataset))

    train_loader, test_loader, split_train_loader, split_test_loader = create_loaders(args,
            root='../data', split=args.split)

    if args.split is not None:
        train_loader = split_train_loader
        test_loader = split_test_loader

    model, adv_models, l_test_classif_paths, model_type = data_and_model_setup(args, di_attack=True)
    model.to(args.dev)
    model.eval()

    print("Testing on %d Test Classifiers with Source Model %s" %(len(l_test_classif_paths), args.source_arch))

    attacker = DIM(args, model, attack_ball=args.attack_ball, eps=args.epsilon,
                   n_iter=args.n_iter, decay_factor=args.momentum)


    eval_helpers = [model, model_type, adv_models, l_test_classif_paths, test_loader]
    total_fool_rate = eval(args, attacker, "DI-Attack", eval_helpers)
    return total_fool_rate

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--n_iter', type=int, default=1000)
parser.add_argument('--transfer', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--sweep', action='store_true')
parser.add_argument("--wandb", action="store_true", default=False, help='Use wandb for logging')
parser.add_argument('--ensemble_adv_trained', action='store_true')
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
parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                    help='Randomly apply input Transformation')
parser.add_argument('--transform_prob', type=float, default=0.5, metavar='M',
                    help='Randomly apply input Transformation')
parser.add_argument('--resize_factor', type=float, default=1.1, metavar='M',
                    help='Resize Factor for Random Resizing')
parser.add_argument('--split', type=int, default=None,
                    help="Which subsplit to use.")
parser.add_argument('--epsilon', type=float, default=0.1, metavar='M',
                    help='Epsilon for Delta (default: 0.1)')
parser.add_argument('--train_with_critic_path', type=str, default=None,
                    help='Train generator with saved critic model')
parser.add_argument('--num_test_samples', default=None, type=int,
                    help="The number of samples used to train and test the attacker.")
parser.add_argument('--model', help='path to model')
parser.add_argument('--adv_models', nargs='*', help='path to adv model(s)')
parser.add_argument('--type', type=int, default=0, help='Model type (default: 0)')
parser.add_argument('--namestr', type=str, default='NoBox', \
        help='additional info in output filename to describe experiments')
if __name__ == '__main__':

    args = parser.parse_args()
    main(args)
