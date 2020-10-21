from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.bernoulli import Bernoulli
import torchvision.transforms as transforms
from iterative_attacks import BIM
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
from diverse_input_attack import DIM
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from utils.utils import create_loaders, load_unk_model
from classifiers import load_all_classifiers
from cnn_models import LeNet as Net
from cnn_models import ResNet18
from cnn_models.mnist_ensemble_adv_train_models import *
from eval import baseline_transfer, baseline_eval_classifier
from defenses.ensemble_adver_train_mnist import *


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

class TIM(DIM):
    def __init__(self, args, stack_kernel, model,
                 loss_fn=nn.CrossEntropyLoss(reduction="sum"), decay_factor=1.,
                 attack_ball='Linf', eps=0.3, eps_iter=0.01, n_iter=50,
                 clip_max=1., clip_min=-0.):
        super(TIM, self).__init__(args, model, loss_fn=loss_fn,
                                  attack_ball=attack_ball,
                                  decay_factor=decay_factor, eps=eps,
                                  eps_iter=eps_iter, n_iter=n_iter,
                                  clip_max=clip_max, clip_min=clip_min)
        self.stack_kernel = stack_kernel

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

            # Main Difference between DIM Attack and this
            delta.grad.data = F.conv2d(delta.grad.data, self.stack_kernel,
                                       stride=1, padding=7)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epsilon', default=32.0, type=float, help='Maximum size of adversarial perturbation.')
    parser.add_argument('--num_iter', default=10, type=int, help="Number of iterations.")
    parser.add_argument("--batch_size", default=256, type=int, help="How many images process at one time.")
    parser.add_argument("--momentum", default=1.0, type=float, help="Momentum.")
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--n_iter', type=int, default=1000)
    parser.add_argument('--transfer', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument("--wandb", action="store_true", default=False, help='Use wandb for logging')
    parser.add_argument('--ensemble_adv_trained', action='store_true')
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
    parser.add_argument('--transform_prob', type=float, default=0.5, metavar='M',
                        help='Randomly apply input Transformation')
    parser.add_argument('--resize_factor', type=float, default=1.1, metavar='M',
                        help='Resize Factor for Random Resizing')
    parser.add_argument('--epsilon', type=float, default=0.1, metavar='M',
                        help='Epsilon for Delta (default: 0.1)')
    parser.add_argument('--train_with_critic_path', type=str, default=None,
                        help='Train generator with saved critic model')
    parser.add_argument('--model', help='path to model')
    parser.add_argument('--adv_models', nargs='*', help='path to adv model(s)')
    parser.add_argument('--type', type=int, default=0, help='Model type (default: 0)')
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
        wandb.init(project='NoBox-sweeps', name='AutoAttack-{}'.format(args.dataset))

    adv_models = None
    if args.dataset == 'cifar':
        args.nc, args.h, args.w = 3, 32, 32
        model, l_test_classif_paths = load_all_classifiers(args, load_archs=[args.source_arch])
        model_type = args.source_arch
        if args.target_arch is not None:
            model_target, l_test_classif_paths = load_all_classifiers(args, load_archs=[args.target_arch])
            model_type = args.target_arch
            del model_target
            torch.cuda.empty_cache()
        if args.ensemble_adv_trained:
            adv_model_names = args.adv_models
            l_test_classif_paths = []
            adv_models = [None] * len(adv_model_names)
            for i in range(len(adv_model_names)):
                adv_path = os.path.join(args.dir_test_models, "pretrained_classifiers",
                                    args.dataset, "ensemble_adv_trained",
                                    adv_model_names[i] + '.pt')
                init_func, _ = ARCHITECTURES[adv_model_names[i]]
                temp_model = init_func().to(args.dev)
                adv_models[i] = nn.DataParallel(temp_model)
                adv_models[i].load_state_dict(torch.load(adv_path))
                l_test_classif_paths.append([adv_path])
            model_type = 'Ensemble Adversarial'
    elif args.dataset == 'mnist':
        args.nc, args.h, args.w = 1, 28, 28
        if args.source_arch == 'natural':
            model, l_test_classif_paths = load_all_classifiers(args, load_archs=["natural"])
            model_type = 'natural'
        elif args.source_arch == 'ens_adv' or args.ensemble_adv_trained:
            adv_model_names = args.adv_models
            adv_models = [None] * len(adv_model_names)
            for i in range(len(adv_model_names)):
                type = get_model_type(adv_model_names[i])
                adv_models[i] = load_model(args, adv_model_names[i], type=type).to(args.dev)

            path = os.path.join(args.dir_test_models, "pretrained_classifiers",
                                args.dataset, "ensemble_adv_trained", args.model)
            model, l_test_classif_paths = load_all_classifiers(args, load_archs=["natural"])
            # model = load_model(args, args.model, type=args.type)
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

    stack_kernel_list = []
    # This is needed for Depth-wise convolution
    for i in range(0, args.nc):
        kernel = gkern(15, 3).astype(np.float32)
        stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
        stack_kernel_list.append( np.expand_dims(stack_kernel, 3))

    stack_kernel = np.stack(stack_kernel_list).squeeze()
    stack_kernel = torch.tensor(stack_kernel).permute(0,3,1,2).to(args.dev)
    attacker = TIM(args, stack_kernel=stack_kernel, model=model,
                   attack_ball=args.attack_ball, eps=args.epsilon,
                   n_iter=args.n_iter, decay_factor=args.momentum,
                   eps_iter=0.01)

    advcorrect = 0

    with ctx_noparamgrad_and_eval(model):
        adv_complete_list = []
        if args.dataset == 'cifar':
            for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
                if (batch_idx + 1) * args.test_batch_size > args.batch_size:
                    break
                x_batch, y_batch = x_batch.to(args.dev), y_batch.to(args.dev)
                adv_complete_list.append(attacker.perturb(x_batch,y_batch))
            adv_complete = torch.cat(adv_complete_list)
        else:
            adv_complete = attacker.perturb(x_test[:args.batch_size],
                                       y_test[:args.batch_size])
    if args.transfer:
        adv_img_list = []
        y_orig = y_test[:args.batch_size]
        for i in range(0, len(adv_complete)):
            adv_img_list.append([adv_complete[i].unsqueeze(0), y_orig[i]])
        # Free memory
        del model
        torch.cuda.empty_cache()
        baseline_transfer(args, attacker, "TI-DI-Attack", model_type,
                          adv_img_list, l_test_classif_paths, adv_models)

if __name__ == '__main__':
    main()
