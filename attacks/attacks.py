from PIL import Image
from torchvision import transforms
import torch
from torch import nn, optim
from torchvision.models import resnet50
from torchvision.models.vgg import VGG
import torchvision.models.densenet as densenet
import torchvision.models.alexnet as alexnet
from torchvision.utils import save_image
import torch.nn.functional as F
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
from advertorch.utils import clamp
from torch import optim
from torch.autograd import Variable
from torch import autograd
import json
import os
import numpy as np
import argparse
import wandb
from tqdm import tqdm
import ipdb
from attack_helpers import *

import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from utils.utils import *
from tests import Madry_test_generator, whitebox_pgd, L2_test_model


kwargs_attack_loss = {'cross_entropy': attack_ce_loss_func, 'carlini': carlini_wagner_loss}
kwargs_perturb_loss = {'Linf': Linf_dist, 'L2': L2_dist}

def PGD_test_model(args, epoch, test_loader, model, G, nc=1, h=28, w=28):
    """ Testing Phase """
    epsilon = args.epsilon
    test_itr = tqdm(enumerate(test_loader),
                    total=len(test_loader.dataset) / args.test_batch_size)
    correct_test = 0
    for batch_idx, (data, target) in test_itr:
        x, target = data.to(args.dev), target.to(args.dev)
        # for t in range(args.PGD_steps):
        if args.model != 'vanilla_G':
            delta, _ = G(x)
        else:
            delta = G(x)
        delta = delta.view(delta.size(0), nc, h, w)
        # Clipping is equivalent to projecting back onto the l_\infty ball
        # This technique is known as projected gradient descent (PGD)
        if args.attack_ball == 'Linf':
            delta.data.clamp_(-epsilon, epsilon)
        elif args.attack_ball == "L2":
            # TODO: add if when norm is small enough
            perturb_dist = kwargs_perturb_loss[args.perturb_loss](adv_inputs,x)
            delta = delta / perturb_dist
        delta.data = torch.clamp(x.data + delta.data, .0, 1.) - x.data
        pred = model(x.detach() + delta)
        out = pred.max(1, keepdim=True)[1]  # get the index of the max log-probability

        correct_test += out.eq(target.unsqueeze(1).data).sum()

    print('\nTest set: PGD Accuracy: {}/{} ({:.0f}%)\n'
          .format(correct_test, len(test_loader.dataset),
                  100. * correct_test / len(test_loader.dataset)))

    # if args.wandb:
    #     test_acc = 100. * correct_test / len(test_loader.dataset)
    #     img2log_clean = save_image_to_wandb(args, clean_image, file_base+"clean.png",normalize=True)
    #     img2log_adv = save_image_to_wandb(args, adv_image, file_base+"adv.png",normalize=True)
    #     img2log_delta = save_image_to_wandb(args,delta_image, file_base+"delta.png",normalize=True)
    #     wandb.log({"Test Adv Accuracy": test_acc, 'Clean_image':
    #         [wandb.Image(img, caption="Clean") for img in img2log_clean], 'Adv_image':
    #         [wandb.Image(img, caption="Adv") for img in img2log_adv], "Delta":
    #         [wandb.Image(img, caption="Delta") for img in img2log_delta]})




def carlini_wagner_loss(args, output, target, scale_const=1):
    # compute the probability of the label class versus the maximum other
    target_onehot = torch.zeros(target.size() + (args.classes,))
    target_onehot = target_onehot.to(args.device)
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    confidence = 0
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    # if targeted:
    #     # if targeted, optimize for making the other class most likely
    #     loss1 = torch.clamp(other - real + confidence, min=0.)  # equiv to max(..., 0.)
    # else:
    #     if non-targeted, optimize for making this class least likely.
    loss1 = torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.mean(scale_const * loss1)

    return loss


def PGD_white_box_generator(args, train_loader, test_loader, model, G, nc=1, h=28, w=28):
    epsilon = args.epsilon
    opt = optim.Adam(G.parameters(), lr=1e-4)

    # Choose Attack Loss

    perturb_loss_func = kwargs_perturb_loss[args.perturb_loss]

    ''' Training Phase '''
    for epoch in range(0, args.attack_epochs):
        train_itr = tqdm(enumerate(train_loader),
                         total=len(train_loader.dataset) / args.batch_size)
        correct = 0
        PGD_test_model(args, epoch, test_loader, model, G, nc, h, w)
        for batch_idx, (data, target) in train_itr:
            x, target = data.to(args.dev), target.to(args.dev)
            for t in range(args.PGD_steps):
                if args.model != 'vanilla_G':
                    delta, kl_div = G(x)
                else:
                    delta = G(x)
                    kl_div = torch.Tensor([0]).to(args.dev)
                delta = delta.view(delta.size(0), nc, h, w)
                # Clipping is equivalent to projecting back onto the l_\infty ball
                # This technique is known as projected gradient descent (PGD)
                delta.data.clamp_(-epsilon, epsilon)
                delta.data = torch.clamp(x.data + delta.data, .0, 1.) - x.data
                pred = model(x.detach() + delta)
                out = pred.max(1, keepdim=True)[1]  # get the index of the max log-probability
                loss = misclassify_loss_func(args, pred, target) + kl_div.sum()
                # if args.wandb:
                #     args.experiment.log_metric("Whitebox CE loss",loss,step=t)
                opt.zero_grad()
                loss.backward()
                for p in G.parameters():
                    p.grad.data.sign_()
                opt.step()
            correct += out.eq(target.unsqueeze(1).data).sum()

        print("\nTrain: Epoch:{} Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(epoch,
                                                                                   loss, correct,
                                                                                   len(train_loader.dataset),
                                                                                   100. * correct / len(
                                                                                       train_loader.dataset)))

    return out, delta


def L2_white_box_generator(args, train_loader, test_loader, model, G, nc=1, h=28, w=28):
    epsilon = args.epsilon
    opt = optim.Adam(G.parameters())
    mode = "Train"

    # Choose Attack and Perturb Loss Functions
    misclassify_loss_func = kwargs_attack_loss[args.attack_loss]
    perturb_loss_func = kwargs_perturb_loss[args.perturb_loss]

    ''' Training Phase '''
    for epoch in range(0, args.attack_epochs):
        train_itr = tqdm(enumerate(train_loader),
                         total=len(train_loader.dataset) / args.batch_size)
        correct = 0
        if epoch == (args.attack_epochs - 1):
            mode = "Test"
        L2_test_model(args, epoch, test_loader, model, G, nc, h, w, mode=mode)
        for batch_idx, (data, target) in train_itr:
            x, target = data.to(args.dev), target.to(args.dev)
            num_unperturbed = 10
            iter_count = 0
            loss_perturb = 20
            loss_misclassify = 10
            while loss_misclassify > 0 and loss_perturb > 0:
                if args.model != 'vanilla_G':
                    delta, kl_div = G(x)
                    kl_div = kl_div.sum() / len(x)
                else:
                    delta = G(x)
                    kl_div = torch.Tensor([0]).to(args.dev)
                delta = delta.view(delta.size(0), nc, h, w)
                adv_inputs = x.detach() + delta
                adv_inputs = torch.clamp(adv_inputs, .0, 1.0)
                pred = model(adv_inputs)
                out = pred.max(1, keepdim=True)[1]  # get the index of the max log-probability
                loss_misclassify = misclassify_loss_func(args, pred, target)
                loss_perturb = perturb_loss_func(x, adv_inputs) / len(x)
                loss = loss_misclassify + args.LAMBDA * loss_perturb + kl_div
                opt.zero_grad()
                loss.backward()
                opt.step()
                iter_count = iter_count + 1
                num_unperturbed = out.eq(target.unsqueeze(1).data).sum()
                if iter_count > args.max_iter:
                    break
            correct += out.eq(target.unsqueeze(1).data).sum()

        print(f"\nTrain: Epoch:{epoch} Loss: {loss:.4f}, Misclassification Loss :{loss_misclassify:.4f}, "
              f"Perturbation Loss {loss_perturb:.4f} Accuracy: {correct}/{len(train_loader.dataset)} "
              f"({100. * correct.cpu().numpy() / len(train_loader.dataset):.0f}%)\n")

    return out, delta


