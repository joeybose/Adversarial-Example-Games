import torch
from torch import nn
import torch.nn.functional as F
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
from advertorch.utils import clamp
from torch import optim
from torch.autograd import Variable
from torch import autograd
import os
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from utils.utils import *
import ipdb
from advertorch.attacks import LinfPGDAttack, L2PGDAttack

def attack_ce_loss_func(args, pred, targ):
    """
    Want to maximize CE, so return negative since optimizer -> gradient descent
    Args:
        pred: model prediction
        targ: true class, we want to decrease probability of this class
    """
    loss = -nn.CrossEntropyLoss(reduction="sum")(pred, targ)
    loss = loss / len(targ)

    return loss

def carlini_wagner_loss(args, output, target, scale_const=1):
    # compute the probability of the label class versus the maximum other
    target_onehot = torch.zeros(target.size() + (args.classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    confidence = 0
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    # if targeted:
        # # if targeted, optimize for making the other class most likely
        # loss1 = torch.clamp(other - real + confidence, min=0.)  # equiv to max(..., 0.)
    # else:
        # if non-targeted, optimize for making this class least likely.
    loss1 = -torch.clamp(other - real + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.mean(scale_const * loss1)

    return loss


def carlini_wagner_loss2(args, output, target, scale_const=1):
    # compute the probability of the label class versus the maximum other
    target_onehot = torch.zeros(target.size() + (args.classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    # target_var = Variable(target_onehot, requires_grad=False)
    real = (target_onehot * output).sum(1)
    confidence = 0
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    # if targeted:
        # # if targeted, optimize for making the other class most likely
        # loss1 = torch.clamp(other - real + confidence, min=0.)  # equiv to max(..., 0.)
    # else:
        # if non-targeted, optimize for making this class least likely.
    loss = real - other   # equiv to max(..., 0.)
    return torch.mean(loss)

def non_saturating_loss(args,pred,target):
    target_onehot = torch.zeros(target.size() + (args.classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    max_input, _ = torch.max(pred, 1)
    pred = pred - max_input.view(-1, 1).repeat(1, pred.shape[1])
    softval = F.softmax(pred, 1)
    other = ((1.-target_onehot) * softval).sum(1)
    loss = -torch.log(other)
    return torch.mean(loss)

def targeted_cw_loss(args, output, target):
    # compute the probability of the label class versus the maximum other
    real = (target * output).sum(1)
    confidence = 0
    other = ((1. - target) * output - target * 10000.).max(1)[0]
    # if targeted:
    # # if targeted, optimize for making the other class most likely
    # loss1 = torch.clamp(other - real + confidence, min=0.)  # equiv to max(..., 0.)
    # else:
    # if non-targeted, optimize for making this class least likely.
    loss1 = torch.clamp(other - real + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.mean(loss1)
    return loss


def ce_loss_func(args, pred, targ):
    """
    Want to maximize CE, so return negative since optimizer -> gradient descent
    Args:
        pred: model prediction
        targ: true class, we want to decrease probability of this class
    """
    loss = nn.CrossEntropyLoss(reduction="sum")(pred, targ)
    loss = loss / len(targ)

    return loss

def Linf_clamp(args, delta):
    """
    Clamp the image between -epsilon and epsilon with a piecewise linear function.
    delta is assumed to be between -1 and 1
    @param args: supposed to contain args.epsilon and arg.leaky_clamp
    @param delta: image to clamp
    @return: delta_clamped
    """
    if args.epsilon >= 1.:
        raise ValueError(f"Epsilon value should be smaller that 1., "
                         f"current value is {args.epsilon}")
    alpha = args.leaky_clamp
    neg_eps_filter = delta < - args.epsilon
    pos_eps_filter = delta > args.epsilon
    no_clamp_filter = ~ (pos_eps_filter + neg_eps_filter)
    slope = alpha * args.epsilon / (1-args.epsilon) # Slope of the second part of the clamping
    constant = args.epsilon*(1 - alpha / (1-args.epsilon))
    pos_eps_delta = pos_eps_filter * (slope * delta + constant)
    neg_eps_delta = neg_eps_filter * (-slope * delta - constant)
    return pos_eps_delta + neg_eps_delta + no_clamp_filter * (1-alpha) * delta

def L2_clamp(args, delta):
    """
    Clamp the image between -epsilon and epsilon with a piecewise linear function.
    delta is assumed to be between -1 and 1
    @param args: supposed to contain args.epsilon and arg.alpha
    @param delta: image to clamp
    @return: delta_clamped
    """
    if args.epsilon >= 1.:
        raise ValueError(f"Epsilon value should be smaller that 1., "
                         f"current value is {args.epsilon}")
    norm_delta = torch.norm(delta)
    slope = args.alpha * args.epsilon / (1 - args.epsilon)
    constant = args.epsilon * (1 - args.alpha / (1 - args.epsilon))
    if norm_delta <= args.epsilon:
        return delta * (1-args.alpha)
    else:

        return delta * (norm_delta * slope + constant)


def linf_constraint(grad):
    """
    Constrain delta to l_infty ball
    """
    return torch.sign(grad)

def reinforce(log_prob, f, **kwargs):
    """
    Based on
    https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
    """
    policy_loss = (-log_prob) * f.detach()
    return policy_loss

def reinforce_new(log_prob, f, **kwargs):
    policy_loss = (-log_prob) * f.detach()
    d_loss = torch.autograd.grad([policy_loss.mean()], [log_prob],
                                        create_graph=True,retain_graph=True)[0]
    return d_loss.detach()

def lax_black(log_prob, f, f_cv, param, cv, cv_opt):
    """
    Returns policy loss equivalent to:
    (f(x) - c(x))*grad(log(policy)) + grad(c(x))
    The l_infty constraint should appear elsewhere
    Args:
        f: unknown function
        f_cv: control variate

    Checkout https://github.com/duvenaud/relax/blob/master/pytorch_toy.py
    """
    log_prob = (-1)*log_prob
    # Gradients of log_prob wrt to Gaussian params
    d_params_probs = torch.autograd.grad([log_prob.sum()],param,
                                    create_graph=True, retain_graph=True)

    # Gradients of cont var wrt to Gaussian params
    d_params_cont = torch.autograd.grad([f_cv], param,
                                    create_graph=True, retain_graph=True)


    # Difference between f and control variate
    ac = f - f_cv

    # Scale gradient, negative cv gradient since reward
    d_log_prob = []
    for p, c in zip(d_params_probs, d_params_cont):
        d_log_prob.append(ac*p - c)

    # Backprop param gradients
    for p, g in zip(param, d_log_prob):
        p.backward(g.detach(), retain_graph=True)

    # Optimize control variate to minimize variance
    var = sum([v**2 for v in d_log_prob])
    d_var = torch.autograd.grad([var.mean()], cv.parameters(),
                                    create_graph=True, retain_graph=True)

    # Set gradients to control variate params
    for p, g in zip(cv.parameters(), d_var):
        p.grad = g

    cv_opt.step()

    return None

def soft_reward(pred, targ):
    """
    BlackBox adversarial soft reward. Highest reward when `pred` for `targ`
    class is low. Use this reward to reinforce action gradients.

    Computed as: 1 - (targ pred).
    Args:
        pred: model log prediction vector, to be normalized below
        targ: true class integer, we want to decrease probability of this class
    """
    # pred = F.softmax(pred, dim=1)
    pred_prob = torch.exp(pred)
    gather = pred[:,targ] # gather target predictions
    ones = torch.ones_like(gather)
    r = ones - gather
    r = r.mean()

    return r

def hard_reward(pred, targ):
    """
    BlackBox adversarial 0/1 reward.
    1 if predict something other than target, 0 if predict target. This reward
    should make it much harder to optimize a black box attacker.
    """
    pred = F.softmax(pred, dim=1)
    out = pred.max(1, keepdim=True)[1] # get the index of the max log-prob
