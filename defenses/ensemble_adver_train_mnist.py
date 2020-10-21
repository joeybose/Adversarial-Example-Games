import torch
import torchvision
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
import os
import argparse
import numpy as np
import ipdb
import json
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from utils.utils import create_loaders, load_unk_model, test_classifier
from cnn_models import *
from cnn_models.mnist_ensemble_adv_train_models import *
from classifiers import load_one_classifier

# Code Taken from: https://github.com/cailk/ensemble-adv-training-pytorch

EVAL_FREQUENCY = 100

ARCHITECTURES = {
            'VGG16': (VGG, 50),
            'res18': (resnet.ResNet18, 500),
            'res18_adv': (resnet.ResNet18, 500),
            'res18_ens': (resnet.ResNet18, 500),
            'dense121': (densenet.densenet_cifar,  500),
            'dense121_adv': (densenet.densenet_cifar,  500),
            'dense121_ens': (densenet.densenet_cifar,  500),
            'googlenet': (googlenet.GoogLeNet, 500),
            'googlenet_adv': (googlenet.GoogLeNet, 500),
            'googlenet_ens': (googlenet.GoogLeNet, 500),
            'lenet': (LeNet, 250),
            'wide_resnet': (wide_resnet.Wide_ResNet, None),
            'wide_resnet_adv': (wide_resnet.Wide_ResNet, None),
            'wide_resnet_ens': (wide_resnet.Wide_ResNet, None)
	    }

def gen_adv_loss(logits, labels, loss='logloss', mean=False):
    '''
    Generate the loss function
    '''
    if loss == 'training':
        # use the model's output instead of the true labels to avoid
        # label leaking at training time
        labels = logits.max(1)[1]
        if mean:
            out = F.cross_entropy(logits, labels, reduction='mean')
        else:
            out = F.cross_entropy(logits, labels, reduction='sum')
    elif loss == 'logloss':
        if mean:
            out = F.cross_entropy(logits, labels, reduction='mean')
        else:
            out = F.cross_entropy(logits, labels, reduction='sum')
    else:
        raise ValueError('Unknown loss: {}'.format(loss))
    return out

def gen_grad(x, model, y, loss='logloss'):
    '''
    Generate the gradient of the loss function.
    '''
    model.eval()
    x.requires_grad = True

    # Define gradient of loss wrt input
    logits = model(x)
    adv_loss = gen_adv_loss(logits, y, loss)
    model.zero_grad()
    adv_loss.backward()
    grad = x.grad.data
    return grad

def symbolic_fgs(data, grad, eps=0.3, clipping=True):
    '''
    FGSM attack.
    '''
    # signed gradien
    normed_grad = grad.detach().sign()

    # Multiply by constant epsilon
    scaled_grad = eps * normed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = data.detach() + scaled_grad
    if clipping:
        adv_x = torch.clamp(adv_x, 0, 1)
    return adv_x

def iter_fgs(model, data, labels, steps, eps):
    '''
    I-FGSM attack.
    '''
    adv_x = data

    # iteratively apply the FGSM with small step size
    for i in range(steps):
        grad = gen_grad(adv_x, model, labels)
        adv_x = symbolic_fgs(adv_x, grad, eps)
    return adv_x

def train_ens(epoch, batch_idx, model, data, labels, optimizer, x_advs=None,
              opt_step=True):
    model.train()
    optimizer.zero_grad()
    # Generate cross-entropy loss for training
    logits = model(data)
    preds = logits.max(1)[1]
    loss1 = gen_adv_loss(logits, labels, mean=True)

    # add adversarial training loss
    if x_advs is not None:

        # choose source of adversarial examples at random
        # (for ensemble adversarial training)
        idx = np.random.randint(len(x_advs))
        logits_adv = model(x_advs[idx])
        loss2 = gen_adv_loss(logits_adv, labels, mean=True)
        loss = 0.5 * (loss1 + loss2)
    else:
        loss2 = torch.zeros(loss1.size())
        loss = loss1

    if opt_step:
        loss.backward()
        optimizer.step()

    if batch_idx % EVAL_FREQUENCY == 0:
        print('Step: {}(epoch: {})\tLoss: {:.6f}<=({:.6f}, {:.6f})\tError: {:.2f}%'.format(
            batch_idx, epoch+1, loss.item(), loss1.item(), loss2.item(), error_rate(preds, labels)
        ))
    return loss

def test(model, data, labels):
    model.eval()
    correct = 0
    logits = model(data)

    # Prediction for the test set
    preds = logits.max(1)[1]
    correct += preds.eq(labels).sum().item()
    return correct

def error_rate(preds, labels):
    '''
    Run the error rate
    '''
    assert preds.size() == labels.size()
    return 100.0 - (100.0 * preds.eq(labels).sum().item()) / preds.size(0)

def get_model_type(model_name):
    model_type = {
        'modelA': 0, 'modelA_adv': 0, 'modelA_ens': 0, 'modelA_ens1': 0,
        'modelB': 1, 'modelB_adv': 1, 'modelB_ens': 1, 'modelB_ens1': 1,
        'modelC': 2, 'modelC_adv': 2, 'modelC_ens': 2, 'modelC_ens1': 2,
        'modelD': 3, 'modelD_adv': 3, 'modelD_ens': 3, 'modelD_ens1': 3,
        'res18': 4, 'res18_adv': 4, 'res18_ens': 4,
        'googlenet': 5, 'googlenet_adv': 5, 'googlenet_ens': 5,
        'wide_resnet': 6, 'wide_resnet_adv': 6, 'wide_resnet_ens': 6,
        'dense121': 7, 'dense121_adv': 7, 'dense121_ens': 7,
    }
    if model_name not in model_type.keys():
        raise ValueError('Unknown model: {}'.format(model_name))
    return model_type[model_name]

def main(args):

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.cuda else 'cpu')
    train_loader, test_loader = create_loaders(args, root='../data')
    eps = args.epsilon

    # if src_models is not None, we train on adversarial examples that come
    # from multiple models
    if args.train_adv:
        adv_model_names = args.adv_models
        adv_models = [None] * len(adv_model_names)
        for i in range(len(adv_model_names)):
            type = get_model_type(adv_model_names[i])
            if args.dataset == 'cifar':
                adv_models[i] = load_one_classifier(args,
                        load_archs=[adv_model_names[i]]).to(device)
                acc = test_classifier(args, adv_models[i], args.dev, test_loader, epoch=0, logger=None)
                print("Dataset: %s Model: %s Test set acc: %f" %(args.dataset,
                    adv_model_names[i], acc))
                adv_models[i] = nn.DataParallel(adv_models[i])
            else:
                adv_models[i] = load_model(args, adv_model_names[i], type=type).to(device)

    if args.dataset == 'cifar':
        init_func, _ = ARCHITECTURES[args.model]
        model = init_func().to(args.dev)
        if "wide_resnet" in args.model:
            model.apply(wide_resnet.conv_init)
        model = nn.DataParallel(model)
    else:
        model = model_mnist(type=args.type).to(device)

    optimizer = optim.Adam(model.parameters())

    # Train model
    if args.train_adv:
        x_advs = [None] * (len(adv_models) + 1)
        for epoch in range(args.epochs):
            for batch_idx, (data, labels) in enumerate(train_loader):
                data, labels = data.to(device), labels.to(device)
                for i, m in enumerate(adv_models + [model]):
                    grad = gen_grad(data, m, labels, loss='training')
                    x_advs[i] = symbolic_fgs(data, grad, eps=eps)
                loss_model = train_ens(epoch, batch_idx, model, data, labels, optimizer, x_advs=x_advs)
    else:
        for epoch in range(int(args.epochs / 2)):
            for batch_idx, (data, labels) in enumerate(train_loader):
                data, labels = data.to(device), labels.to(device)
                loss_model = train_ens(epoch, batch_idx, model, data, labels, optimizer)

    # Finally print the result
    correct = 0
    with torch.no_grad():
        for (data, labels) in test_loader:
            data, labels = data.to(device), labels.to(device)
            correct += test(model, data, labels)
    test_error = 100. - 100. * correct / len(test_loader.dataset)
    print('Test Set Error Rate: {:.2f}%'.format(test_error))
    path = os.path.join(args.dir_test_models, "pretrained_classifiers",
                        args.dataset, "ensemble_adv_trained", args.model)
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    torch.save(model.state_dict(), path + args.namestr + '.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial Training MNIST model')
    parser.add_argument('--model', help='path to model')
    parser.add_argument('--adv_models', nargs='*', help='path to adv model(s)')
    parser.add_argument('--type', type=int, default=0, help='Model type (default: 0)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--disable_cuda', action='store_true', default=False, help='Disable CUDA (default: False)')
    parser.add_argument('--epochs', type=int, default=12, help='Number of epochs (default: 12)')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--train_adv', default=False, action='store_true',
                        help='Whether to train normally or Adversarially')
    parser.add_argument("--wandb", action="store_true", default=False, help='Use wandb for logging')
    parser.add_argument('--batch_size', type=int, default=256, metavar='S')
    parser.add_argument('--test_batch_size', type=int, default=512, metavar='S')
    parser.add_argument('--train_set', default='train',
                        choices=['train_and_test','test','train'],
                        help='add the test set in the training set')
    parser.add_argument('--attack_ball', type=str, default="Linf",
                        choices= ['L2','Linf'])
    parser.add_argument('--architecture', default="VGG16",
                        help="The architecture we want to attack on CIFAR.")
    parser.add_argument('--dir_test_models', type=str, default="../",
                        help="The path to the directory containing the classifier models for evaluation.")
    parser.add_argument('--epsilon', type=float, default=0.1, metavar='M',
                        help='Epsilon for Delta (default: 0.1)')
    parser.add_argument('--train_with_critic_path', type=str, default=None,
                        help='Train generator with saved critic model')
    parser.add_argument('--namestr', type=str, default='1', \
            help='additional info in output filename to describe experiments')

    args = parser.parse_args()
    args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    main(args)
