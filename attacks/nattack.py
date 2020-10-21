import robustml
import wandb
import json
import os
import sys
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as tfs
import torchvision.datasets as dst
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.autograd as autograd
import pickle
import ipdb
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from utils.utils import create_loaders, load_mnist, load_cifar, test_classifier
from cnn_models.vgg_robustnet import VGG_noisy
from cnn_models import LeNet as Net
from classifiers import load_all_classifiers, load_list_classifiers, load_dict_classifiers
from eval import baseline_transfer, baseline_eval_classifier

def softmax(x):
    return np.divide(np.exp(x),np.sum(np.exp(x),-1,keepdims=True))

def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    return (np.log((1 + x) / (1 - x))) * 0.5

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and m.affine:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class NAttack():
    def __init__(self, args, model, dataset='cifar', n_iter=500, eps=0.031,
                 norm='Linf', end=100, seed=0, device='cuda'):
        self.args = args
        self.model = model
        self.dataset = dataset
        self.n_iter = n_iter
        self.eps = eps
        self.seed = seed
        self.device = device
        self.test_loss = 0
        self.total = 0
        self.small_val = 1e-30

        self.npop = 300     # population size
        self.sigma = 0.1    # noise standard deviation
        self.alpha = 0.02  # learning rate
        self.boxmin = 0
        self.boxmax = 1
        self.boxplus = (self.boxmin + self.boxmax) / 2.
        self.boxmul = (self.boxmax - self.boxmin) / 2.
        self.start = 0
        self.end = end
        if self.dataset == 'cifar':
            self.nc, self.h, self.w = 3,32,32
            with torch.no_grad():
                self.means = torch.from_numpy(np.array([0.4914, 0.4822, 0.4465]).reshape([1,
                    3, 1, 1]).astype('float32')).to(device)
                self.stds = torch.from_numpy(np.array([0.2023, 0.1994, 0.2010]).reshape([1,
                    3, 1, 1]).astype('float32')).to(device)
        elif self.dataset == 'mnist':
            self.nc, self.h, self.w = 1,28,28
            with torch.no_grad():
                # self.means = torch.from_numpy(np.array([0.1307]).reshape([1,
                    # 1, 1, 1]).astype('float32')).to(device)
                # self.stds = torch.from_numpy(np.array([0.3081]).reshape([1,
                    # 1, 1, 1]).astype('float32')).to(device)
                self.means = torch.from_numpy(np.array([0.]).reshape([1,
                    1, 1, 1]).astype('float32')).to(device)
                self.stds = torch.from_numpy(np.array([1.]).reshape([1,
                    1, 1, 1]).astype('float32')).to(device)
        else:
            raise NotImplementedError

    def perturb(self, provider):
        successlist = []
        printlist = []
        faillist = []
        adv_img_list = []
        totalImages = 0
        succImages = 0
        for i in range(self.start, self.end):
            success = False
            print('evaluating %d of [%d, %d)' % (i, self.start, self.end))
            inputs, targets = provider[i]
            if inputs.shape[0] != self.nc:
                if len(inputs.shape) < 3:
                    inputs = np.expand_dims(inputs, axis=-1)
                with torch.no_grad():
                    input_var = torch.from_numpy(inputs.transpose(2, 0,
                        1)).to(self.device)
            else:
                input_var = inputs.to(self.device)
                input_var.requires_grad = False
            modify = np.random.randn(1,self.nc,self.h,self.w) * 0.001
            logits = self.model((input_var-self.means)/self.stds).data.cpu().numpy()

            probs = softmax(logits)
            if np.argmax(probs[0]) != targets:
                print('skip the wrong example ', i)
                continue

            totalImages += 1

            for runstep in range(self.n_iter):
                Nsample = np.random.randn(self.npop, self.nc,self.h,self.w)
                modify_try = modify.repeat(self.npop,0) + self.sigma*Nsample
                if inputs.shape[0] != self.nc:
                    newimg = torch_arctanh((inputs-self.boxplus) / self.boxmul).transpose(2,0,1)
                else:
                    newimg = torch_arctanh((inputs-self.boxplus) / self.boxmul)

                inputimg = np.tanh(newimg+modify_try) * self.boxmul + self.boxplus
                if runstep % 10 == 0:
                    realinputimg = np.tanh(newimg+modify) * self.boxmul + self.boxplus
                    realdist = realinputimg - (np.tanh(newimg) * self.boxmul + self.boxplus)
                    realclipdist = np.clip(realdist, -self.eps, self.eps)
                    realclipinput = realclipdist + (np.tanh(newimg) *
                            self.boxmul + self.boxplus)
                    l2real =  np.sum((realclipinput - (np.tanh(newimg) * self.boxmul + self.boxplus))**2)**0.5
                    with torch.no_grad():
                        input_var = torch.from_numpy(realclipinput.astype('float32')).to(self.device)

                    adv_img = (input_var- self.means)/self.stds
                    outputsreal = self.model((input_var- self.means)/self.stds).data.cpu().numpy()[0]
                    outputsreal = softmax(outputsreal)
                    # print('probs ', np.sort(outputsreal)[-1:-6:-1])
                    # print('target label ', np.argsort(outputsreal)[-1:-6:-1])
                    # print('negative_probs ', np.sort(outputsreal)[0:3:1])

                    if (np.argmax(outputsreal) != targets) and (np.abs(realclipdist).max() <= self.eps):
                        succImages += 1
                        success = True
                        print('clipimage succImages: '+str(succImages)+'  totalImages: '+str(totalImages))
                        # print('lirealsucc: '+str(realclipdist.max()))
                        successlist.append(i)
                        printlist.append(runstep)
                        adv_img_list.append([adv_img, targets])
                        break

                dist = inputimg - (np.tanh(newimg) * self.boxmul + self.boxplus)
                clipdist = np.clip(dist, -self.eps, self.eps)
                clipinput = (clipdist + (np.tanh(newimg) * self.boxmul +
                                         self.boxplus)).reshape(self.npop,self.nc,self.h,self.w)
                target_onehot =  np.zeros((1,10))
                target_onehot[0][targets]=1.
                # clipinput = np.squeeze(clipinput)
                clipinput = np.asarray(clipinput, dtype='float32')
                with torch.no_grad():
                    input_var = torch.from_numpy(clipinput).to(self.device)
                outputs = self.model((input_var-self.means)/self.stds).data.cpu().numpy()
                outputs = softmax(outputs)
                target_onehot = target_onehot.repeat(self.npop,0)
                real = np.log((target_onehot * outputs).sum(1)+self.small_val)
                other = np.log(((1. - target_onehot) * outputs - target_onehot
                    * 10000.).max(1)[0]+self.small_val)

                loss1 = np.clip(real - other, 0.,1000)
                Reward = 0.5 * loss1

                Reward = -Reward

                A = (Reward - np.mean(Reward)) / (np.std(Reward)+1e-7)

                modify = modify + (self.alpha/(self.npop*self.sigma)) * ((np.dot(Nsample.reshape(self.npop,-1).T,
                         A)).reshape(self.nc,self.h,self.w))
            if not success:
                faillist.append(i)
                adv_img_list.append([adv_img, targets])
                print('failed:', faillist)
            else:
                print('successed:', successlist)

        print(faillist)
        success_rate = succImages/float(totalImages)
        print('run steps: ', printlist)
        np.savez('runstep',printlist)
        print('succ rate', success_rate)
        return success_rate, adv_img_list

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='../../Nattack/cifar10_data/test_batch',
            help='path to the test_batch file from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--n_iter', type=int, default=500)
    parser.add_argument('--query_step', type=int, default=1)
    parser.add_argument('--transfer', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument("--wandb", action="store_true", default=False, help='Use wandb for logging')
    parser.add_argument('--noise', type=float, default=0.3)
    parser.add_argument('--epsilon', type=float, default=0.031)
    parser.add_argument('--batch_size', type=int, default=256, metavar='S')
    parser.add_argument('--test_batch_size', type=int, default=512, metavar='S')
    parser.add_argument('--train_set', default='test',
                        choices=['train_and_test','test','train'],
                        help='add the test set in the training set')
    parser.add_argument('--modelIn', type=str, default='../../Nattack/all_models/robustnet/noise_0.3.pth')
    parser.add_argument('--robust_model_path', type=str,
                        default="../madry_challenge_models/mnist/adv_trained/mnist_lenet5_advtrained.pt")
    parser.add_argument('--source_arch', default="res18",
                        help="The architecture we want to attack on CIFAR.")
    parser.add_argument('--target_arch', default=None,
                        help="The architecture we want to blackbox transfer to on CIFAR.")
    parser.add_argument('--dir_test_models', type=str,
                        default="../",
                        help="The path to the directory containing the classifier models for evaluation.")
    parser.add_argument("--max_test_model", type=int, default=2,
                    help="The maximum number of pretrained classifiers to use for testing.")
    parser.add_argument('--train_on_madry', default=False, action='store_true',
                        help='Train using Madry tf grad')
    parser.add_argument('--train_on_list', default=False, action='store_true',
                        help='train on a list of classifiers')
    parser.add_argument('--train_with_critic_path', type=str, default=None,
                        help='Train generator with saved critic model')
    parser.add_argument('--namestr', type=str, default='NoBox', \
            help='additional info in output filename to describe experiments')
    args = parser.parse_args()
    args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if os.path.isfile("../settings.json"):
        with open('../settings.json') as f:
            data = json.load(f)
        args.wandb_apikey = data.get("wandbapikey")

    if args.wandb:
        os.environ['WANDB_API_KEY'] = args.wandb_apikey
        wandb.init(project='NoBox-sweeps', name='NAttack-{}'.format(args.dataset))

    adv_models = None
    if args.dataset == 'cifar':
        args.nc, args.h, args.w = 3, 32, 32
        provider = robustml.provider.CIFAR10(args.data_path)
        model, l_test_classif_paths = load_all_classifiers(args, load_archs=[args.source_arch])
        model_type = args.source_arch
        if args.target_arch is not None:
            model_target, l_test_classif_paths = load_all_classifiers(args, load_archs=[args.target_arch])
            model_type = args.target_arch
            del model_target
            torch.cuda.empty_cache()
    elif args.dataset == 'mnist':
        mnist_data_path = '../data/MNIST/raw/t10k-images-idx3-ubyte.gz'
        mnist_label_path = '../data/MNIST/raw/t10k-labels-idx1-ubyte.gz'
        args.nc, args.h, args.w = 1, 28, 28
        provider = robustml.provider.MNIST(mnist_data_path, mnist_label_path)
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
    train_loader, test_loader = create_loaders(args, root='../data')
    test_classifier(args, model, args.dev, test_loader, 1, logger=None)
    print("Testing on %d Test Classifiers" %(len(l_test_classif_paths)))

    if not args.sweep:
        attacker = NAttack(args, model, dataset=args.dataset, n_iter=args.n_iter,
                           eps=args.epsilon, end=args.end)
        fool_rate, adv_img_list = attacker.perturb(provider)
        if args.wandb:
            wandb.log({"Fool Rate": fool_rate,
                       "queries": args.n_iter
                       })
        if args.transfer:
            baseline_transfer(args, attacker, "NAttack", model_type,
                              adv_img_list, l_test_classif_paths, adv_models)
    else:
        for n_iter in range(0, args.n_iter, args.query_step):
            attacker = NAttack(args, model, dataset=args.dataset, n_iter=n_iter,
                               eps=args.epsilon, end=args.end)
            fool_rate = attacker.perturb(provider)
            if args.wandb:
                wandb.log({"Fool Rate": fool_rate,
                           "queries": n_iter
                           })
            if args.transfer:
                baseline_transfer(args, attacker, "NAttack", model_type,
                                  adv_img_list, l_test_classif_paths,
                                  adv_models)

if __name__ == '__main__':
    main()
