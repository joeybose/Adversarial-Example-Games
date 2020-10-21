from advertorch.context import ctx_noparamgrad_and_eval
from torch import autograd
import sys
import os
import random
sys.path.insert(0, "..")  # Adds higher directory to python modules path.
from attacks.autopgd_pt import APGDAttack
from attacks.fab_pt import FABAttack
from attacks.square_pt import SquareAttack
from attacks.autopgd_pt import APGDAttack_targeted
from classifiers import load_all_classifiers, load_list_classifiers, load_dict_classifiers
from cnn_models import LeNet as Net
from cnn_models import ResNet18
from cnn_models.mnist_ensemble_adv_train_models import *
from defenses.ensemble_adver_train_mnist import *
from eval import baseline_transfer, baseline_eval_classifier
import numpy as np
import torch

def load_attacker(args):
    raise NotImplementedError

def data_and_model_setup(args, di_attack=False, no_box_attack=False):
    if args.dataset == 'cifar':
        args.nc, args.h, args.w = 3, 32, 32
        try:
            print(f'\nLoading training architecture(s): {args.source_arch}\n')
            model, l_test_classif_paths = load_all_classifiers(args, load_archs=[args.source_arch])
        except:
            pass
        model_type = args.source_arch
        adv_models = None
        # Check if we need to Load Target model paths
        if args.target_arch is not None:
            #_, l_test_classif_paths = load_all_classifiers(args,
            #        load_archs=args.target_arch, load_train=False)
            model_type = args.target_arch
            l_test_classif_paths = []
        if args.ensemble_adv_trained:
            adv_model_names = args.adv_models
            l_test_classif_paths = []
            # adv_models = [None] * len(adv_model_names)
            path = os.path.join(args.dir_test_models, "pretrained_classifiers",
                                args.dataset, "ensemble_adv_trained",
                                args.adv_models[0])
            all_paths = []
            if os.path.exists(path):
                list_dir = os.listdir(path)
                if len(list_dir) > 0:
                    num_test_model = len(list_dir)
                for i in range(num_test_model):
                    filename = os.path.join(path, list_dir[i])
                    all_paths.append(filename)
            l_test_classif_paths = all_paths
            model_type = adv_model_names[0]
            adv_models = None
            print("Adv Models will be loaded at Test time")
            print("Transferring attack to the following models")
            print(*all_paths, sep = "\n")
            try:
                args.ens_adv_models = args.adv_models[1:]
            except:
                pass
        elif args.source_arch == 'adv':
            list_classifiers = load_list_classifiers(args, "madry_challenge_models")
            model = list_classifiers['madry_challenge']['natural']
            adv_models = [list_classifiers['madry_challenge']['secret']]
            model_type = 'PGD Adversarial Training'
            path = os.path.join(args.dir_test_models, "madry_challenge_models",
                                args.dataset, "adv_trained")
            l_test_classif_paths = [path]
    elif args.dataset == 'mnist':
        if args.source_arch == 'natural':
            model, l_test_classif_paths = load_all_classifiers(args, load_archs=["natural"])
            model_type = 'natural'
            adv_models = None
        elif args.source_arch == 'adv':
            list_classifiers = load_list_classifiers(args, "madry_challenge_models")
            model = list_classifiers['madry_challenge']['natural']
            adv_models = [list_classifiers['madry_challenge']['secret']]
            model_type = 'PGD Adversarial Training'
            path = os.path.join(args.dir_test_models, "madry_challenge_models",
                                args.dataset, "adv_trained")
            l_test_classif_paths = [path]
        elif args.source_arch == 'ens_adv' or args.ensemble_adv_trained:
            adv_model_names = args.adv_models
            adv_models = [None] * len(adv_model_names)
            # for i in range(len(adv_model_names)):
                # type = get_model_type(adv_model_names[i])
                # adv_models[i] = load_model(args, adv_model_names[i], type=type).to(args.dev)
            # try:
                # args.ens_adv_models = adv_models[1:]
            # except:
                # pass
            path = os.path.join(args.dir_test_models, "pretrained_classifiers",
                                args.dataset, "ensemble_adv_trained",
                                args.adv_models[0])
            all_paths = []
            # Reset Type to model type
            if os.path.exists(path):
                list_dir = os.listdir(path)
                if len(list_dir) > 0:
                    num_test_model = len(list_dir)
                for i in range(num_test_model):
                    filename = os.path.join(path, list_dir[i])
                    all_paths.append(filename)
            if di_attack:
                path = os.path.join(args.dir_test_models,
                                    "pretrained_classifiers", args.dataset,
                                    "natural")
                list_dir = os.listdir(path)
                random.shuffle(list_dir)
                filename = os.path.join(path, list_dir[0])
                model = Net(args.nc, args.h, args.w).to(args.dev)
                model.load_state_dict(torch.load(filename))
            elif no_box_attack:
                args.type = get_model_type(args.model_name)
                model = load_model(args, args.model_name, type=args.type)
            else:
                args.type = get_model_type(args.model)
                model = load_model(args, args.model, type=args.type)
            l_test_classif_paths = all_paths
            model_type = 'Ensemble Adversarial'
            adv_models = None
            print("Adv Models will be loaded at Test time")
            print("Transferring attack to the following models")
            print(*all_paths, sep = "\n")
    return model, adv_models, l_test_classif_paths, model_type

def load_data(args, test_loader):
    l = [x.unsqueeze(0) for (x, y) in test_loader.dataset]
    x_test = torch.cat(l, 0)[:args.batch_size].to(args.dev)
    l = [y for (x, y) in test_loader.dataset]
    y_test = torch.Tensor(l).long()[:args.batch_size].to(args.dev)
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print("CUDA Device Count is %d, Error might happen. Use export CUDA_VISIBLE_DEVICES=0" %(device_count))
    return x_test, y_test


def eval(args, attacker, attack_name, eval_helpers, num_eval_samples=None):
    model, model_type, adv_models, l_test_classif_paths, test_loader = eval_helpers[:]
    advcorrect = 0
    with ctx_noparamgrad_and_eval(model):
        adv_complete_list, output_list, y_list = [], [], []
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            x_batch, y_batch = x_batch.to(args.dev), y_batch.to(args.dev)
            adv_imgs = attacker.perturb(x_batch,y_batch).detach()
            adv_complete_list.append(adv_imgs.cpu())
            output_list.append(model(adv_imgs).cpu())
            y_list.append(y_batch.cpu())
        output = torch.cat(output_list)
        adv_complete = torch.cat(adv_complete_list)
        y_test = torch.cat(y_list)
        pred = output.max(1, keepdim=True)[1]
        advcorrect += pred.eq(y_test.view_as(pred)).sum().item()
        fool_rate = 1 - advcorrect / float(len(test_loader.dataset))
        print('Test set base model fool rate: %f' %(fool_rate))
    if args.transfer:
        if num_eval_samples is not None:
            adv_complete = adv_complete[:num_eval_samples]
            y_test = y_test[:num_eval_samples]

        adv_img_list = torch.utils.data.TensorDataset(adv_complete, y_test)
        adv_img_list = torch.utils.data.DataLoader(adv_img_list, batch_size=args.test_batch_size)
        # Free memory
        del model
        torch.cuda.empty_cache()
        fool_rate = []
        if args.target_arch is not None:
            model_type = args.target_arch
        if not isinstance(model_type, list):
            model_type = [model_type]

        for name in model_type:
            fool_rate.append(baseline_transfer(args, attacker, attack_name, name,
                            adv_img_list, l_test_classif_paths, adv_models))
        return np.mean(fool_rate)

    return fool_rate
