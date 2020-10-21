import os
import sys
import warnings
import random
from utils.utils import load_unk_model, ARCHITECTURES
# TODO: the default list should come from args not here
DEFAULT_LIST_CLASSIFIERS = ["pretrained_classifiers", "madry_challenge_models"]
import numpy as np
import ipdb
sys.path.append("..")  # Adds higher directory to python modules path.
from cnn_models.mnist_ensemble_adv_train_models import *

def load_dict_classifiers(args, classifiers_name=DEFAULT_LIST_CLASSIFIERS):
    dict_classifier = load_list_classifiers(args, classifiers_name=classifiers_name)
    l_train, l_test = split_classif_list(args, dict_classifier)
    print(f'\nTraining on {l_train.keys()}\n')
    print(f'\nTesting on {l_test.keys()}\n')
    if args.ensemble_adv_trained:
        l_test = load_ens_adv_model(args)

    return l_train, l_test

def load_ens_adv_model(args):
    list_classifiers = {}
    adv_models = {}
    from defenses.ensemble_adver_train_mnist import get_model_type
    if args.dataset=="mnist":
        adv_model_names = args.adv_models
        for i in range(len(adv_model_names)):
            type = get_model_type(adv_model_names[i])
            adv_models[i] = load_model(args, adv_model_names[i], type=type).to(args.dev)

    elif args.dataset =='cifar':
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
    return adv_models

def load_list_classifiers(args, classifiers_name=DEFAULT_LIST_CLASSIFIERS):
    list_classifiers = {}
    if "madry_challenge_models" in classifiers_name:
        list_classifiers["madry_challenge"] = {}
        from .madry_challenge.madry_et_al_utils import get_madry_et_al_tf_model
        for model_type in ['natural', 'adv_trained', 'secret']:
            path = os.path.join(args.dir_test_models, "madry_challenge_models", args.dataset, model_type)
            # A bit hacky, check if Madry model already loaded
            try:
                key_n = "madry_challenge/" + model_type
                list_classifiers["madry_challenge"][model_type]= args.madry_model[path]
            except:
                if os.path.exists(path):
                    try:
                        model = get_madry_et_al_tf_model(args.dataset, path)
                        args.madry_model = {path: model}
                        list_classifiers["madry_challenge"][model_type] = model
                    except Exception as e:
                        msg = f"WARNING: Couldn't load Madry challenge. {e}"
                        warnings.warn(msg)
                        sys.exit(1)
                else:
                    msg = f"WARNING: Couldn't load Madry challenge, {path} not found."
                    warnings.warn(msg)

    if "pretrained_classifiers" in classifiers_name:
        list_classifiers["pretrained_classifiers"] = {}
        # for model_type in ['natural', 'adv_trained', 'ensemble_adv_trained']:
        for model_type in ['natural', 'adv_trained']:
            list_classifiers["pretrained_classifiers"][model_type] = {}
            path = os.path.join(args.dir_test_models, "pretrained_classifiers", args.dataset,model_type)
            if os.path.exists(path):
                list_dir = os.listdir(path)
                if len(list_dir) > 0:
                    num_test_model = args.max_test_model
                    if (num_test_model is None) or (num_test_model > len(list_dir)):
                        num_test_model = len(list_dir)
                for i in range(num_test_model):
                    filename = os.path.join(path, list_dir[i])
                    print("Loaded: %s" % (filename))
                    list_classifiers["pretrained_classifiers"][model_type][i] = load_unk_model(args, filename)
            else:
                warnings.warn(f"WARNING: Couldn't load any pretrained classifiers, {path} doesn't exists or is empty.")
    return list_classifiers

def load_all_classifiers(args, load_archs=None, split=0, load_train=True):
    train_classifiers = {}
    test_classifiers = {}
    all_paths = []
    if load_archs is None:
        archs = ARCHITECTURES
    else:
        archs = load_archs

    for model_type in archs:
        if args.split is None:
            path = os.path.join(args.dir_test_models, "pretrained_classifiers", args.dataset, model_type)
        else:
            path = os.path.join(args.dir_test_models, "split_classifiers",
                    args.dataset, model_type)
        if os.path.exists(path):
            list_dir = os.listdir(path)
            if len(list_dir) > 0:
                num_test_model = len(list_dir)
            for i in range(num_test_model):
                if args.split is None:
                    filename = os.path.join(path, list_dir[i])
                else:
                    filename = os.path.join(path, 'split_' + str(i) +
                    '/model_0.pt')
                all_paths.append(filename)
        else:
            warnings.warn(f"WARNING: Couldn't load any pretrained classifiers, {path} doesn't exists or is empty.")

    # random.shuffle(all_paths)
    if args.split is not None:
        train_path = all_paths.pop(args.split)
        test_paths = all_paths
    else:
        train_path = all_paths[0]
        test_paths = all_paths[1:]
    train_model_name = train_path.split('/')[-1]
    if load_train:
        train_classifiers = load_unk_model(args, train_path, name=model_type)
    else:
        train_classifiers = None
    print(f'\nLoading {train_path} {train_model_name}\n')

    return train_classifiers, test_paths

def load_one_classifier(args, load_archs=None):
    train_classifiers = {}
    all_paths = []
    if load_archs is None:
        archs = ARCHITECTURES
    else:
        archs = load_archs

    for model_type in archs:
        path = os.path.join(args.dir_test_models, "pretrained_classifiers", args.dataset, model_type)
        if os.path.exists(path):
            list_dir = os.listdir(path)
            if len(list_dir) > 0:
                    num_test_model = len(list_dir)
            for i in range(num_test_model):
                filename = os.path.join(path, list_dir[i])
                all_paths.append(filename)
        else:
            warnings.warn(f"WARNING: Couldn't load any pretrained classifiers, {path} doesn't exists or is empty.")

    random.shuffle(all_paths)
    train_path = all_paths[0]
    train_model_name = train_path.split('/')[-1]
    train_classifiers = load_unk_model(args, train_path, name=model_type)
    print(f'\nTraining on {train_model_name}\n')

    return train_classifiers

def load_cifar_classifiers(args, load_archs=None):
    train_classifiers = {}
    test_classifiers = {}
    if load_archs is None:
        archs = ARCHITECTURES
    else:
        archs = load_archs

    for model_type in archs:
        path = os.path.join(args.dir_test_models, "pretrained_classifiers", args.dataset, model_type)
        if os.path.exists(path):
            list_dir = os.listdir(path)
            if len(list_dir) > 0:
                num_test_model = args.max_test_model
                if (num_test_model is None) or (num_test_model > len(list_dir)):
                    num_test_model = len(list_dir) - 1
            for i in range(num_test_model):
                filename = os.path.join(path, list_dir[i])
                test_classifiers[f"pretrained_classifiers\{model_type}_{i}"] = load_unk_model(args, filename, name=model_type).cpu()

            filename = os.path.join(path, list_dir[num_test_model])
            if args.architecture == model_type:
                train_classifiers[f"pretrained_classifiers\{model_type}_{num_test_model}"] = load_unk_model(args, filename, name=model_type)
        else:
            warnings.warn(f"WARNING: Couldn't load any pretrained classifiers, {path} doesn't exists or is empty.")

    print(f'\nTraining on {train_classifiers.keys()}\n')
    print(f'\nTesting on {test_classifiers.keys()}\n')

    return train_classifiers, test_classifiers


def split_classif_list(args, dict_classif):
    train_dict = {}
    test_dict = {}
    if args.train_on_list:
        thresh = .5
    else:
        thresh = 1.
    if 'madry_challenge' in dict_classif.keys():
        for name,model in dict_classif['madry_challenge'].items():
            if name == "adv_trained" and args.train_on_list:
                print(f'Not using Madry {name}')
                # train_dict[f"madry_challenge_{name}"] = model
            else:
                test_dict[f"madry_challenge_{name}"] = model
    for model_type in ['natural', 'adv_trained']:
        num_pretrained = len(dict_classif['pretrained_classifiers'][model_type])
        for i,(name, classif) in enumerate(dict_classif['pretrained_classifiers'][model_type].items()):
            key = f"pretrained_{model_type}_{name}"
            if i >= np.ceil(num_pretrained * thresh):
                train_dict[key] = classif
            else:
                test_dict[key] = classif.cpu()
    return train_dict, test_dict
