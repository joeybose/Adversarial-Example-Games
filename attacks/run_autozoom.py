import argparse
import glob
import json
import os
import os.path as osp
import random
import glog as log
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.autograd as autograd
from autozoom_attack import ZOO, ZOO_AE, AutoZOOM_BiLIN, AutoZOOM_AE
from nattack import weights_init
from codec import Codec
# from autozoom_dataset.dataset_loader_maker import DataLoaderMaker
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from utils.utils import *
from cnn_models.vgg_robustnet import VGG_noisy
from cnn_models import LeNet as Net
from cnn_models import VGG
from classifiers import load_all_classifiers, load_list_classifiers, load_dict_classifiers
from eval import baseline_transfer, baseline_eval_classifier
import ipdb
PY_ROOT = "./"

IMAGE_SIZE = {"cifar":(32,32), "CIFAR-100":(32,32), "ImageNet":(224,224),
              "mnist":(28, 28), "FashionMNIST":(28,28), "SVHN":(32,32),
              "TinyImageNet": (64,64)}
IN_CHANNELS = {"mnist":1, "FashionMNIST":1, "cifar":3, "ImageNet":3, "CIFAR-100":3, "SVHN":3, "TinyImageNet":3}
CLASS_NUM = {"mnist":10,"FashionMNIST":10, "cifar":10, "CIFAR-100":100, "ImageNet":1000, "SVHN":10, "TinyImageNet":200}

class AutoZoomAttackFramework(object):

    def __init__(self, args, dataset_loader):
        self.dataset_loader = dataset_loader
        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images)
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.not_done_loss_all = torch.zeros_like(self.query_all)
        self.not_done_prob_all = torch.zeros_like(self.query_all)

    def cw_loss(self, logit, label, target=None):
        if target is not None:
            # targeted cw loss: logit_t - max_{i\neq t}logit_i
            _, argsort = logit.sort(dim=1, descending=True)
            target_is_max = argsort[:, 0].eq(target).long()
            second_max_index = target_is_max.long() * argsort[:, 1] + (1 - target_is_max).long() * argsort[:, 0]
            target_logit = logit[torch.arange(logit.shape[0]), target]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return target_logit - second_max_logit
        else:
            # untargeted cw loss: max_{i\neq y}logit_i - logit_y
            _, argsort = logit.sort(dim=1, descending=True)
            gt_is_max = argsort[:, 0].eq(label).long()
            second_max_index = gt_is_max.long() * argsort[:, 1] + (1 - gt_is_max).long() * argsort[:, 0]
            gt_logit = logit[torch.arange(logit.shape[0]), label]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return second_max_logit - gt_logit

    def make_adversarial_examples(self, batch_index, images, true_labels, args, attacker, target_model, codec):
        if args.attack_method == "zoo_ae" or args.attack_method == "autozoom_ae":
            # log ae info
            decode_img = codec(images)
            diff_img = (decode_img - images)
            diff_mse = torch.mean(diff_img.view(-1).pow(2)).item()
            print("[AE] MSE:{:.4f}".format(diff_mse))

        batch_size = 1
        selected = torch.arange(batch_index * batch_size,
                                (batch_index + 1) * batch_size)  # 选择这个batch的所有图片的index
        if args.attack_type == "targeted":

            if args.target_type == "random":
                with torch.no_grad():
                    logit = target_model(images)
                target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset], size=true_labels.size()).long().cuda()
                invalid_target_index = target_labels.eq(true_labels)
                while invalid_target_index.sum().item() > 0:
                    target_labels[invalid_target_index] = torch.randint(low=0, high=logit.shape[1],
                                                                        size=target_labels[
                                                                            invalid_target_index].shape).long().cuda()
                    invalid_target_index = target_labels.eq(true_labels)
            elif args.target_type == 'least_likely':
                with torch.no_grad():
                    logit = target_model(images)
                target_labels = logit.argmin(dim=1)
            else:
                target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
        else:
            target_labels = None
        print("Begin attack batch {}!".format(batch_index))
        with torch.no_grad():
            adv_images, stats_info = attacker.attack(images, true_labels, target_labels)
        query = stats_info["query"]
        correct = stats_info["correct"]
        not_done = stats_info["not_done"]
        success = stats_info["success"]
        success_query = stats_info["success_query"]
        not_done_prob = stats_info["not_done_prob"]
        adv_logit = stats_info["adv_logit"]
        adv_loss = self.cw_loss(adv_logit, true_labels, target_labels)
        not_done_loss = adv_loss * not_done
        return success, query, adv_images


    def attack_dataset_images(self, args, attacker, arch_name, target_model,
                              codec, l_test_classif_paths, adv_models, result_dump_path='.'):
        success_list, query_list, adv_img_list = [], [], []
        for batch_idx, data_tuple in enumerate(self.dataset_loader):
            print(batch_idx)
            if batch_idx > args.num_attack:
                break
            if args.dataset == "ImageNet":
                if args.input_size >= 299:
                    images, true_labels = data_tuple[1], data_tuple[2]
                else:
                    images, true_labels = data_tuple[0], data_tuple[2]
            else:
                images, true_labels = data_tuple[0], data_tuple[1]

            if images.size(-1) != args.input_size:
                images = F.interpolate(images, size=target_model.module.input_size[-1], mode='bilinear',align_corners=True)
            success, query, adv_images = self.make_adversarial_examples(batch_idx, images.cuda(), true_labels.cuda(),
                                                                 args, attacker, target_model, codec)
            success_list.append(success)
            adv_img_list.append([adv_images, true_labels])
        avg_correct = sum(success_list) / float(len(success_list))
        print('{} is attacked finished ({} images)'.format(arch_name, self.total_images))
        print('        avg correct: {:.4f}'.format(avg_correct.item()))
        # print('        avg correct: {:.4f}'.format(self.correct_all.mean().item()))
        print('       avg not_done: {:.4f}'.format(self.not_done_all.mean().item()))  # 有多少图没做完
        if self.success_all.sum().item() > 0:
            print(
                '     avg mean_query: {:.4f}'.format(self.success_query_all[self.success_all.byte()].mean().item()))
            print(
                '   avg median_query: {:.4f}'.format(self.success_query_all[self.success_all.byte()].median().item()))
            print('     max query: {}'.format(self.success_query_all[self.success_all.byte()].max().item()))
        if self.not_done_all.sum().item() > 0:
            print(
                '  avg not_done_loss: {:.4f}'.format(self.not_done_loss_all[self.not_done_all.byte()].mean().item()))
            print(
                '  avg not_done_prob: {:.4f}'.format(self.not_done_prob_all[self.not_done_all.byte()].mean().item()))
        print('Saving results to {}'.format(result_dump_path))
        # meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          # "avg_not_done": self.not_done_all.mean().item(),
                          # "mean_query": self.success_query_all[self.success_all.byte()].mean().item(),
                          # "median_query": self.success_query_all[self.success_all.byte()].median().item(),
                          # "max_query": self.success_query_all[self.success_all.byte()].max().item(),
                          # "not_done_loss": self.not_done_loss_all[self.not_done_all.byte()].mean().item(),
                          # "not_done_prob": self.not_done_prob_all[self.not_done_all.byte()].mean().item()}
        # meta_info_dict['args'] = vars(args)
        # with open(result_dump_path, "w") as result_file_obj:
            # json.dump(meta_info_dict, result_file_obj, indent=4, sort_keys=True)
        print("done, write stats info to {}".format(result_dump_path))
        if args.transfer:
            baseline_transfer(args, attacker, args.attack_method, arch_name,
                              adv_img_list, l_test_classif_paths, adv_models)

def main(args, arch):
    adv_models = None
    train_loader, test_loader = create_loaders(args, root='../data')
    if args.dataset == 'cifar':
        args.nc, args.h, args.w = 3, 32, 32
        args.input_size = 32
        model, l_test_classif_paths = load_all_classifiers(args, load_archs=[args.source_arch])
        model_type = args.source_arch
        if args.target_arch is not None:
            model_target, l_test_classif_paths = load_all_classifiers(args, load_archs=[args.target_arch])
            model_type = args.target_arch
            del model_target
            torch.cuda.empty_cache()
    elif args.dataset == 'mnist':
        args.input_size = 28
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
    test_classifier(args, model, args.dev, test_loader, epoch=1)
    print("Testing on %d Test Classifiers" %(len(l_test_classif_paths)))
    # attack related settings
    if args.attack_method == "zoo" or args.attack_method == "autozoom_bilin":
        if args.img_resize is None:
            args.img_resize = args.input_size
            print("Argument img_resize is not set and not using autoencoder, set to image original size:{}".format(
                args.img_resize))

    codec = None
    if args.attack_method == "zoo_ae" or args.attack_method == "autozoom_ae":
        codec = Codec(args.input_size, IN_CHANNELS[args.dataset],
                      args.compress_mode, args.resize, use_tanh=args.use_tanh)
        codec.load_codec(args,codec_path)
        codec.cuda()
        decoder = codec.decoder
        args.img_resize = decoder.input_shape[1]
        print("Loading autoencoder: {}, set the attack image size to:{}".format(args.codec_path, args.img_resize))

    # setup attack
    if args.attack_method == "zoo":
        blackbox_attack = ZOO(model, args.dataset, args)
    elif args.attack_method == "zoo_ae":
        blackbox_attack = ZOO_AE(model, args.dataset, args, decoder)
    elif args.attack_method == "autozoom_bilin":
        blackbox_attack = AutoZOOM_BiLIN(model, args.dataset, args)
    elif args.attack_method == "autozoom_ae":
        blackbox_attack = AutoZOOM_AE(model, args["dataset"], args, decoder)
    target_str = "untargeted" if  args.attack_type!="targeted" else "targeted_{}".format(args.target_type)
    attack_framework = AutoZoomAttackFramework(args, test_loader)
    attack_framework.attack_dataset_images(args, blackbox_attack, arch, model,
            codec, l_test_classif_paths=l_test_classif_paths,
            adv_models=adv_models)
    model.cpu()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--attack_method", type=str, required=True,
                        choices=["zoo", "zoo_ae", "autozoom_bilin", "autozoom_ae"], help="the attack method")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="the batch size for zoo, zoo_ae attack")
    parser.add_argument("-c", "--init_const", type=float, default=1, help="the initial setting of the constant lambda")
    parser.add_argument("-d", "--dataset", type=str, required=True, choices=["cifar", "CIFAR-100", "ImageNet", "mnist", "FashionMNIST"])
    parser.add_argument("-m", "--max_iterations", type=int, default=None, help="set 0 to use the default value")
    parser.add_argument("-n", "--num_attack", type=int, default=100,
                        help="Number of images to attack")
    parser.add_argument("-p", "--print_every", type=int, default=100,
                        help="print information every PRINT_EVERY iterations")
    parser.add_argument("--attack_type", default="untargeted", choices=["targeted", "untargeted"],
                        help="the type of attack")
    parser.add_argument('--transfer', action='store_true')
    parser.add_argument("--early_stop_iters", type=int, default=100,
                        help="print objs every EARLY_STOP_ITER iterations, 0 is maxiter//10")
    parser.add_argument("--confidence", default=0, type=float, help="the attack confidence")
    parser.add_argument("--codec_path", default=None, type=str, help="the coedec path, load the default codec is not set")
    parser.add_argument("--target_type", type=str, default="increment",  choices=['random', 'least_likely',"increment"],
                        help="if set, choose random target, otherwise attack every possible target class, only works when ATTACK_TYPE=targeted")
    parser.add_argument("--num_rand_vec", type=int, default=1,
                        help="the number of random vector for post success iteration")
    parser.add_argument("--img_offset", type=int, default=0,
                        help="the offset of the image index when getting attack data")
    parser.add_argument("--img_resize", default=None, type=int,
                        help="this option only works for ATTACK METHOD zoo and autozoom_bilin")
    parser.add_argument("--epsilon", type=float, default=4.6, help="the maximum threshold of L2 constraint")
    parser.add_argument("--resize", default=None,type=int, help="this option only works for the preprocess resize of images")
    parser.add_argument("--switch_iterations", type=int, default=None,
                        help="the iteration number for dynamic switching")
    parser.add_argument("--compress_mode", type=int, default=None,
                        help="specify the compress mode if autoencoder is used")
    parser.add_argument('--test_archs', action="store_true")
    parser.add_argument('--use_tanh', default=False, action="store_true")
    parser.add_argument('--source_arch', default="res18",
                        help="The architecture we want to attack on CIFAR.")
    parser.add_argument('--target_arch', default=None,
                        help="The architecture we want to blackbox transfer to on CIFAR.")
    parser.add_argument('--noise', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate for the generator (default: 0.01)')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--robust_load_path', type=str, default='../../Nattack/all_models/robustnet/noise_0.3.pth')
    parser.add_argument('--load_path', type=str,
                        default='../pretrained_classifiers/cifar/VGG16/model_1.pt')
    parser.add_argument('--robust_model_path', type=str,
                        default="../madry_challenge_models/mnist/adv_trained/mnist_lenet5_advtrained.pt")
    parser.add_argument('--dir_test_models', type=str,
                        default="../",
                        help="The path to the directory containing the classifier models for evaluation.")
    parser.add_argument('--train_set', default='test',
                        choices=['train_and_test','test','train'],
                        help='add the test set in the training set')
    parser.add_argument('--train_on_list', default=False, action='store_true',
                        help='train on a list of classifiers')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='S')
    parser.add_argument('--train_with_critic_path', type=str, default=None,
                        help='Train generator with saved critic model')
    # args = vars(args)
    args = parser.parse_args()
    args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.max_test_model = 2
    # args["codec_path"] = list(glob.glob(args["codec_path"].format(PY_ROOT)))[0]

    if args.img_resize is not None:
        if args.attack_method == "zoo_ae" or args.attack_method == "autozoom_ae":
            print("Attack method {} cannot use option img_resize, arugment ignored".format(args["attack_method"]))

    if args.attack_type == "targeted" and args.max_iterations < 20000:
            args.max_iterations = 5 * args.max_iterations

    print('Command line is: {}'.format(' '.join(sys.argv)))
    print('Called with args:')

    # setup random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)


    archs = [args.source_arch]
    dataset = args.dataset
    if args.test_archs:
        archs.clear()
        if dataset == "cifar" or dataset == "CIFAR-100":
            for arch in MODELS_TEST_STANDARD[dataset]:
                test_model_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/{}/checkpoint.pth.tar".format(
                    PY_ROOT,
                    dataset, arch)
                if os.path.exists(test_model_path):
                    archs.append(arch)
                else:
                    print(test_model_path + " does not exists!")
        else:
            for arch in MODELS_TEST_STANDARD[dataset]:
                test_model_list_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/checkpoints/{}*.pth.tar".format(
                    PY_ROOT,
                    dataset, arch)
                test_model_list_path = list(glob.glob(test_model_list_path))
                if len(test_model_list_path) == 0:  # this arch does not exists in args.dataset
                    continue
                archs.append(arch)
        args.arch = ",".join(archs)
    for arch in archs:
        main(args, arch)

