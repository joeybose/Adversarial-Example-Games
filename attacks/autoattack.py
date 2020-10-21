import os
import wandb
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import argparse
import time
import ipdb
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from utils.utils import create_loaders, load_unk_model, test_classifier
from classifiers import load_all_classifiers, load_list_classifiers, load_dict_classifiers
from cnn_models import LeNet as Net
from cnn_models import ResNet18
from eval import baseline_transfer, baseline_eval_classifier

class AutoAttack():
    def __init__(self, args, n_iter, model, norm='Linf', eps=.3, seed=None,
                 verbose=True, attacks_to_run=['fab', 'square', 'apgd-ce',
                                                'apgd-dlr'], plus=False,
                 is_tf_model=False, log_path=None):
        self.model = model
        self.args = args
        self.norm = norm
        self.n_iter = n_iter
        assert norm in ['Linf', 'L2']
        self.epsilon = eps
        self.seed = seed
        self.verbose = verbose
        if plus:
            attacks_to_run.extend(['apgd-t', 'fab-t'])
        self.attacks_to_run = attacks_to_run
        self.plus = plus
        self.is_tf_model = is_tf_model
        self.device = args.dev
        self.logger = Logger(log_path)

        # Import Attacks
        try:
            from .autopgd_pt import APGDAttack
            from .fab_pt import FABAttack
            from .square_pt import SquareAttack
            from .autopgd_pt import APGDAttack_targeted
        except:
            from autopgd_pt import APGDAttack
            from fab_pt import FABAttack
            from square_pt import SquareAttack
            from autopgd_pt import APGDAttack_targeted

        self.apgd = APGDAttack(args, self.model, n_restarts=5, n_iter=self.n_iter,
                               verbose=False, eps=self.epsilon, norm=self.norm,
                               eot_iter=1, rho=.75, seed=self.seed)

        self.fab = FABAttack(args, self.model, n_restarts=5, n_iter=self.n_iter,
                             eps=self.epsilon, seed=self.seed, norm=self.norm,
                             verbose=False)

        self.square = SquareAttack(args, self.model, p_init=.8,
                                   n_queries=5000, eps=self.epsilon,
                                   norm=self.norm, n_restarts=5,
                                   seed=self.seed, verbose=False,
                                   resc_schedule=False)

        self.apgd_targeted = APGDAttack_targeted(args, self.model,
                                                 n_restarts=5,
                                                 n_iter=self.n_iter,
                                                 verbose=False,
                                                 eps=self.epsilon,
                                                 norm=self.norm, eot_iter=1,
                                                 rho=.75, seed=self.seed)

    def get_logits(self, x):
        if not self.is_tf_model:
            return self.model(x)
        else:
            return self.model.predict(x)

    def get_seed(self):
        return time.time() if self.seed is None else self.seed

    def train(self, train_loader, test_loader, l_test_classif, l_train_classif):
        pass

    def run_standard_evaluation(self, x_orig, y_orig, bs=250, adv_models=None):
        # update attacks list if plus activated or after initialization
        if self.plus:
            if not 'apgd-t' in self.attacks_to_run:
                self.attacks_to_run.extend(['apgd-t'])
            if not 'fab-t' in self.attacks_to_run:
                self.attacks_to_run.extend(['fab-t'])

        with torch.no_grad():
            # calculate accuracy
            n_batches = int(np.ceil(x_orig.shape[0] / bs))
            robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
            for batch_idx in range(n_batches):
                start_idx = batch_idx * bs
                end_idx = min( (batch_idx + 1) * bs, x_orig.shape[0])

                x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                y = y_orig[start_idx:end_idx].clone().to(self.device)
                output = self.get_logits(x)
                correct_batch = y.eq(output.max(dim=1)[1])
                robust_flags[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)

            robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
            if self.verbose:
                # self.logger.log('initial accuracy: {:.2%}'.format(robust_accuracy))
                print('initial accuracy: {:.2%}'.format(robust_accuracy))

            x_adv = x_orig.clone().detach()
            startt = time.time()
            for attack in self.attacks_to_run:
                # item() is super important as pytorch int division uses floor rounding
                num_robust = torch.sum(robust_flags).item()

                if num_robust == 0:
                    break

                n_batches = int(np.ceil(num_robust / bs))

                robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
                if num_robust > 1:
                    robust_lin_idcs.squeeze_()

                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, num_robust)

                    batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                    if len(batch_datapoint_idcs.shape) > 1:
                        batch_datapoint_idcs.squeeze_(-1)
                    x = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                    y = y_orig[batch_datapoint_idcs].clone().to(self.device)

                    # make sure that x is a 4d tensor even if there is only a single datapoint left
                    if len(x.shape) == 3:
                        x.unsqueeze_(dim=0)

                    # run attack
                    if attack == 'apgd-ce':
                        # apgd on cross-entropy loss
                        self.apgd.loss = 'ce'
                        self.apgd.seed = self.get_seed()
                        _, adv_curr = self.apgd.perturb(x, y, cheap=True)

                    elif attack == 'apgd-dlr':
                        # apgd on dlr loss
                        self.apgd.loss = 'dlr'
                        self.apgd.seed = self.get_seed()
                        _, adv_curr = self.apgd.perturb(x, y, cheap=True)

                    elif attack == 'fab':
                        # fab
                        self.fab.targeted = False
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)

                    elif attack == 'square':
                        # square
                        self.square.seed = self.get_seed()
                        adv_curr = self.square.perturb(x, y)

                    elif attack == 'apgd-t':
                        # targeted apgd
                        self.apgd_targeted.seed = self.get_seed()
                        _, adv_curr = self.apgd_targeted.perturb(x, y, cheap=True)

                    elif attack == 'fab-t':
                        # fab targeted
                        self.fab.targeted = True
                        self.fab.n_restarts = 1
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)

                    else:
                        raise ValueError('Attack not supported')
                    output = self.get_logits(adv_curr)
                    false_batch = ~y.eq(output.max(dim=1)[1]).to(robust_flags.device)
                    non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                    robust_flags[non_robust_lin_idcs] = False

                    x_adv[non_robust_lin_idcs] = adv_curr[false_batch].detach().to(x_adv.device)

                    if self.verbose:
                        num_non_robust_batch = torch.sum(false_batch)
                        print(('{} - {}/{} - {} out of {} successfully perturbed'.format( attack, batch_idx + 1,
                               n_batches, num_non_robust_batch, x.shape[0])))

                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                if self.verbose:
                    print('robust accuracy after {}: {:.2%} (total time {:.1f} s)'.format(attack.upper(), robust_accuracy,
                              time.time() - startt))

            # final check
            if self.verbose:
                if self.norm == 'Linf':
                    res = (x_adv - x_orig).abs().view(x_orig.shape[0], -1).max(1)[0]
                elif self.norm == 'L2':
                    res = ((x_adv - x_orig) ** 2).view(x_orig.shape[0], -1).sum(-1).sqrt()
                print('max {} perturbation: {:.5f}, nan in tensor: {}, max: {:.5f}, min: {:.5f}'.format(
                    self.norm, res.max(), (x_adv != x_adv).sum(), x_adv.max(), x_adv.min()))
                print('robust accuracy: {:.2%}'.format(robust_accuracy))
        return x_adv, robust_accuracy

    def clean_accuracy(self, x_orig, y_orig, bs=250):
        n_batches = x_orig.shape[0] // bs
        acc = 0.
        for counter in range(n_batches):
            x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            y = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            output = self.get_logits(x)
            acc += (output.max(1)[1] == y).float().sum()

        if self.verbose:
            print('clean accuracy: {:.2%}'.format(acc / x_orig.shape[0]))
        return acc.item() / x_orig.shape[0]

    def run_standard_evaluation_individual(self, args, x_orig, y_orig, bs=250,
                                           adv_models=None, transfer_eval=False, test_paths=None):
        # update attacks list if plus activated after initialization
        if self.plus:
            if not 'apgd-t' in self.attacks_to_run:
                self.attacks_to_run.extend(['apgd-t'])
            if not 'fab-t' in self.attacks_to_run:
                self.attacks_to_run.extend(['fab-t'])

        l_attacks = self.attacks_to_run
        adv = {}
        self.plus = False
        verbose_indiv = self.verbose
        attack_fool_rate = []
        acc_indiv  = self.clean_accuracy(x_orig, y_orig, bs=bs)
        for c in l_attacks:
            startt = time.time()
            self.attacks_to_run = [c]
            adv[c], robust_accuracy = self.run_standard_evaluation(x_orig,
                    y_orig, bs=bs, adv_models=adv_models)
            if transfer_eval:
                model_type = args.source_arch
                adv_img_list = []
                for i in range(0, len(adv[c])):
                    adv_img_list.append([adv[c][i].unsqueeze(0), y_orig[i]])
                baseline_transfer(args, c, c, model_type, adv_img_list,
                                  test_paths, adv_models)

            fool_rate = 1 - robust_accuracy
            attack_fool_rate.append([c, fool_rate])
            space = '\t \t' if c == 'fab' else '\t'
            print('robust accuracy by {} {} {:.2%} \t (time attack: {:.1f} s)'.format(
                c.upper(), space, acc_indiv,  time.time() - startt))

        return adv, attack_fool_rate

    def cheap(self):
        self.apgd.n_restarts = 1
        self.fab.n_restarts = 1
        self.apgd_targeted.n_restarts = 1
        self.square.n_queries = 1000
        self.square.resc_schedule = True
        self.plus = False

class Logger():
    def __init__(self, log_path):
        self.log_path = log_path

    def log(self, str_to_log):
        print(str_to_log)
        if not self.log_path is None:
            with open(self.log_path, 'a') as f:
                f.write(str_to_log + '\n')
                f.flush()

def main():

    parser = argparse.ArgumentParser()
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
    parser.add_argument('--eps', type=float, default=0.031)
    parser.add_argument('--batch_size', type=int, default=256, metavar='S')
    parser.add_argument('--test_batch_size', type=int, default=512, metavar='S')
    parser.add_argument('--train_set', default='test',
                        choices=['train_and_test','test','train'],
                        help='add the test set in the training set')
    # parser.add_argument('--modelIn', type=str, default='../../Nattack/all_models/robustnet/noise_0.3.pth')
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
    parser.add_argument('--epsilon', type=float, default=0.1, metavar='M',
                        help='Epsilon for Delta (default: 0.1)')
    parser.add_argument('--train_with_critic_path', type=str, default=None,
                        help='Train generator with saved critic model')
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
    elif args.dataset == 'mnist':
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
    print("Testing on %d Test Classifiers" %(len(l_test_classif_paths)))
    l = [x.unsqueeze(0) for (x, y) in test_loader.dataset]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader.dataset]
    y_test = torch.Tensor(l).long()
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print("CUDA Device Count is %d, Error might happen. Use export CUDA_VISIBLE_DEVICES=0" %(device_count))

    if not args.sweep:
        attacker = AutoAttack(args, args.n_iter, model, norm=args.attack_ball,
                              verbose=True, eps=args.epsilon)
        adv, attack_fool_rate = attacker.run_standard_evaluation_individual(args,
                                                    x_test[:args.batch_size],
                                                    y_test[:args.batch_size],
                                                    bs=args.batch_size,
                                                    transfer_eval=args.transfer,
                                                    test_paths=l_test_classif_paths,
                                                    adv_models=adv_models)
        adv_complete, success_rate = attacker.run_standard_evaluation(x_test[:args.batch_size],
                                                y_test[:args.batch_size],
                                                       bs=args.batch_size,
                                                       adv_models=adv_models)
        if args.transfer:
            adv_img_list = []
            y_orig = y_test[:args.batch_size]
            for i in range(0, len(adv_complete)):
                adv_img_list.append([adv_complete[i].unsqueeze(0), y_orig[i]])
            baseline_transfer(args, attacker, "AutoAttack", model_type,
                              adv_img_list, l_test_classif_paths, adv_models)
    else:
        for n_iter in range(0, args.n_iter, args.query_step):
            attacker = AutoAttack(args, n_iter, model, verbose=True,
                                  norm=args.attack_ball, eps=args.epsilon)
            adv, attack_fool_rate = attacker.run_standard_evaluation_individual(args,
                                                        x_test[:args.batch_size],
                                                        y_test[:args.batch_size],
                                                        bs=args.batch_size,
                                                        transfer_eval=args.transfer,
                                                        test_paths=l_test_classif_paths,
                                                        adv_models=adv_models)
            adv_complete, success_rate = attacker.run_standard_evaluation(x_test[:args.batch_size],
                                                    y_test[:args.batch_size],
                                                           bs=args.batch_size,
                                                           adv_models=adv_models)
            if args.transfer:
                adv_img_list = []
                y_orig = y_test[:args.batch_size]
                for i in range(0, len(adv_complete)):
                    adv_img_list.append([adv_complete[i].unsqueeze(0), y_orig[i]])
                baseline_transfer(args, attacker, "AutoAttack", model_type,
                                  adv_img_list, l_test_classif_paths,
                                  adv_models)
            if args.wandb:
                wandb.log({"APGD-CE": attack_fool_rate[0][1],
                           "APGD-DLR": attack_fool_rate[1][1],
                           "FAB": attack_fool_rate[2][1],
                           "Square": attack_fool_rate[3][1],
                           "queries": n_iter
                           })

if __name__ == '__main__':
    main()
