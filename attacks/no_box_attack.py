from advertorch.attacks import Attack
import torch
from tqdm import tqdm
from torch import optim
import ipdb
import sys
from __init__ import data_and_model_setup, load_data
from __init__ import eval as eval_attacker
import wandb
import torch
from torch import autograd
from torch.distributions.bernoulli import Bernoulli
import os
from advertorch.attacks import LinfPGDAttack
import argparse
from advertorch.context import ctx_noparamgrad_and_eval
from torchvision import transforms


sys.path.append("..")  # Adds higher directory to python modules path.
from utils import config as cf
from cnn_models import LeNet as Net
from attack_helpers import Linf_dist, L2_dist, ce_loss_func, save_image_to_wandb
from attack_helpers import attack_ce_loss_func, carlini_wagner_loss, non_saturating_loss, targeted_cw_loss
from models import create_generator
from utils.extragradient import Extragradient
import eval
from eval import baseline_transfer
from utils.utils import create_loaders, load_unk_model, nobox_wandb, CIFAR_NORMALIZATION
from classifiers import load_all_classifiers
from cnn_models.mnist_ensemble_adv_train_models import *
from defenses.ensemble_adver_train_mnist import *
kwargs_perturb_loss = {'Linf': Linf_dist, 'L2': L2_dist}
kwargs_attack_loss = {'cross_entropy': attack_ce_loss_func, 'carlini': carlini_wagner_loss,
                      'non_saturating': non_saturating_loss, 'targeted_cw': targeted_cw_loss}


class NoBoxAttack(Attack):
    def __init__(self, predict, loss_fn, args, logger=None):
        super(NoBoxAttack, self).__init__(predict, loss_fn, args.clip_min, args.clip_max)
        self.args = args
        flow_args = [args.n_blocks, args.flow_hidden_size, args.flow_hidden,
                     args.flow_model, args.flow_layer_type]
        self.G = create_generator(args, args.model, args.deterministic_G,
                                  args.dataset, flow_args)

        self.pick_prob_start = None
        self.robust_train_flag = False
        if args.train_with_robust:
            if args.ensemble_adv_trained:
                if args.dataset=="mnist":
                    adv_model_names = args.adv_models
                    type = get_model_type(adv_model_names[0])
                    self.robust_critic = load_model(args, adv_model_names[0],
                                                    type=type).to(args.dev)

                elif args.dataset =='cifar':
                    adv_model_names = args.adv_models
                    l_test_classif_paths = []
                    adv_path = os.path.join(args.dir_test_models, "pretrained_classifiers",
                                        args.dataset, "ensemble_adv_trained",
                                        adv_model_names + '.pt')
                    init_func, _ = ARCHITECTURES[adv_model_names[i]]
                    temp_model = init_func().to(args.dev)
                    self.robust_critic = nn.DataParallel(temp_model)
                    self.robust_critic.load_state_dict(torch.load(adv_path))
            else:
                self.robust_critic = Net(1, 28, 28).to(args.dev)
                self.robust_critic.load_state_dict(torch.load(args.robust_model_path))

            self.robust_train_flag = True
            self.pick_prob_start = args.robust_sample_prob
            self.pick_rob_prob = Bernoulli(torch.tensor([1-args.robust_sample_prob]))
            self.anneal_rate = (1.0 - args.robust_sample_prob) / args.attack_epochs

        self.logger = logger

    def load(self, args):
        if args.save_model is None:
            filename = f'saved_models/generator.pt'
        else:
            filename = os.path.join(args.save_model, f'generator.pt')

        checkpoint = torch.load(filename)
        if "args" in checkpoint:
            args = checkpoint["args"]
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]

        self.G.load_state_dict(checkpoint)
        self.args = args

    def perturb(self, x, target=None, compute_kl=False, anneal_eps=None):
        if self.args.model in ['vanilla_G', 'DC_GAN']:
            adv_inputs = self.G(x)
            kl_div = torch.Tensor([0]).to(self.args.dev).item()
            if anneal_eps is not None and anneal_eps > 1. :
                raise("epsilon annealing not supported")
        elif self.args.model in ['CondGen', 'Resnet']:
            epsilon = self.args.epsilon
            if anneal_eps is not None:
                epsilon *= anneal_eps
            adv_inputs, entropy = self.G(x, torch.tensor([epsilon]).to(self.args.dev), target=target)
            kl_div = torch.Tensor([0]).to(self.args.dev).item() #TODO: maybe return the entropy
        else:
            adv_inputs, kl_div = self.G(x)
            kl_div = kl_div.sum() / len(x)
        adv_inputs = adv_inputs.view_as(x)
        if compute_kl:
            return adv_inputs, kl_div
        else:
            return adv_inputs

    def get_optim(self,*args,**kwargs):
        if self.args.fixed_critic:
            class NullOpt():
                def __init__(self):
                    pass
                def step(self):
                    pass
            return NullOpt()
        else:
            return optim.Adam(*args,**kwargs)

    def gen_update(self, args, epoch, batch_idx, x, target, adv_inputs,
                   l_train_classif, kl_div, perturb_loss_func, gen_opt):

        clamped_adv_input = torch.clamp(adv_inputs, min=.0, max=1.)
        dist_perturb = L2_dist(clamped_adv_input, adv_inputs)
        adv_inputs = torch.clamp(adv_inputs, min=.0, max=1.)
        loss_perturb = dist_perturb.mean()
        if self.robust_train_flag and use_robust_critic:
            pred = self.robust_critic(adv_inputs)
        else:
            pred = self.predict(adv_inputs)

        if args.train_on_list:
            self.predict.cpu()
            for name, predict in l_train_classif.items():
                predict.to(args.dev)
                pred = torch.cat((pred,predict(adv_inputs)),0)
                predict.cpu()
            num_target = (len(l_train_classif) + 1)
            self.predict.to(args.dev)
        else:
            num_target = 1

        if args.attack_loss == 'targeted_cw':
            cat_target = torch.cat(num_target * [adv_target])
        else:
            cat_target = torch.cat(num_target * [target])

        # Compute the loss of the generator
        if args.perturb_magnitude is not None:
            loss_soft_perturb = args.perturb_magnitude * (loss_perturb - args.perturb_magnitude)**2
        else:
            loss_soft_perturb = torch.Tensor([0]).to(args.dev).item()
        loss_misclassify = self.loss_fn(args, pred, cat_target)
        loss_gen = loss_misclassify + loss_soft_perturb + kl_div + args.hinge_coeff * loss_perturb
        # Optim step for the generator
        grad_gen = autograd.grad(loss_gen, self.G.parameters(), retain_graph=True, allow_unused=True)
        for p, g in zip(self.G.parameters(), grad_gen):
            p.grad = g
        gen_opt.step()
        return loss_misclassify, loss_gen, loss_perturb

    def critic_update(self, args, epoch, train_loader, batch_idx, x, target, adv_pred,
                      model_opt, pgd_adversary=None):
            if args.source_arch == 'ens_adv' or args.ensemble_adv_trained:
                x_advs = [None] * (len(args.ens_adv_models) + 1)
                for i, m in enumerate(args.ens_adv_models + [self.predict]):
                    grad = gen_grad(x, m, target, loss='training')
                    x_advs[i] = symbolic_fgs(x, grad, eps=args.epsilon)
                loss_model = train_ens(epoch, batch_idx , self.predict, x, target,
                                       model_opt, x_advs=x_advs,
                                       opt_step=False)
                clean_pred = self.predict(x.detach())
                clean_out = clean_pred.max(1, keepdim=True)[1]  # get the index of the max log-probability
                cat_pred = torch.cat((clean_pred, adv_pred), 0)
                cat_target = torch.cat((target, target), 0)
            else:
                clean_pred = self.predict(x.detach())
                clean_out = clean_pred.max(1, keepdim=True)[1]  # get the index of the max log-probability
                target_clean = target
                cat_pred =  adv_pred
                cat_target = target
                if args.pgd_on_critic:
                    pgd_input = pgd_adversary.perturb(x,target)
                    pgd_pred = self.predict(pgd_input.detach())
                    cat_pred = torch.cat((cat_pred,pgd_pred), 0)
                    cat_target = torch.cat((cat_target,target),0)
                else:
                    cat_pred = adv_pred
                    cat_target = target
                loss_model = ce_loss_func(args, cat_pred, cat_target)

            if args.lambda_on_clean > .0:
                x_train, clean_target = next(iter(train_loader))
                x_train, clean_target = x_train.to(args.dev), clean_target.to(args.dev)
                clean_pred = self.predict(x_train.detach())
                clean_out = clean_pred.max(1, keepdim=True)[1]  # get the index of the max log-probability
                target_clean = clean_target
                loss_model = loss_model + args.lambda_on_clean * ce_loss_func(args, clean_pred, clean_target)
            if args.gp_coeff > 0.:
                grad_model_input = autograd.grad(loss_model, x, create_graph=True,
                                                 retain_graph=True)
                norm_grad = 0
                for g in grad_model_input:
                    norm_grad += (g.norm(1) ** 2).mean()
            else:
                norm_grad = torch.Tensor([0]).to(args.dev).item()
            grad_model = autograd.grad(loss_model + args.gp_coeff * norm_grad, self.predict.parameters())
            for p, g in zip(self.predict.parameters(), grad_model):
                p.grad = g
            model_opt.step()
            return loss_model, clean_out, target_clean


    def batch_update(self, args, x, target, gen_opt, max_grad_steps=10):
        """Given a batch, update the generator only
        Args:
            max_tries: number of gradient steps to try and get an adv sample
        Returns the adversarial samples generated
        """
        x, target = x.to(args.dev), target.to(args.dev)
        x_prime = torch.zeros_like(x)
        success_mask = torch.zeros_like(x)
        batch_size = x.shape[0]
        successful = 0 # number of samples successfully fooled
        perturb_loss_func = kwargs_perturb_loss[args.perturb_loss]
        tries = 0
        while successful < batch_size and tries < max_grad_steps:
            adv_inputs, kl_div = self.perturb(x, compute_kl=True, target=target)

            dist_perturb = perturb_loss_func(x, adv_inputs)
            loss_perturb = dist_perturb.mean()
            if self.robust_train_flag and use_robust_critic:
                pred = self.robust_critic(adv_inputs)
            else:
                pred = self.predict(adv_inputs)
            out = pred.max(1, keepdim=True)[1]  # get the index of the max log-probability
            cat_target = target
            # Compute the loss of the generator
            if args.perturb_magnitude is not None:
                loss_soft_perturb = args.perturb_magnitude * (loss_perturb - args.perturb_magnitude)**2
            else:
                loss_soft_perturb = torch.Tensor([0]).to(args.dev).item()

            # If samples incorrectly classified, then no backprop for those
            # TODO
            # Get samples that successfully fooled
            mask = ~out.eq(target.unsqueeze(1).data)
            successful += mask.sum()
            # We only care about new successes
            mask = mask.unsqueeze(2).unsqueeze(3).repeat(1,1,*x.shape[-2:])
            mask = mask.to(success_mask.dtype) #cast bool to float
            new_successes = torch.relu(mask - success_mask)
            success_mask += new_successes

            # Save new samples that successfully fooled
            x_prime = x_prime + adv_inputs*new_successes

            loss_misclassify = self.loss_fn(args, pred, cat_target)
            loss_gen = loss_misclassify + loss_soft_perturb + kl_div
            # Optim step for the generator
            grad_gen = autograd.grad(loss_gen, self.G.parameters(), retain_graph=True)
            for p, g in zip(self.G.parameters(), grad_gen):
                p.grad = g

            gen_opt.step()
            tries += 1
        return x_prime

    def train(self, train_loader, test_loader, adv_models,
            l_test_classif_paths, l_train_classif=None, eval_fn=None):
        args = self.args
        pgd_adversary = None

        if args.save_model is None:
            filename = 'saved_models/generator.pt'
        else:
            filename = os.path.join(args.save_model, 'generator.pt')

        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        gen_opt = optim.Adam(self.G.parameters(), lr=args.lr,
                             betas=(args.momentum, .99))
        if args.lr_model is None:
            args.lr_model = args.lr
        model_opt = self.get_optim(self.predict.parameters(), lr=args.lr_model,
                                   betas=(args.momentum, .99))
        if args.extragradient:
            gen_opt = Extragradient(gen_opt, self.G.parameters())
            if not args.fixed_critic:
                model_opt = Extragradient(model_opt, self.predict.parameters())
        if args.lr_model is None:
            args.lr_model = args.lr
        if args.pgd_on_critic:
            pgd_adversary = LinfPGDAttack(
                self.predict, loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
                eps=args.epsilon, nb_iter=40, eps_iter=0.01, rand_init=True,
                clip_min=0.0,clip_max=1.0, targeted=False)

        # Choose Attack Loss
        perturb_loss_func = kwargs_perturb_loss[args.perturb_loss]
        pick_prob = self.pick_prob_start
        ''' Training Phase '''
        best_acc = 100
        for epoch in range(0, args.attack_epochs):
            adv_correct = 0
            clean_correct = 0
            num_clean = 0
            if args.train_set == 'test':
                print("Training generator on the test set")
                loader = test_loader
            elif args.train_set == 'train_and_test':
                print("Training generator on the test set and the train set")
                loader = concat_dataset(args, train_loader,test_loader)
            else:
                print("Training generator on the train set")
                loader = train_loader
            train_itr = tqdm(enumerate(loader),
                         total=len(list(loader)))
            if self.robust_train_flag:
                use_robust_critic = self.pick_rob_prob.sample()
                pick_prob = min(1.0, pick_prob + self.anneal_rate)
                self.pick_rob_prob = Bernoulli(torch.tensor([1-pick_prob]))
                print("Using Robust Critic this Epoch with Prob: %f" %(1-pick_prob))

            for batch_idx, (data, target) in train_itr:
                x, target = data.to(args.dev), target.to(args.dev)
                num_unperturbed = 10
                iter_count = 0
                loss_perturb = 20
                loss_model = 0
                loss_misclassify = -10
                anneal_eps = 1 + args.anneal_eps**(epoch+1)
                for i in range(args.max_iter):
                    if args.not_use_labels:
                        # erase the current target to provide
                        adv_inputs, kl_div = self.perturb(x, compute_kl=True,
                                                          anneal_eps=anneal_eps)
                    else:
                        adv_inputs, kl_div = self.perturb(x, compute_kl=True,
                                                          anneal_eps=anneal_eps, target=target)

                    # Optim step for the generator
                    loss_misclassify, loss_gen, loss_perturb = self.gen_update(args, epoch,
                                                                 batch_idx, x,
                                                                 target,
                                                                 adv_inputs,
                                                                 l_train_classif,
                                                                 kl_div,
                                                                 perturb_loss_func,
                                                                 gen_opt)

                    iter_count = iter_count + 1
                    if iter_count > args.max_iter:
                        break

                if args.gp_coeff > 0.:
                    x = autograd.Variable(x, requires_grad=True)

                adv_inputs = torch.clamp(self.perturb(x, anneal_eps=anneal_eps), min=0., max=1.0)
                adv_pred = self.predict(adv_inputs)
                adv_out = adv_pred.max(1, keepdim=True)[1]

                # Optim step for the classifier
                loss_model, clean_out, target_clean = self.critic_update(args, epoch, train_loader,
                                                           batch_idx, x,
                                                           target, adv_pred,
                                                           model_opt,
                                                           pgd_adversary)

                adv_correct += adv_out.eq(target.unsqueeze(1).data).sum()
                clean_correct += clean_out.eq(target_clean.unsqueeze(1).data).sum()
                num_clean += target_clean.shape[0]

            if args.wandb:
                nobox_wandb(args, epoch, x, target, adv_inputs, adv_out,
                            adv_correct, clean_correct, loss_misclassify,
                            loss_model, loss_perturb, loss_gen, train_loader)

            print(f'\nTrain: Epoch:{epoch} Loss: {loss_model:.4f}, Gen Loss :{loss_gen:.4f}, '
                  f'Missclassify Loss :{loss_misclassify:.4f} '
                  f'Clean. Acc: {clean_correct}/{num_clean} '
                  f'({100. * clean_correct.cpu().numpy()/num_clean:.0f}%) '
                  f'Perturb Loss {loss_perturb:.4f} Adv. Acc: {adv_correct}/{len(loader.dataset)} '
                  f'({100. * adv_correct.cpu().numpy()/len(loader.dataset):.0f}%)\n')

            if (epoch + 1) % args.eval_freq == 0:
                if eval_fn is None:
                    with torch.no_grad():
                        mean_acc, std_acc, all_acc = eval.eval(args, self,
                                test_loader, l_test_classif_paths,
                                logger=self.logger, epoch=epoch)
                else:
                    if args.target_arch is not None:
                        model_type = args.target_arch
                    elif args.source_arch == "adv" or (args.source_arch == "ens_adv" and args.dataset == "mnist"):
                        model_type =  [args.model_type]
                    else:
                        model_type = [args.adv_models[0]]

                    eval_helpers = [self.predict, model_type, adv_models, l_test_classif_paths, test_loader]

                    mean_acc = eval_attacker(args, self, "AEG", eval_helpers, args.num_eval_samples)
                    # eval_fn(list(l_train_classif.values())[0])

                if mean_acc < best_acc:
                    best_acc = mean_acc
                    try:
                        torch.save({"model": self.G.state_dict(), "args": args}, filename)
                    except:
                        print("Warning: Failed to save model !")

        return adv_out, adv_inputs


def main():

    parser = argparse.ArgumentParser(description='NoBox')
    # Hparams
    parser.add_argument('--gp_coeff', type=float, default=0.,
                        help='coeff for the gradient penalty')
    parser.add_argument('--latent_dim', type=int, default=20, metavar='N',
                        help='Latent dim for VAE')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate for the generator (default: 0.01)')
    parser.add_argument('--lr_model', type=float, default=None, metavar='LR',
                        help='learning rate for the model (default: None -> default to args.lr)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='optimizer momentum (default: 0.5)')
    parser.add_argument('--extragradient', default=False, action='store_true',
                        help='Use extragadient algorithm')
    parser.add_argument('--latent_size', type=int, default=50, metavar='N',
                        help='Size of latent distribution (default: 50)')
    parser.add_argument('--flow_model', default=None, const='soft',
                    nargs='?', choices=[None, 'RealNVP', 'planar', 'radial'],
                    help='Type of Normalizing Flow (default: %(default)s)')
    parser.add_argument('--flow_layer_type', type=str, default='Linear',
                        help='Which type of layer to use ---i.e. GRevNet or Linear')
    parser.add_argument('--flow_hidden_size', type=int, default=128,
                        help='Hidden layer size for Flows.')
    parser.add_argument('--n_blocks', type=int, default=2,
                        help='Number of blocks to stack in flow')
    parser.add_argument('--flow_hidden', type=int, default=1, help='Number of hidden layers in each Flow.')
    parser.add_argument('--eval_set', default="test",
                        help="Evaluate model on test or validation set.")
    parser.add_argument('--train_with_critic_path', type=str, default=None,
                        help='Train generator with saved critic model')
    parser.add_argument('--train_on_file', default=False, action='store_true',
                        help='Train using Madry tf grad')
    # Training
    parser.add_argument('--lambda_on_clean', default=0.0, type=float,
                        help='train the critic on clean examples of the train set')
    parser.add_argument('--not_use_labels', default=False, action='store_true',
                        help='Use the labels for the conditional generator')
    parser.add_argument('--hinge_coeff', default=10., type=float,
                        help='coeff for the hinge loss penalty')
    parser.add_argument('--anneal_eps', default=0., type=float,
                        help='coeff for the epsilon annealing')
    parser.add_argument('--fixed_critic', default=False, action='store_true',
                        help='Critic is not trained')
    parser.add_argument('--train_on_list', default=False, action='store_true',
                        help='train on a list of classifiers')
    parser.add_argument('--train_set', default='train',
                        choices=['train_and_test','test','train'],
                        help='add the test set in the training set')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--n_iter', type=int, default=500,
                        help='N iters for quere based attacks')
    parser.add_argument('--PGD_steps', type=int, default=40, metavar='N',
                        help='max gradient steps (default: 30)')
    parser.add_argument('--max_iter', type=int, default=10, metavar='N',
                        help='max gradient steps (default: 10)')
    parser.add_argument('--epsilon', type=float, default=0.1, metavar='M',
                        help='Epsilon for Delta (default: 0.1)')
    parser.add_argument('--attack_ball', type=str, default="L2",
                        choices= ['L2','Linf'],
                        help='type of box attack')
    parser.add_argument('--bb_steps', type=int, default=2000, metavar='N',
                        help='Max black box steps per sample(default: 1000)')
    parser.add_argument('--attack_epochs', type=int, default=100, metavar='N',
                        help='Max numbe of epochs to train G')
    parser.add_argument('--num_flows', type=int, default=2, metavar='N',
                        help='Number of Flows')
    parser.add_argument('--seed', type=int, metavar='S',
                        help='random seed (default: None)')
    parser.add_argument('--input_size', type=int, default=784, metavar='S',
                        help='Input size for MNIST is default')
    parser.add_argument('--batch_size', type=int, default=256, metavar='S',
                        help='Batch size')
    parser.add_argument('--test_batch_size', type=int, default=512, metavar='S',
                        help='Test Batch size')
    parser.add_argument('--pgd_on_critic', default=False, action='store_true',
                        help='Train Critic on pgd samples')
    parser.add_argument('--train_with_robust', default=False, action='store_true',
                        help='Train with Robust model + Critic')
    parser.add_argument('--test', default=False, action='store_true',
                        help='just test model and print accuracy')
    parser.add_argument('--clip_grad', default=True, action='store_true',
                        help='Clip grad norm')
    parser.add_argument('--train_vae', default=False, action='store_true',
                        help='Train VAE')
    parser.add_argument('--train_ae', default=False, action='store_true',
                        help='Train AE')
    parser.add_argument('--attack_type', type=str, default='nobox',
                        help='Which attack to run')
    parser.add_argument('--attack_loss', type=str, default='cross_entropy',
                        help='Which loss func. to use to optimize G')
    parser.add_argument('--perturb_loss', type=str, default='L2', choices= ['L2','Linf'],
                        help='Which loss func. to use to optimize to compute constraint')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--deterministic_G', default=False, action='store_true',
                        help='Deterministic Latent State')
    parser.add_argument('--run_baseline', default=False, action='store_true',
                        help='Run baseline PGD')
    parser.add_argument('--resample_test', default=False, action='store_true',
                help='Load model and test resampling capability')
    parser.add_argument('--resample_iterations', type=int, default=100, metavar='N',
                        help='How many times to resample (default: 100)')
    parser.add_argument('--architecture', default="VGG16",
                        help="The architecture we want to attack on CIFAR.")
    parser.add_argument('--eval_freq', default=5, type=int,
                        help="Evaluate and save model every eval_freq epochs.")
    parser.add_argument('--num_test_samples', default=None, type=int,
                        help="The number of samples used to train and test the attacker.")
    parser.add_argument('--num_eval_samples', default=None, type=int,
                        help="The number of samples used to train and test the attacker.")
    # Bells
    parser.add_argument("--wandb", action="store_true", default=False, help='Use wandb for logging')
    parser.add_argument('--model_path', type=str, default="mnist_cnn.pt",
                        help='where to save/load')
    parser.add_argument('--namestr', type=str, default='NoBox', \
            help='additional info in output filename to describe experiments')
    parser.add_argument('--dir_test_models', type=str, default="./dir_test_models",
                        help="The path to the directory containing the classifier models for evaluation.")
    parser.add_argument('--robust_model_path', type=str,
                        default="./madry_challenge_models/mnist/adv_trained/mnist_lenet5_advtrained.pt",
                        help="The path to our adv robust classifier")
    parser.add_argument('--robust_sample_prob', type=float, default=1e-1, metavar='N',
                        help='1-P(robust)')
    #parser.add_argument('--madry_model_path', type=str, default="./madry_challenge_models",
    #                    help="The path to the directory containing madrys classifiers for testing")
    parser.add_argument("--max_test_model", type=int, default=1,
                    help="The maximum number of pretrained classifiers to use for testing.")
    parser.add_argument("--perturb_magnitude", type=float, default=None,
                        help="The amount of perturbation we want to enforce with lagrangian.")
    parser.add_argument("--log_path", type=str, default="./logs",
                        help="Where to save logs if logger is specified.")
    parser.add_argument("--save_model", type=str, default=None,
                            help="Where to save the models, if it is specified.")
    parser.add_argument("--fixed_testset", action="store_true",
                            help="If used then makes sure that the same set of samples is always used for testing.")
    parser.add_argument('--normalize', default=None, choices=(None, "default", "meanstd"))

    ###
    parser.add_argument('--source_arch', default="res18",
                        help="The architecture we want to attack on CIFAR.")
    parser.add_argument('--target_arch', nargs='*',
                        help="The architecture we want to blackbox transfer to on CIFAR.")
    parser.add_argument('--ensemble_adv_trained', action='store_true')
    parser.add_argument('--adv_models', nargs='*', help='path to adv model(s)')
    parser.add_argument('--type', type=int, default=0, help='Model type (default: 0)')
    parser.add_argument('--model_name', help='path to model')
    parser.add_argument('--transfer', action='store_true')
    parser.add_argument('--command', choices=("eval", "train"), default="train")
    parser.add_argument('--split', type=int, default=None,
                        help="Which subsplit to use.")
    parser.add_argument('--path_to_data', default="../data", type=str)
    args = parser.parse_args()

    args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    normalize = None
    if args.normalize == "meanstd":
        normalize = transforms.Normalize(cf.mean["cifar10"], cf.std["cifar10"])
    elif args.normalize == "default":
        normalize = CIFAR_NORMALIZATION

    train_loader, test_loader, split_train_loader, split_test_loader = create_loaders(args,
            root=args.path_to_data, split=args.split, num_test_samples=args.num_test_samples, normalize=normalize)

    if args.split is not None:
        train_loader = split_train_loader
        test_loader = split_test_loader

    if os.path.isfile("../settings.json"):
        with open('../settings.json') as f:
            data = json.load(f)
        args.wandb_apikey = data.get("wandbapikey")

    if args.wandb:
        os.environ['WANDB_API_KEY'] = args.wandb_apikey
        wandb.init(project='NoBox-table2',
                   name='NoBox-Attack-{}-{}'.format(args.dataset, args.namestr))

    model, adv_models, l_test_classif_paths, args.model_type = data_and_model_setup(args, no_box_attack=True)
    model.to(args.dev)
    model.eval()

    print("Testing on %d Test Classifiers with Source Model %s" %(len(l_test_classif_paths), args.source_arch))
    x_test, y_test = load_data(args, test_loader)

    if args.dataset == "mnist":
        critic = load_unk_model(args)
    elif args.dataset == "cifar":
        name = args.source_arch
        if args.source_arch == "adv":
            name = "res18"
        critic = load_unk_model(args, name=name)

    misclassify_loss_func = kwargs_attack_loss[args.attack_loss]
    attacker = NoBoxAttack(critic, misclassify_loss_func, args)

    print("Evaluating clean error rate:")
    list_model = [args.source_arch]
    if args.source_arch == "adv":
        list_model = [args.model_type]
    if args.target_arch is not None:
        list_model = args.target_arch

    for model_type in list_model:
        num_samples = args.num_eval_samples
        if num_samples is None:
            num_samples = len(test_loader.dataset)

        eval_loader = torch.utils.data.Subset(test_loader.dataset,
                         np.random.randint(len(test_loader.dataset), size=(num_samples,)))
        eval_loader = torch.utils.data.DataLoader(eval_loader, batch_size=args.test_batch_size)
        baseline_transfer(args, None, "Clean", model_type, eval_loader,
            list_classifiers=l_test_classif_paths)

    def eval_fn(model):
        advcorrect = 0
        model.to(args.dev)
        model.eval()
        with ctx_noparamgrad_and_eval(model):
            if args.source_arch == 'googlenet':
                adv_complete_list = []
                for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
                    if (batch_idx + 1) * args.test_batch_size > args.batch_size:
                        break
                    x_batch, y_batch = x_batch.to(args.dev), y_batch.to(args.dev)
                    adv_complete_list.append(attacker.perturb(x_batch, target=y_batch))
                adv_complete = torch.cat(adv_complete_list)
            else:
                adv_complete = attacker.perturb(x_test[:args.batch_size],
                                        target=y_test[:args.batch_size])
            adv_complete = torch.clamp(adv_complete, min=0., max=1.0)
            output = model(adv_complete)
            pred = output.max(1, keepdim=True)[1]
            advcorrect += pred.eq(y_test[:args.batch_size].view_as(pred)).sum().item()
            fool_rate = 1 - advcorrect / float(args.batch_size)
            print('Test set base model fool rate: %f' %(fool_rate))
        model.cpu()

        if args.transfer:
            adv_img_list = []
            y_orig = y_test[:args.batch_size]
            for i in range(0, len(adv_complete)):
                adv_img_list.append([adv_complete[i].unsqueeze(0), y_orig[i]])
            # Free memory
            del model
            torch.cuda.empty_cache()
            baseline_transfer(args, attacker, "AEG", model_type,
                            adv_img_list, l_test_classif_paths, adv_models)


    if args.command == "eval":
        attacker.load(args)
    elif args.command == "train":
        attacker.train(train_loader, test_loader, adv_models,
                l_test_classif_paths, l_train_classif={"source_model": model},
                eval_fn=eval_fn)

if __name__ == '__main__':
    main()
