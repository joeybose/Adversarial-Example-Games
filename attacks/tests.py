import wandb
from tqdm import tqdm
from attack_helpers import *
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from utils.utils import load_unk_model
from advertorch.attacks import LinfPGDAttack, L2PGDAttack
kwargs_PGD_attack = {'Linf': LinfPGDAttack, 'L2': L2PGDAttack}

class Test_generator(object):
    def __init__(self, args, test_loader, dir_test_model, model_type, G, nc=1, h=28, w=28):
        self.args=args
        self.max_test_model=args.max_test_model
        self.test_loader=test_loader
        if os.path.exists(dir_test_model):
            self.dir_test_model=dir_test_model
        else:
            raise ValueError(f"The path to models should contain a {model_type} subfolder")
        self.model_type=model_type
        self.G=G
        self.nc=nc
        self.h=h
        self.w=w
        self.PGDAttack = kwargs_PGD_attack[args.attack_ball]

    def test(self):
        print(f'Testing against {self.model_type} model(s)')
        dir_test_model_type = os.path.join(dir_test_model, model_type)
        if os.path.exists(dir_test_model_type):
            list_dir = os.listdir(dir_test_model_type)
            num_test_model = self.max_test_model
            if (num_test_model is None) or (num_test_model > len(list_dir)):
                num_test_model = len(list_dir)
            for i in range(num_test_model):
                filename = os.path.join(dir_test_model, list_dir[i])
                print(f"Test on {filename}")
                model = load_unk_model(args, filename)
                if self.args.attack_loss == 'carlini':
                    raise NotImplementedError('test ton carlini loss not implemented')
                whitebox_pgd(self.args, self.model, self.test_loader)
                L2_test_model(self.args, self.test_loader, self.model, self.G,
                              self.nc, self.h, self.w, mode="Test")
                del model


class Madry_test_generator(Test_generator):
    def __init__(self, args, test_loader, dir_test_model, model_type, G, nc=1, h=28, w=28):
        super(Madry_test_generator, self).__init__(args, test_loader,
              dir_test_model, model_type, G, nc, h, w)
        self.model=None
        self.dataset=args.dataset

    def test(self):
        print(f'Testing against {self.model_type} Madry model(s)')
        dir_test_model = os.path.join(self.dir_test_model,self.model_type)
        if self.model is None:
            print(dir_test_model)
            self.model = get_madry_et_al_tf_model(self.dataset, dir_test_model)
        # whitebox_pgd(args, model, test_loader)
        # accuracy is 94% on the trained model
        L2_test_model(self.args, self.test_loader, self.model,
                      self.G, self.nc, self.h, self.w, mode="Test")




def whitebox_pgd(args, model, test_loader):
    # TODO: implement [-1,1] clipping for CIFAR, implement carlini loss
    PGDAttack = kwargs_PGD_attack[args.attack_ball]
    adversary = PGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
        nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=.0, clip_max=1.0,
        targeted=False)
    correct = 0
    test_itr = tqdm(enumerate(test_loader),
                    total=len(list(test_loader)))
    for _,(data, target) in test_itr:
        x, target = data.to(args.dev), target.to(args.dev)
        adv_image = adversary.perturb(x, target)
        pred = model(adv_image)
        out = pred.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += out.eq(target.unsqueeze(1).data).sum()
    acc = 100. * correct.cpu().numpy() / len(test_loader.dataset)
    print("\nSuccess rate after PGD attack : {}/{} ({:.0f}%), ".format(correct,
                                                                 len(test_loader.dataset),
                                                                 acc.item()))


