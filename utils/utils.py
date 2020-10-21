import os
import wandb
import sys
import torch
import torchvision
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.models import resnet50
import torchvision.utils as vutils
from torchvision.utils import save_image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
from random import randint
from PIL import Image
from cnn_models import *
from cnn_models import LeNet as Net
from cnn_models.mnist_ensemble_adv_train_models import *
from models.dcgan28 import DiscriminatorCNN28
import ipdb
import numpy as np
import random
import utils.config as cf
from utils.sls import Sls
from torch.utils.data import Subset, ConcatDataset
from utils.dataset_split import create_dataset_split

CIFAR_NORMALIZATION = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

''' Set Random Seed '''
def seed_everything(seed):
    if seed:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def project_name(dataset_name):
    if dataset_name:
        return "NoBox-{}".format(dataset_name)
    else:
        return "NoBox"

def reduce_sum(x, keepdim=True):
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    return x.squeeze().sum()

def L2_dist(x, y):
    return torch.mean(torch.norm(x - y,p=2,dim=(1,2,3)))

def L2_norm_dist(x, y):
    dist = torch.norm(x - y, p=2,dim=(1,2,3))
    return dist

def Linf_dist(x, y):
    dist = torch.norm(x - y, float('inf'),dim=(1,2,3))
    return dist

class Normalize(nn.Module):
    """
    Normalize an image as part of a torch nn.Module
    """
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        num = (x - self.mean.type_as(x)[None,:,None,None])
        denom = self.std.type_as(x)[None,:,None,None]
        return num / denom

def to_cuda(model):
    cuda_stat = torch.cuda.is_available()
    if cuda_stat:
        model = torch.nn.DataParallel(model,\
                device_ids=range(torch.cuda.device_count())).cuda()
    return model

def tensor_to_cuda(x):
    cuda_stat = torch.cuda.is_available()
    if cuda_stat:
        x = x.cuda()
    return x

def display_tensor(tensor):
    plt.imshow((tensor)[0].detach().numpy().transpose(1,2,0))
    plt.show()

def save_image_to_wandb(args,image,name,normalize):
    batch_size,nc,h,w = image.shape
    image, image_reshaped = to_img(image.cpu().data,nc,h,w)
    save_image(image, name, normalize=normalize)
    return image_reshaped.detach().cpu().numpy()

def load_imagenet_classes():
    with open("references/adver_robust/introduction/imagenet_class_index.json") as f:
        imagenet_classes = {int(i):x[1] for i,x in json.load(f).items()}
    return imagenet_classes

def get_single_data(args):
    """
    Data loader. For now, just a test sample
    """
    assert args.split is None
    if args.dataset == 'mnist':
        trainloader, testloader = load_mnist(augment=False)
        tensor,target = trainloader.dataset[randint(1,\
            100)]
        tensor = tensor_to_cuda(tensor.unsqueeze(0))
        target = tensor_to_cuda(target.unsqueeze(0))
        args.classes = 10
    elif args.dataset=='cifar':
        trainloader, testloader = load_cifar(args,augment=True)
        tensor,target = trainloader.dataset[randint(1,\
            100)]
        tensor = tensor_to_cuda(tensor.unsqueeze(0))
        target = tensor_to_cuda(target.unsqueeze(0))
        args.classes = 10
    else:
        pig_img = Image.open("references/adver_robust/introduction/pig.jpg")
        preprocess = transforms.Compose([
           transforms.Resize(224),
           transforms.ToTensor(),
        ])
        tensor = tensor_to_cuda(preprocess(pig_img)[None,:,:,:])
        source_class = 341 # pig class
        target = tensor_to_cuda(torch.LongTensor([source_class]))
        args.classes = 1000

    # Get flat input size
    args.input_size = tensor[0][0].flatten().shape[0]
    return tensor, target

def create_loaders(args, augment=True, normalize=None, root='./data', num_test_samples=None, split=None):
    """
    Data loader. For now, just a test sample
    """
    if args.dataset == 'mnist':
        # Normalize image for MNIST
        # normalize = Normalize(mean=(0.1307,), std=(0.3081,))
        normalize = None
        if args.split is None:
            trainloader, testloader = load_mnist(args, augment=False,
                    root=root, num_test_samples=args.num_test_samples)
        else:
            trainloader, testloader, s_train_loader, s_test_loader = load_mnist(args, augment=False,
                    root=root, num_test_samples=args.num_test_samples,
                    split=split)
        args.classes = 10
        args.input_size = 784
        args.nc, args.h, args.w = 1,28,28
        args.clip_min = 0.
        args.clip_max = 1.
    elif args.dataset=='cifar':
        #normalize=Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
        args.input_size = 32*32*3
        if args.split is None:
            trainloader, testloader = load_cifar(args, augment=False,
                    normalize=normalize, root=root,
                    num_test_samples=num_test_samples)
        else:
            trainloader, testloader, s_train_loader, s_test_loader = load_cifar(args, augment=False,
                    normalize=normalize, root=root,
                    num_test_samples=num_test_samples, split=split)
        args.classes = 10
        args.nc, args.h, args.w = 3,32,32
        args.clip_min = -1.
        args.clip_max = 1.
    else:
        # Normalize image for ImageNet
        # normalize=utils.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        # args.input_size = 150528
        raise NotImplementedError
    if split is None:
        return trainloader, testloader, None, None
    else:
        return trainloader, testloader, s_train_loader, s_test_loader


def load_unk_model(args, filename=None, madry_model=False, name="VGG16"):
    """
    Load an unknown model. Used for convenience to easily swap unk model
    """
    # First, check if target model is specified in args
    if args.train_with_critic_path is not None:
        path = args.train_with_critic_path
        if not os.path.isdir(path):
            msg = "You passed arg `train_with_critic_path` with path "
            msg += path + " which is not a valid dir"
            sys.exit(msg)

        if "madry" in path:
            try:
                model = args.madry_model[path]
            except:
                from classifiers.madry_challenge.madry_et_al_utils import get_madry_et_al_tf_model
                model = get_madry_et_al_tf_model(args.dataset, path)
                # Hack for now, we cannot load this model twice (tensorflow), so
                # store pointer
                args.madry_model = {path: model}

    elif args.dataset == 'mnist':
        # First case is to laod the Madry model
        if (
                (filename is None) and (args.attack_type == 'nobox') and (args.fixed_critic)
           ):
            # Super hacky, check if Madry model already loaded
            print('loading Madry model')
            try:
                model = args.madry_model
            except:
                from classifiers.madry_challenge.madry_et_al_utils import get_madry_et_al_tf_model
                path = os.path.join(args.dir_test_models, "madry_challenge_models", args.dataset, "adv_trained")
                model = get_madry_et_al_tf_model(args.dataset, path)
                # Hack for now, we cannot load this model twice (tensorflow), so
                # store pointer
                args.madry_model = model

        # Generic classifier

        elif (filename is None) and (args.attack_type == 'nobox') or  (args.source_arch == 'ens_adv'):
            print('Loading generic classifier')
            if args.source_arch == 'ens_adv' or args.ensemble_adv_trained:
                model = model_mnist(type=args.type)
                # model = load_model(args, args.model_name, type=args.type)
            else:
                model = MadryLeNet(args.nc, args.h, args.w).to(args.dev)

        elif name == 'PGD Adversarial Training':
            from classifiers.madry_challenge.madry_et_al_utils import get_madry_et_al_tf_model
            model = get_madry_et_al_tf_model(args.dataset, filename)

        else:
            if filename is None:
                filename = "saved_models/mnist_cnn.pt"
            if os.path.exists(filename):
                # very hacky, it fail if the architecture is not the correct one
                try:
                    model = Net(args.nc, args.h, args.w).to(args.dev)
                    model.load_state_dict(torch.load(filename))
                    model.eval()
                except:
                    model = MadryLeNet(args.nc, args.h, args.w).to(args.dev)
                    model.load_state_dict(torch.load(filename))
                    model.eval()
            else:
                print("training a model from scratch")
                model = main_mnist(args, filename=filename)

    elif args.dataset == 'cifar':
        if name == 'PGD Adversarial Training':
            from classifiers.madry_challenge.madry_et_al_utils import get_madry_et_al_tf_model
            model = get_madry_et_al_tf_model(args.dataset, filename)
        else:
            init_func, _ = ARCHITECTURES[name]
            if (filename is None) and (args.attack_type == 'nobox'):
                model = init_func().to(args.dev)
            else:
                if (filename is None) or (not os.path.exists(filename)):
                    model = main_cifar(args, name=name)
                else:
                    # model = DenseNet121().to(args.dev)
                    model = init_func().to(args.dev)
                    model = nn.DataParallel(model)
                    #print(filename, model, init_func, name)
                    model.load_state_dict(torch.load(filename))
                    model.eval()

    else:
        # load pre-trained ResNet50
        model = resnet50(pretrained=True).to(args.dev)

    model = nn.DataParallel(model)

    return model


def main_mnist(args, filename, lr=1e-3, num_epochs=11, logger=None, split=None):
    if filename is None:
        filename = os.path.join('./saved_models/', "mnist_%s.pt" % name)
    train_loader, test_loader = create_loaders(args, augment=False, split=split)
    model = Net(args.nc, args.h, args.w).to(args.dev)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, num_epochs):
        train_classifier(args, model, args.dev, train_loader, optimizer, epoch, logger)
        test_classifier(args, model, args.dev, test_loader, epoch, logger)

    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    torch.save(model.state_dict(), filename)
    return model


ARCHITECTURES = {
            'VGG16': (VGG, 50),
            'res18': (resnet.ResNet18, 500),
            'dense121': (densenet.densenet_cifar,  500),
            'googlenet': (googlenet.GoogLeNet, 500),
            'lenet': (LeNet, 250),
            'wide_resnet': (wide_resnet.Wide_ResNet, None),
            'VGG16_ens': (VGG, 50),
            'res18_ens': (resnet.ResNet18, 500),
            'dense121_ens': (densenet.densenet_cifar,  500),
            'googlenet_ens': (googlenet.GoogLeNet, 500),
            'wide_resnet_ens': (wide_resnet.Wide_ResNet, None)
}

def train_cifar(args, name="VGG16", augment=True, normalize=None,
        filename=None, lr=1e-4, num_epochs=100, logger=None, optimizer="adam",
        i =0):
    if filename is None:
        filename = os.path.join('./pretrained_classifiers/cifar/', "%s/" %
                name, 'model_%s.pt' % i)

    init_func, _ = ARCHITECTURES[name]

    print("Training %s" % (name))
    model = init_func().to(args.dev)
    if name == "wide_resnet":
        model.apply(wide_resnet.conv_init)
    model = nn.DataParallel(model)

    train_loader, test_loader, split_train_loader, split_test_loader = create_loaders(args,
                                      split=args.split, augment=augment, normalize=normalize)

    if args.split is not None:
        train_loader = split_train_loader
        test_loader = split_test_loader

    if optimizer == "adam":
        _optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        _optimizer = optim.SGD(model.parameters(), lr=cf.learning_rate(lr, num_epochs), momentum=0.9, weight_decay=5e-4)
    elif optimizer == "sls":

        n_batches_per_epoch = len(train_loader)/float(train_loader.batch_size)
        _optimizer = Sls(model.parameters(), n_batches_per_epoch=n_batches_per_epoch)
    else:
        raise ValueError("Only supports adam or sgd for optimizer.")

    best_acc = 0
    for epoch in range(1, num_epochs):
        if optimizer == "sgd":
            _optimizer = optim.SGD(model.parameters(), lr=cf.learning_rate(lr, num_epochs), momentum=0.9, weight_decay=5e-4)

        train_classifier(args, model, args.dev, train_loader, _optimizer, epoch, logger=logger)
        acc = test_classifier(args, model, args.dev, test_loader, epoch, logger=logger)

        if acc > best_acc:
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            torch.save(model.state_dict(), filename)
            best_acc = acc

    return model


def main_cifar(args, augment=True):
    for name in ARCHITECTURES:
        model = train_cifar(args, name=name, augment=augment)
    return model


def train_classifier(args, model, device, train_loader, optimizer, epoch, logger=None):
    model.train()
    criterion = nn.CrossEntropyLoss(reduction="mean")
    train_loss = 0
    correct = 0
    total = 0
    early_stop_param = 0.01
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        if isinstance(optimizer, Sls):
            def closure():
                output = model(data)
                loss = criterion(output, target)
                return loss
            optimizer.step(closure)
        else:
            loss.backward()
            optimizer.step()

        train_loss += loss.item()
        running_loss = loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch:d} [{batch_idx * len(data):d}/{len(train_loader.dataset):d} '
                  f'{100. * batch_idx / len(train_loader):.0f}] \tLoss: {loss.item():.6f} | '
                  f'Acc: {100. * correct / total:.3f}')

            if running_loss < early_stop_param:
                print("Early Stopping !!!!!!!!!!")
                break
            running_loss = 0.0

    if logger is not None:
        logger.write(dict(train_accuracy=100. * correct / total, loss=loss.item()), epoch)

def test_classifier(args, model, device, test_loader, epoch, logger=None):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            # sum up batch loss
            test_loss += loss.item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    acc = 100. * correct / len(test_loader.dataset)

    if logger is None:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'\
            .format(test_loss, correct, len(test_loader.dataset), acc))
    else:
        logger.write(dict(test_loss=test_loss, test_accuracy=acc), epoch)

    return acc

def load_cifar(args, augment=False, normalize=None, root='./data', num_test_samples=None, split=None):
    """
    Load and normalize the training and test data for CIFAR10
    """
    torch.multiprocessing.set_sharing_strategy('file_system')
    print('==> Preparing data..')

    if augment:
        transform_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]
    else:
        transform_train = [transforms.ToTensor()]

    transform_test = [transforms.ToTensor()]

    if normalize is not None:
        transform_train.append(normalize)
        transform_test.append(normalize)

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    if split is None:
        train_set = torchvision.datasets.CIFAR10(root=root, train=True,
                                    download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root=root, train=False,
                                      download=True, transform=transform_test)
    else:
        dataset_split = create_dataset_split(args, root=root, num_splits=6, augment=augment)
        assert split < len(dataset_split)
        train_splits, test_splits = [], []
        for i in range(0,len(dataset_split)):
            dataset = dataset_split[i]
            train_set, test_set = dataset["train"], dataset["test"]
            train_splits.append(train_set)
            test_splits.append(test_set)

        split_train_dataset = ConcatDataset(train_splits)
        split_test_dataset = ConcatDataset(test_splits)

        if num_test_samples is not None:
            generator = None
            indices = torch.randint(len(split_test_dataset), size=(num_test_samples,), generator=generator)
            if args.fixed_testset is True:
                generator = torch.random.manual_seed(1234)
                indices = torch.arange(num_test_samples)
            split_test_dataset = Subset(split_test_dataset, indices)

        split_train_loader = torch.utils.data.DataLoader(split_train_dataset,
                batch_size=args.batch_size, shuffle=True, pin_memory=True)
        split_test_loader = torch.utils.data.DataLoader(split_test_dataset,
                batch_size=args.batch_size, shuffle=False, pin_memory=True)

        dataset = dataset_split[split]
        train_set, test_set = dataset["train"], dataset["test"]

    if num_test_samples is not None:
        generator = None
        indices = torch.randint(len(test_set), size=(num_test_samples,), generator=generator)
        if args.fixed_testset is True:
            generator = torch.random.manual_seed(1234)
            indices = torch.arange(num_test_samples)
        test_set = Subset(test_set, indices)

    # if args.train_set == 'test':
    #     train_set = test_set
    # elif args.train_set == 'train_and_test':
    #     train_set = torch.utils.data.ConcatDataset([train_set,test_set])

    trainloader = torch.utils.data.DataLoader(train_set,batch_size=args.batch_size,
                                                shuffle=True, num_workers=0,
                                                pin_memory=True)

    testloader = torch.utils.data.DataLoader(test_set,batch_size=args.test_batch_size,
                                                 shuffle=False, num_workers=0,
                                                 pin_memory=True)

    if split is None:
        return trainloader, testloader
    else:
        return trainloader, testloader, split_train_loader, split_test_loader

    return trainloader, testloader

def load_mnist(args, augment=True, root='./data', num_test_samples=None, split=None):
    """
    Load and normalize the training and test data for MNIST
    """
    print('==> Preparing data..')
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    if augment:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        mnist_transforms = transforms.Compose([
                           transforms.ToTensor(),
                       ])

    if split is None:
        train_set = datasets.MNIST(root=root, train=True, download=True,
                       transform=mnist_transforms)
        test_set = datasets.MNIST(root=root, train=False, transform=mnist_transforms)
    else:
        dataset_split = create_dataset_split(args, root=root, num_splits=7,
                transform=mnist_transforms, augment=augment)
        assert split < len(dataset_split)
        train_splits, test_splits = [], []
        for i in range(0,7):
            dataset = dataset_split[i]
            train_set, test_set = dataset["train"], dataset["test"]
            train_splits.append(train_set)
            test_splits.append(test_set)

        split_train_dataset = ConcatDataset(train_splits)
        split_test_dataset = ConcatDataset(test_splits)

        if num_test_samples is not None:
            generator = None
            indices = torch.randint(len(split_test_dataset), size=(num_test_samples,), generator=generator)
            if args.fixed_testset is True:
                generator = torch.random.manual_seed(1234)
                indices = torch.arange(num_test_samples)
            split_test_dataset = Subset(split_test_dataset, indices)

        split_train_loader = torch.utils.data.DataLoader(split_train_dataset,
                batch_size=args.batch_size, shuffle=True)
        split_test_loader = torch.utils.data.DataLoader(split_test_dataset,
                batch_size=args.batch_size, shuffle=False)

        dataset = dataset_split[split]
        train_set, test_set = dataset["train"], dataset["test"]

    if num_test_samples is not None:
        generator = None
        indices = torch.randint(len(test_set), size=(num_test_samples,), generator=generator)
        if args.fixed_testset is True:
            generator = torch.random.manual_seed(1234)
            indices = torch.arange(num_test_samples)
        test_set = Subset(test_set, indices)

    # if args.train_set == 'test':
    #     train_set = test_set
    # elif args.train_set == 'train_and_test':
    #     train_set = torch.utils.data.ConcatDataset([train_set, test_set])

    trainloader = torch.utils.data.DataLoader(train_set,
                               batch_size=args.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set,
                batch_size=args.batch_size, shuffle=False)
    if split is None:
        return trainloader, testloader
    else:
        return trainloader, testloader, split_train_loader, split_test_loader

def to_img(x,nc,h,w):
    # x = x.clamp(0, 1)
    x = x.view(x.size(0), nc, h, w)
    x_reshaped = x.permute(0,2,3,1)
    return x, x_reshaped

def freeze_model(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False
    return model

def nobox_wandb(args, epoch, x, target, adv_inputs, adv_out, adv_correct,
                clean_correct, loss_misclassify, loss_model, loss_perturb,
                loss_gen, train_loader):
    n_imgs = min(30, len(x))
    clean_image = (x)[:n_imgs].detach()
    adver_image = (adv_inputs)[:n_imgs].detach()
    if args.dataset == 'cifar':
        factor = 10.
    else:
        factor = 1.
    delta_image = factor*(adver_image - clean_image)
    file_base = "adv_images/train/" + args.namestr + "/"
    if not os.path.exists(file_base):
        os.makedirs(file_base)
    # import pdb; pdb.set_trace()
    img2log_clean = save_image_to_wandb(args, clean_image, file_base + "clean.png", normalize=True)
    img2log_adver = save_image_to_wandb(args, adver_image, file_base + "adver.png", normalize=True)
    img2log_delta = save_image_to_wandb(args, delta_image, file_base + "delta.png", normalize=True)
    adv_acc = 100. * adv_correct.cpu().numpy() / len(train_loader.dataset)
    clean_acc = 100. * clean_correct.cpu().numpy() / len(train_loader.dataset)
    wandb.log({"Adv. Train Accuracy": adv_acc,
               "Clean Train Accuracy": clean_acc,
               "Misclassification Loss": loss_misclassify.item(),
               "Critic Loss": loss_model.item(),
               "Perturb Loss": loss_perturb.item(), "x": epoch,
               'Gen Loss': loss_gen.item(),
               'Train_Clean_image': [wandb.Image(img, caption="Train Clean") for img in img2log_clean],
               'Train_Adver_image': [wandb.Image(img, caption="Train Adv, "f"Label: {target[i]}"
                                                 f" Predicted: {adv_out[i].item()}")
                                     for i, img in enumerate(img2log_adver)],
               'Train_Delta_image': [wandb.Image(img, caption="Train Delta") for img in img2log_delta]
               })

def concat_dataset(args, loader_1, loader_2):
    dataset_1 = loader_1.datset
    dataset_2 = loader_2.dataset
    dataset_tot = torch.utils.data.ConcatDataset([dataset_1, dataset_2])
    return torch.utils.data.DataLoader(dataset_tot,
                                       batch_size=args.batch_size, shuffle=True)


kwargs_perturb_loss = {'Linf': Linf_dist, 'L2': L2_dist}
