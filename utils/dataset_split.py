from torchvision import  datasets, transforms
import torch
import os
import argparse
import ipdb
from numpy.random import default_rng
from torch._utils import _accumulate
from torch.utils.data import Subset


default_generator = default_rng()


def random_split(dataset, lengths, generator=default_generator):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = generator.permutation(sum(lengths)).tolist()
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]


def create_dataset_split(args, root='./data', num_splits=7,
        train_split=0.8, generator=default_generator,
        partition="train_and_test", augment=False, transform=None,
        normalize=None):

    if args.dataset == "mnist":
        name = "MNIST"
    else:
        name = "cifar-split"

    path_to_split = os.path.join(root, "%s/split_%i/data.pt"%(name,num_splits))
    if os.path.exists(path_to_split):
        print("Loading %i splits of the %s dataset..."%(num_splits, args.dataset))
        list_split_dataset = torch.load(path_to_split)
    else:
        print("Split_%i dataset for %s does not exist. Creating a %i split of the dataset..."%(num_splits, args.dataset, num_splits))
        if args.dataset == "mnist":
            test_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            if augment:
                mnist_transforms = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            elif transform is not None:
                mnist_transforms = transforms
            else:
                mnist_transforms = transforms.Compose([
                                   transforms.ToTensor(),
                               ])

            trainset = datasets.MNIST(root=root, train=True, download=True,
                               transform=mnist_transforms)
            testset = datasets.MNIST(root=root, train=False, transform=mnist_transforms)

        elif args.dataset == "cifar":
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

            trainset = datasets.CIFAR10(root=root, train=True,
                                            download=True, transform=transform_train)
            testset = datasets.CIFAR10(root=root, train=False,
                                              download=True, transform=transform_test)

        else:
            raise ValueError()

        if partition == "train_and_test":
            dataset = torch.utils.data.ConcatDataset([trainset, testset])
        elif partition == "train":
            dataset = trainset
        elif partition == "test":
            dataset == testset
        else:
            raise ValueError()

        list_splits = [len(dataset)//num_splits]*num_splits + [len(dataset)%num_splits]
        split_dataset = random_split(dataset, list_splits, generator=generator)[:-1]
        #split_dataset = torch.utils.data.random_split(dataset, list_splits)[:-1]

        list_split_dataset = []
        for dataset in split_dataset:
            train_size = int(len(dataset)*train_split)
            list_splits = [train_size, len(dataset)-train_size]
            split = random_split(dataset, list_splits, generator=generator)
            #split = torch.utils.data.random_split(dataset, list_splits)
            list_split_dataset.append({"train": split[0], "test": split[1]})

        dirname = os.path.dirname(path_to_split)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(list_split_dataset, path_to_split)

    return list_split_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_dataset', default="./data", type=str)
    parser.add_argument('--dataset', default="mnist", choices=("mnist", "cifar"))
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--num_splits', default=7, type=int)
    args = parser.parse_args()

    generator = default_rng(args.seed)
    split_dataset = create_dataset_split(args, root=args.path_to_dataset, num_splits=args.num_splits,
                                         generator=generator, augment=False)
    print(len(split_dataset))
    print(split_dataset[0])
    print(len(split_dataset[0]["train"]))
