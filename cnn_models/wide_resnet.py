import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropout_rate=0, num_classes=10, *args, **kwargs):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        #print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def train(args, logger=None):
    from utils.utils import create_loaders, seed_everything, CIFAR_NORMALIZATION
    import utils.config as cf
    import os
    import torch.backends.cudnn as cudnn
    import time

    seed_everything(args.seed)

    normalize = None
    if args.normalize == "meanstd":
        from torchvision import transforms
        normalize = transforms.Normalize(cf.mean["cifar10"], cf.std["cifar10"])
    elif args.normalize == "default":
        normalize = CIFAR_NORMALIZATION

    # Hyper Parameter settings
    use_cuda = torch.cuda.is_available()
    best_acc = 0
    start_epoch, num_epochs = cf.start_epoch, cf.num_epochs

    # Data Uplaod
    trainloader, testloader = create_loaders(args, augment=not args.no_augment, normalize=normalize)

    # Model
    print('\n[Phase 2] : Model setup')
    net = Wide_ResNet(**vars(args))
    file_name = os.path.join(args.output, "%s/%s/model_%i.pt" % (args.dataset, "wide_resnet", args.seed))
    net.apply(conv_init)

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "adam":
        from torch.optim import Adam
        optimizer = Adam(net.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        from torch.optim import SGD
        optimizer = None
    elif args.optimizer == "sls":
        from utils.sls import Sls
        n_batches_per_epoch = len(trainloader)
        print(n_batches_per_epoch)
        optimizer = Sls(net.parameters(), n_batches_per_epoch=n_batches_per_epoch)
    else:
        raise ValueError("Only supports adam or sgd for optimizer.")

    # Training
    def train(epoch, optimizer=None):
        net.train()
        net.training = True
        train_loss = 0
        correct = 0
        total = 0
        if args.optimizer == "sgd":
            optimizer = SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)

        print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)               # Forward Propagation
            loss = criterion(outputs, targets)  # Loss

            if args.optimizer == "sls":
                def closure():
                    output = net(inputs)
                    loss = criterion(output, targets)
                    return loss
                optimizer.step(closure)
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                    %(epoch, num_epochs, batch_idx+1,
                        len(trainloader), loss.item(), 100.*correct/total))
            sys.stdout.flush()

            if logger is not None:
                logger.write(dict(train_accuracy=100. * correct / total, loss=loss.item()), epoch)

    def test(epoch, best_acc=0):
        net.eval()
        net.training = False
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

            # Save checkpoint when best model
            acc = 100.*correct/total
            if logger is None:
                print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))
            else:
                logger.write(dict(test_loss=loss.item(), test_accuracy=acc), epoch)
            
            if acc > best_acc:
                print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
                state = {
                        'net':net.module if use_cuda else net,
                        'acc':acc,
                        'epoch':epoch,
                }
                dirname = os.path.dirname(file_name)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                torch.save(net.state_dict(), file_name)
                best_acc = acc
        return best_acc

    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(num_epochs))
    print('| Initial Learning Rate = ' + str(args.lr))

    elapsed_time = 0
    for epoch in range(start_epoch, start_epoch+num_epochs):
        start_time = time.time()

        train(epoch, optimizer)
        best_acc = test(epoch, best_acc)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

    print('\n[Phase 4] : Testing model')
    print('* Test results : Acc@1 = %.2f%%' %(best_acc))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add_argument('output', help='Output path wehre to save the different pretrained models.')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--batch_size', type=int, default=128, metavar='S', help='Batch size')
    parser.add_argument('--test_batch_size', type=int, default=512, metavar='S',
                            help='Test Batch size')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-lr', default=1e-1, type=float)
    parser.add_argument('--train_set', default='train',
                            choices=['train_and_test','test','train'],
                            help='add the test set in the training set')
    parser.add_argument('--no_augment', action="store_false")
    parser.add_argument('--optimizer',  default="sgd", choices=("sgd", "adam", "sls"))
    parser.add_argument('--normalize',  default="meanstd", choices=("none", "default", "meanstd"))
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    
    args = parser.parse_args()
    
    train(args)