# -*- coding: utf-8 -*-
"""
Supplementary code for paper under review by the International Conference on Machine Learning (ICML).
Do not distribute.
"""
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import ipdb

_NOISE_DIM = 128
_H_FILTERS = 64


class Conv2dSN(torch.nn.Conv2d):
    """
    Implements a Convolution Layer with Spectral Normalization (see paper [1]).
    Derives from torch.nn.Conv2d (see docs [2]).
    Note:
        - Use <model_name>.eval() and <model_name>.train() to set the model into
        evaluation and training mode, respectively. In the former mode, the att-
        ributes of this layer (vector 'u' of Algorithm 1 [1]) remain intact.
    Attributes:
        u: [torch.nn.Variable] Variable of tensor u in R^m, where m is the first
        dimension of the parameters (weights) of the layer, i.e. W in R^{mxn}.
        Initialized from normal distribution. See Algorithm 1 [1], vector 'u'.
        w_view: [nn.View] 2D-View of the weights of the layer
        n_iters_power_method: [int] Number of iterations of the power iteration
        method. Default: 1, as in [1].
        & attributes of torch.nn.Conv2d (see [2])
    [1] https://openreview.net/forum?id=B1QRgziT-
    [2] http://pytorch.org/docs/master/_modules/torch/nn/modules/conv.html#Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, n_iters_power_method=1):
        super(Conv2dSN, self).__init__(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups, bias)
        self.w_view = None
        self.u = None
        self.n_iters_power_method = n_iters_power_method

    def forward(self, _input):
        """
        Overwrites the nn.Module.forward method.
        Raises:
            ValueError: _input is not 4D tensor. See conv2d in nn.functional
        :param _input: [Variable] Input Variable of a Tensor
        :return: [Variable]
        """
        if self.w_view is None or self.u is None:
            _cuda = next(self.parameters()).is_cuda
            _dtype = torch.cuda.FloatTensor if _cuda else torch.FloatTensor
            self.w_view = Variable(self.weight.data.view(self.weight.size(0), -1))
            self.u = Variable(_dtype(self.w_view.size(0), 1).normal_(0, 1),
                              requires_grad=False)
        if self.training:
            # step#1, Algorithm 1 [1]: power iteration method
            for i in range(self.n_iters_power_method):
                v = self._normalize_l2(self.w_view.data.t().mm(self.u.data))
                self.u.data = self._normalize_l2(self.w_view.data.mm(v))
            # step#2, Algorithm 1 [1]]: normalize the weights
            sigma = self.u.t().mm(self.w_view).mm(Variable(v, requires_grad=False))
            w_bar = self.weight / sigma
        return nn.functional.conv2d(_input, weight=w_bar, bias=self.bias,
                                    stride=self.stride, padding=self.padding,
                                    dilation=self.dilation, groups=self.groups)

    @staticmethod
    def _normalize_l2(input_tensor):
        """
        Normalizes the input tensor with its l2 norm.
        Note: Operation *not* in-place
        :param input_tensor: [torch.FloatTensor]
        :return: [torch.FloatTensor]
        Raises:
            TypeError: input_tensor is not torch.FloatTensor
        """
        if not isinstance(input_tensor, torch.FloatTensor):
            raise TypeError(f"Unsupported operand type: {type(input_tensor)}. "
                            "Expected torch.FloatTensor.")
        return input_tensor.div((input_tensor.pow(2).sum() ** .5) + 1e-15)


class DiscriminatorCNN28(nn.Module):
    """
    Implements a CNN-based Discriminator Network, of the following layers:
    Sequential (
        (0): Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): LeakyReLU (0.2, inplace)
        (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        (4): LeakyReLU (0.2, inplace)
        (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        (7): LeakyReLU (0.2, inplace)
        (8): Conv2d(256, 1, kernel_size=(3, 3), stride=(1, 1), bias=False)
        (9): Sigmoid ()
    )
    where Conv2d is either nn.Module.Conv2d, or Conv2dSN (see above)
    It represents an adjusted version of the DCGAN architecture, where the
    input is expected to be 1x28x28 and has lower depth.
    See: https://github.com/pytorch/examples/tree/master/dcgan
    Attributes:
        img_channels: [int] number of input channels
        img_size: [int] size of the input space, assuming square input
        n_outputs: [int] number of outputs. Default is 1
        main: [torch.nn.Sequential] Network architecture
    Raises:
        TypeError: unsupported type of input arguments (for constructor)
        ValueError: the constructor input arguments are negative or zero
    """

    def __init__(self, img_channels=1, h_filters=_H_FILTERS,
                 spectral_norm=False, img_size=None, n_outputs=10):
        if any(not isinstance(_arg, int) for _arg in [img_channels, h_filters, n_outputs]):
            raise TypeError("Unsupported operand type. Expected integer.")
        if not isinstance(spectral_norm, bool):
            raise TypeError(f"Unsupported operand type: {type(spectral_norm)}. "
                            "Expected bool.")
        if min([img_channels, h_filters, n_outputs]) <= 0:
            raise ValueError("Expected nonzero positive input arguments for: the "
                             "number of output channels, the dimension of the noise "
                             "vector, as well as the depth of the convolution kernels.")
        super(DiscriminatorCNN28, self).__init__()
        _conv = Conv2dSN if spectral_norm else nn.Conv2d
        self.img_channels = img_channels
        self.img_size = img_size
        self.n_outputs = n_outputs
        self.main = nn.Sequential(
            _conv(img_channels, h_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            _conv(h_filters, h_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(h_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            _conv(h_filters * 2, h_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(h_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            _conv(h_filters * 4, self.n_outputs, 3, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, x):
        """
        Overwrites the nn.Module.forward method.
        Raises:
            ValueError: if the product of the dimensions of the input is in-
            divisible by the product of the dimensions of the expected input
        :param x: [Variable] Input Variable of a Tensor
        :return: [Variable] self.main(x)
        """
        if self.img_channels is not None and self.img_size is not None:
            if numpy.prod(list(x.size())) % (self.img_size ** 2 * self.img_channels) != 0:
                raise ValueError(f"Size mismatch. Input size: {numpy.prod(list(x.size()))}. "
                                 f"Expected input divisible by: {self.noise_dim}")
            x = x.view(-1, self.img_channels, self.img_size, self.img_size)
        x = self.main(x)
        return x.view(-1, self.n_outputs)


class GeneratorCNN28(nn.Module):
    """
    Implements a CNN-based Generator Network, of the following layers:
    Sequential (
        (0): ConvTranspose2d(128, 512, kernel_size=(3, 3), stride=(1, 1), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        (2): ReLU (inplace)
        (3): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        (5): ReLU (inplace)
        (6): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), bias=False)
        (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        (8): ReLU (inplace)
        (9): ConvTranspose2d(128, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (10): Tanh ()
    )
    It represents an adjusted version of the DCGAN architecture [1], where the
    output is 1x28x28 and has lower depth.

    [1] https://github.com/pytorch/examples/tree/master/dcgan
    Attributes:
        noise_dim: [int] Dimension Z of the latent vector in R^{Z}
        main: [torch.nn.Sequential] Network architecture
    Raises:
        TypeError: the constructor input arguments are non-integers
        ValueError: the constructor input arguments are negative or zero
    """

    def __init__(self, device, epsilon=.3, norm = "Linf", img_channels=1, img_dim=784, h_filters=_H_FILTERS):
        if any(not isinstance(_arg, int) for _arg in [img_channels, img_dim, h_filters]):
            raise TypeError("Unsupported operand type. Expected integer.")
        if min([img_channels, img_dim, h_filters]) <= 0:
            raise ValueError("Expected strictly positive input arguments for the "
                             "number of output channels, the dimension of the noise "
                             "vector, as well as the depth of the convolution kernels.")
        super(GeneratorCNN28, self).__init__()
        self.device = device
        self.epsilon = epsilon
        self.norm = norm
        self.img_dim = img_dim
        self.fcx = nn.Linear(img_dim,2*img_dim)
        self.fcz = nn.Linear(10,100)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(2*img_dim+100, h_filters * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(_H_FILTERS * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(h_filters * 8, h_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(_H_FILTERS * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(h_filters * 4, h_filters * 2, 4, 2, 0, bias=False),
            nn.BatchNorm2d(_H_FILTERS * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(h_filters * 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Overwrites the nn.Module.forward method.
        Raises:
            ValueError: if the product of the dimensions of the input is
            indivisible by self.noise_dim
        :param x: [Variable] Input Variable of a Tensor
        :return: [Variable] self.main(x)
        """
        if numpy.prod(list(x.size())) % self.img_dim != 0:
            raise ValueError(f"Size mismatch. Input size: {numpy.prod(list(x.size()))}. "
                             f"Expected input divisible by: {self.img_dim}")
        x=x.view(-1, self.img_dim)
        # z = F.relu(self.fcz(torch.randn_like(x))).view(-1,self.img_dim,1,1)
        batch_size = x.shape[0]
        y = torch.randint(0, 10, (batch_size, 1))
        # # One hot encoding buffer that you create out of the loop and just keep reusing
        y_onehot = torch.zeros(batch_size, 10)
        z = y_onehot.scatter_(1, y, 1).to(self.device)
        # z = torch.randn((batch_size,10)),dim=1).to(self.device)
        z = F.relu(self.fcz(z)).view(-1,100,1,1)
        h = F.relu(self.fcx(x)).view(-1,2*self.img_dim,1,1)
        h = torch.cat((h,z),1)
        out = .5*(self.main(h) + 1.)
        delta = out.view(-1,self.img_dim) - x
        if self.norm == "Linf":
            delta = self.epsilon * delta
        elif self.norm == "L2":
            norm = torch.norm(delta, dim=1).view(-1,1).repeat(1,self.img_dim)
            mask_norm = norm > self.epsilon
            delta = ~mask_norm * delta + self.epsilon * delta * mask_norm / norm
        else:
            NotImplementedError(f"Generator architecture not implemented for norm: {self.norm}" )
        return x + delta
