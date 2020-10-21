import torch
import math
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal
import ipdb
from torch.distributions import Normal
from flows import *


def entropy(input):
    max_input, _ = torch.max(input, 1)
    input = input - max_input.view(-1, 1).repeat(1, input.shape[1])
    softval = F.softmax(input, 1)
    entropy = torch.sum(softval *
                      (input - input.exp().sum(1).log().view(-1, 1).repeat(1, 10)),1)
    return torch.mean(entropy)

def sample(input, dim=-1):
    softval = F.softmax(input,dim)
    index = torch.multinomial(softval,1).view(-1)
    output = torch.zeros_like(softval)
    output[torch.arange(softval.shape[0]),index] = 1.
    # output.eq_(0.)
    return output.detach(), entropy

class SampleST(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        output, entropy = sample(input, dim=dim)
        ctx.save_for_backward(input, output)
        ctx.other_params = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        gr = None
        if ctx.needs_input_grad[0]:
            input, output = ctx.saved_variables
            dim = ctx.other_params
            s = F.softmax(input, dim)
            gs = (grad_output * s).sum(dim, True)
            gr = s * (grad_output - gs)

        return gr, None


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out

class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

# Initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=192):
        return input.view(input.size(0), size, 1, 1)

class Generator(nn.Module):
    def __init__(self, input_size, latent=50, deterministic=False):
        """
        A modified VAE. Latent is Gaussian (0, sigma) of dimension latent.
        Decode latent to a noise vector of `input_size`,

        Note the Gaussian \mu is not learned since input `x` acts as mean

        Args:
            input_size: size of image, 784 in case of MNIST
            latent: size of multivar Gaussian params
        """
        super(Generator, self).__init__()
        self.input_size = input_size
        self.deterministic = deterministic


        self.fc1_mu = nn.Linear(input_size, 400)
        self.fc1_sig = nn.Linear(input_size, 400)
        self.fc2_sig = nn.Linear(400, latent)
        self.fc2_mu = nn.Linear(400,latent)
        self.fc3 = nn.Linear(latent, 400)
        self.fc4 = nn.Linear(400, input_size)

    def encode(self, x):
        h_mu = F.relu(self.fc1_mu(x))
        h_sig = F.relu(self.fc1_sig(x))
        return self.fc2_mu(h_mu), self.fc2_sig(h_sig)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        sample = eps.mul(std).add_(mu)
        # covar_mat = torch.diag(sample[0])
        return sample

    def decode(self, z):
        """
        Final layer should probably not have activation?
        """
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x, epsilon, target=None):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        delta = self.decode(z)
        if self.deterministic:
            kl_div = torch.Tensor([0.]).cuda()
        else:
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_div = kl_div
            kl_div = kl_div / x.size(0)  # mean over batch

        return delta, kl_div

class ConvGenerator(nn.Module):
    def __init__(self, nchannels, block, nblocks, deterministic, flow_args, growth_rate=12, reduction=0.5,\
            num_classes=10, latent=50, norm="Linf"):
        """
        A modified VAE.
        Encode the image into h to generate a probability vector p
        use p to sample a categorical variable (using gamble softmax)
        Decode the concatenation of h and the categorical variable into a delta.
        """
        super(ConvGenerator, self).__init__()
        self.growth_rate = growth_rate
        self.norm = norm

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(nchannels, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear_1 = nn.Linear(num_planes+1, 100)
        self.linear_2 = nn.Linear(100, num_classes)
        self.linear_3 = nn.Linear(num_classes,latent)
        ngf = 64
        self.latent = latent
        self.log_det_j = 0.

        self.deterministic = deterministic

        _st_sample = SampleST()
        self.sample = lambda x, target=None: _st_sample.apply(x, 1, target)

        n_blocks, flow_hidden_size, n_hidden = flow_args[0], flow_args[1], flow_args[2]
        flow_model, flow_layer_type = flow_args[3], flow_args[4]
        self.flow_model = flow_model
        # Flow parameters
        if flow_model is not None:
            self.flow = flows
            self.num_flows = 30
            self.num_flows = self.num_flows
            # Amortized flow parameters
            self.amor_u = nn.Linear(num_planes, self.num_flows * self.latent)
            self.amor_w = nn.Linear(num_planes, self.num_flows * self.latent)
            self.amor_b = nn.Linear(num_planes, self.num_flows)

            # Normalizing flow layers
            for k in range(self.num_flows):
                flow_k = self.flow()
                self.add_module('flow_' + str(k), flow_k)

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(num_planes+1+latent, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, 3, 4, 2, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            # # state size. (ngf) x 32 x 32
            nn.Tanh()
        )

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def encode(self, out):
        batch_size = out.size(0)
        out_1 = self.linear_1(out)
        out_2 = self.linear_2(out)
        h1 = F.relu(out_1)
        h2 = F.relu(out_2)
        u,w,b = None,None,None
        if self.flow_model is not None:
            # return amortized u an w for all flows
            u = self.amor_u(out).view(batch_size, self.num_flows, self.latent, 1)
            w = self.amor_w(out).view(batch_size, self.num_flows, 1, self.latent)
            b = self.amor_b(out).view(batch_size, self.num_flows, 1, 1)
        return h1,h2,u,w,b

    def reparameterize(self, mu, logvar):
        if self.deterministic:
            z = mu + logvar.mul(0.5).exp_()
            return z
        else:
            std = logvar.mul(0.5).exp_()
            eps = torch.cuda.FloatTensor(std.size()).normal_()
            eps = Variable(eps)
            return eps.mul(std).add_(mu)

    def decode(self, z):
        z = z.view(-1,self.latent,1,1)
        gen = self.decoder(z)
        return gen

    def forward(self, x, epsilon, target=None):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        h = out.view(out.size(0), -1)
        # mu,logvar,u,w,b  = self.encode(out)
        # h = self.reparameterize(mu,logvar)
        h = torch.cat((h, epsilon.repeat(x.shape[0], 1)), 1)
        logits = self.linear_2(F.relu(self.linear_1(h)))
        one_hot = self.sample(logits, target=target)
        z = F.relu(self.linear_3(one_hot))  # 8,2,2
        h = torch.cat((h, z), 1).view(out.size(0), -1, 1, 1)
        delta = self.decoder(h)
        # delta = out.view(-1, self.img_dim) - x.view(-1,self.img_dim)
        if self.norm == "Linf":
            delta = epsilon.item() * delta
        elif self.norm == "L2":
            raise("L2 norm not implemented on CIFAR not implemented")
            norm = torch.norm(delta, dim=1).view(-1, 1).repeat(1, self.img_dim)
            mask_norm = norm > epsilon
            delta = ~mask_norm * delta + epsilon * delta * mask_norm / norm
        else:
            NotImplementedError(f"Generator architecture not implemented for norm: {self.norm}")
        return torch.clamp(x + delta, min=0., max=1.), entropy(logits)

class DCGAN(nn.Module):
    def __init__(self, num_channels=3, ngf=100):
        super(DCGAN, self).__init__()
        """
        Initialize a DCGAN. Perturbations from the GAN are added to the inputs to
        create adversarial attacks.
        - num_channels is the number of channels in the input
        - ngf is size of the conv layers
        """
        self.generator = nn.Sequential(
                # input is (nc) x 32 x 32
                nn.Conv2d(num_channels, ngf, 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                #nn.Dropout2d(),
                # state size. 48 x 32 x 32
                nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                #nn.Dropout2d(),
                # state size. 48 x 32 x 32
                nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                #nn.Dropout(),
                # state size. 48 x 32 x 32
                nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                #nn.Dropout(),
                # state size. 48 x 32 x 32
                nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 48 x 32 x 32
                nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 48 x 32 x 32
                nn.Conv2d(ngf, ngf, 1, 1, 0, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 3 x 32 x 32
                nn.Conv2d(ngf, num_channels, 1, 1, 0, bias=False),
                nn.Tanh()
        )

    def forward(self, inputs, target=None):
        return self.generator(inputs), inputs

    def save(self, fn):
        torch.save(self.generator.state_dict(), fn)

    def load(self, fn):
        self.generator.load_state_dict(torch.load(fn))


class Cond_DCGAN(nn.Module):
    def __init__(self, num_channels=3, ngf=100):
        super(Cond_DCGAN, self).__init__()
        """
        Initialize a DCGAN. Perturbations from the GAN are added to the inputs to
        create adversarial attacks.
        - num_channels is the number of channels in the input
        - ngf is size of the conv layers
        """
        self.fcy = nn.Linear(10,100)
        self.fcz = nn.Linear(784, 200)
        self.generator = nn.Sequential(
                # input is (nc) x 32 x 32
                nn.Conv2d(num_channels, ngf, 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                #nn.Dropout2d(),
                # state size. 48 x 32 x 32
                nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                #nn.Dropout2d(),
                # state size. 48 x 32 x 32
                nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                #nn.Dropout(),
                # state size. 48 x 32 x 32
                nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                #nn.Dropout(),
                # state size. 48 x 32 x 32
                nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 48 x 32 x 32
                nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 48 x 32 x 32
                nn.Conv2d(ngf, ngf, 1, 1, 0, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 3 x 32 x 32
                nn.Conv2d(ngf, num_channels, 1, 1, 0, bias=False),
                nn.Tanh()
        )

    def forward(self, inputs, labels=None, nb_digits=10, target=None):
        if labels is None:
            batch_size = inputs.shape[0]
            y = torch.randint(0, nb_digits,(batch_size,1))
            # One hot encoding buffer that you create out of the loop and just keep reusing
            y_onehot = torch.zeros(batch_size, nb_digits)
            labels = y_onehot.scatter_(1, y, 1)

        x = F.relu(self.fcz(inputs))
        y = F.relu(self.fcy(labels))
        inputs = torch.cat([x, y], 1)
        return self.generator(inputs), inputs

    def save(self, fn):
        torch.save(self.generator.state_dict(), fn)

    def load(self, fn):
        self.generator.load_state_dict(torch.load(fn))

class MnistGenerator(nn.Module):
    def __init__(self, norm="Linf"):
        super(MnistGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=3, padding=1),  # b, 64, 10, 10
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2),  # b, 64, 5, 5
            nn.Conv2d(64, 32, 3, stride=2, padding=1),  # b, 32, 3, 3
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=1)  # b, 32, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2),  # b, 32, 5, 5
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 5, stride=3, padding=1),  # b, 16, 15, 15
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )
        _st_sample = SampleST()
        self.sample = lambda x: _st_sample.apply(x,1)
        self.fc = nn.Sequential(
            nn.Linear(32*2*2+1,64),
            nn.LeakyReLU(.2),
            nn.Linear(64,10))
        self.fc_z = nn.Linear(10,32*2*2-1)
        self.fc_input = nn.Linear(10,32*2*2-1)
        self.norm = norm
        self.img_dim = 28*28

    def forward(self, x, epsilon, target=None):
        h = self.encoder(x)
        h = torch.cat((h.view(-1,32*2*2),epsilon.repeat(x.shape[0],1)),1)
        logits = self.fc(h.view(-1,32*2*2+1))
        one_hot = self.sample(logits)
        z = self.fc_z(one_hot)
        if target is not None:
            target_onehot = torch.zeros(target.size() + (10,)).cuda()
            target_onehot.scatter_(1, target.detach().unsqueeze(1), 1)
            z += self.fc_input(target_onehot)
        z = F.relu(z)
        h = torch.cat((h,z),1).view(-1,64,2,2)
        # delta = self.decoder(h).view(-1,self.img_dim)
        out = .5*(self.decoder(h) + 1.)
        delta = out - x
        if self.norm == "Linf":
            delta = epsilon.item() * delta
        elif self.norm == "L2":
            norm = torch.norm(delta, dim=1).view(-1,1).repeat(1,self.img_dim)
            mask_norm = norm > self.epsilon
            delta = ~mask_norm * delta + self.epsilon * delta * mask_norm / norm
        else:
            NotImplementedError(f"Generator architecture not implemented for norm: {self.norm}" )
        output = x + delta
        return output , entropy(logits)
        # if self.norm == "Linf":
            # delta = epsilon.item() * delta
        # elif self.norm == "L2":
            # norm = torch.norm(delta, dim=1).view(-1, 1).repeat(1, self.img_dim)
            # mask_norm = norm > epsilon
            # delta = ~mask_norm * delta + epsilon * delta * mask_norm / norm
        # else:
            # NotImplementedError(f"Generator architecture not implemented for norm: {self.norm}")
        # output = x.view(-1, self.img_dim) + delta
        # return output , entropy(logits)


    def save(self, fn_enc, fn_dec):
        torch.save(self.encoder.state_dict(), fn_enc)
        torch.save(self.decoder.state_dict(), fn_dec)

    def load(self, fn_enc, fn_dec):
        self.encoder.load_state_dict(torch.load(fn_enc))
        self.decoder.load_state_dict(torch.load(fn_dec))



class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, device, input_nc, output_nc, epsilon=.3, norm="Linf",
                 ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        _st_sample = SampleST()
        self.sample = lambda x: _st_sample.apply(x, 1)
        self.ngf=ngf
        self.device = device
        self.epsilon = epsilon
        self.norm = norm
        self.fc_z = nn.Linear(10, 63)
        self.fc_input = nn.Linear(10,63)
        self.fc_h = nn.Sequential(
            nn.Linear(16*32*32-63,100),
            nn.ReLU(True),
            nn.Linear(100,10)
        )
        use_bias = norm_layer == nn.InstanceNorm2d

        pre_model = [nn.ReflectionPad2d(3),
                     nn.Conv2d(input_nc, ngf-1, kernel_size=7, padding=0, bias=use_bias),
                     norm_layer(ngf-1),
                     nn.ReLU(True)]

        n_downsampling = 2

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            pre_model += [nn.Conv2d(ngf * mult-1, ngf * mult * 2-1, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2-1),
                          nn.ReLU(True)]

        mult = 2 ** n_downsampling
        assert(n_blocks % 2 == 0)
        for i in range(n_blocks//2):       # add ResNet blocks

            pre_model += [ResnetBlock(ngf * mult-1, padding_type=padding_type,
                                      norm_layer=norm_layer,
                                      use_dropout=use_dropout, use_bias=use_bias)]
        self.pre_model = nn.Sequential(*pre_model)
        model = []
        for i in range(n_blocks//2):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                  norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias)]
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, epsilon, target=None):
        """Standard forward"""
        h = self.pre_model(input)
        batch_size = input.shape[0]
        h = torch.cat((h.view(batch_size, -1), epsilon.repeat(batch_size, 1)), 1)
        logit = self.fc_h(h)
        one_hot = self.sample(logit)
        z = self.fc_z(one_hot)
        if target is not None:
            target_onehot = torch.zeros(target.size() + (10,)).cuda()
            target_onehot.scatter_(1, target.detach().unsqueeze(1), 1)
            z += self.fc_input(target_onehot)
        z = F.relu(z)
        h = torch.cat((h,z),1).view(batch_size,256,8,8)
        delta = self.model(h)
        if self.norm == "Linf":
            delta = epsilon.item() * delta # outputs in [-1,1]
        else:
            norm = torch.norm(delta, p=2,dim=(1,2,3)).view(-1, 1,1,1).repeat(1, 3,32,32)
            mask_norm = norm > self.epsilon
            delta = ~mask_norm * delta + epsilon * delta * mask_norm / norm
        return input + delta, entropy(logit)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
