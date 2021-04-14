import torch
import math
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal
import ipdb
from torch.distributions import Normal


class BlackAttack(nn.Module):
    def __init__(self, input_size, latent):
        """
        A modified VAE. Latent is Gaussian (0, sigma) of dimension latent.
        Decode latent to a noise vector of `input_size`,

        Note the Gaussian \mu is not learned since input `x` acts as mean

        Args:
            input_size: size of image, 784 in case of MNIST
            latent: size of multivar Gaussian params
        """
        super(BlackAttack, self).__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, 400)
        self.fc_sig = nn.Linear(400, latent)
        self.fc3 = nn.Linear(latent, 400)
        self.fc4 = nn.Linear(400, input_size)
        # self.covar_mat = torch.eye(latent)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_sig(h1)

    def reparameterize(self, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        mu = torch.zeros(len(std[0])).cuda()
        covar_mat = torch.eye(len(std[0])).cuda()
        covar_mat = std*covar_mat
        m = MultivariateNormal(mu,covar_mat)
        log_prob_a = m.log_prob(eps.mul(std))
        return eps.mul(std), log_prob_a

    def decode(self, z):
        """
        Final layer should probably not have activation?
        """
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        # Reshape data for net
        if len(x.shape)==4:
            batch, chan, h, w = x.shape
            x = x.view(batch,chan,h*w).squeeze(1)

        # Forward pass
        logvar = self.encode(x.view(-1, self.input_size))
        z,log_prob_a = self.reparameterize(logvar)

        # Shape noise to match original data
        delta = self.decode(z).unsqueeze(1)
        delta = delta.view(batch,chan,h,w)

        return delta, logvar, log_prob_a

# Initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)

class GaussianPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, latent, decode=False):
        super(GaussianPolicy, self).__init__()

        self.decode = decode

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        if decode:
            self.linear3 = nn.Linear(latent, input_size)
        else:
            # If no decode, then Gaussian must be size of input
            latent = input_size # for now, no decode

        self.mean_linear = nn.Linear(hidden_size, latent)
        self.log_std_linear = nn.Linear(hidden_size, latent)


        self.apply(weights_init)
        self.LOG_SIG_MAX = 2
        self.LOG_SIG_MIN = -20
        self.epsilon = 1e-6

    def forward(self, x):
        # Reshape data for net
        if len(x.shape)==4:
            batch, chan, h, w = x.shape
            x = x.view(batch,chan,h*w).squeeze(1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)

        std = log_std.exp()
        normal = Normal(mean, std)
        delta = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        log_prob = normal.log_prob(delta)

        if self.decode:
            delta = self.linear(delta)
            # Problem here is log_prob can't be for delta unless same size...

        # Enforcing Action Bound
        # log_prob -= torch.log(1 - action.pow(2) + self.epsilon)
        # log_prob = log_prob.sum(-1, keepdim=True)


        # Shape noise to match original data
        delta = delta.unsqueeze(1)
        delta = delta.view(batch,chan,h,w)

        return delta, mean, log_std, log_prob

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=192):
        return input.view(input.size(0), size, 1, 1)

class CifarVAE(nn.Module):
    def __init__(self, device, image_channels=3, h_dim=192, z_dim=32):
        super(CifarVAE, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=3, stride=1,padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(16, 3 , kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(image_channels, 3 , kernel_size=3, stride=1, padding=1),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 16, kernel_size=4, stride=1,padding=0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

reconstruction_function = nn.MSELoss(size_average=False)
def vae_loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD

class MnistBottleneck(nn.Module):
    def __init__(self):
        super(MnistBottleneck, self).__init__()
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

    def forward(self, h):
        mu, logvar = self.fc21(h), self.fc22(h)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

class MnistVAE(nn.Module):
    def __init__(self):
        super(MnistVAE, self).__init__()

        self.encoder = nn.Sequential(
	    nn.Linear(784, 400),
            nn.LeakyReLU(0.2),
        )
        self.fc21 = nn.Linear(400, 30)
        self.fc22 = nn.Linear(400, 30)

        self.decoder = nn.Sequential(
	    nn.Linear(30, 400),
            nn.LeakyReLU(0.2),
	    nn.Linear(400, 784),
            nn.LeakyReLU(0.2),
	    nn.Tanh(),
	)

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def bottleneck(self, h):
        mu, logvar = self.fc21(h), self.fc22(h)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

    def save(self, fn_enc, fn_dec):
        torch.save(self.encoder.state_dict(), fn_enc)
        torch.save(self.decoder.state_dict(), fn_dec)

    def load(self, fn_enc, fn_dec):
        self.encoder.load_state_dict(torch.load(fn_enc))
        self.decoder.load_state_dict(torch.load(fn_dec))

class Mnistautoencoder(nn.Module):
    def __init__(self):
        super(Mnistautoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        h = self.encoder(x)
        x = self.decoder(h)
        return x


    def save(self, fn_enc, fn_dec):
        torch.save(self.encoder.state_dict(), fn_enc)
        torch.save(self.decoder.state_dict(), fn_dec)

    def load(self, fn_enc, fn_dec):
        self.encoder.load_state_dict(torch.load(fn_enc))
        self.decoder.load_state_dict(torch.load(fn_dec))

