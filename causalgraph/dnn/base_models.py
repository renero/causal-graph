import pytorch_lightning as pl
import torch
import math
import numpy as np
import pytorch_lightning as pl

from torch import nn
from torch import Tensor
from torch import Tensor
from torch.autograd import Variable
from torch.nn.functional import softmax
from torch.distributions import Categorical
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .loss import MMDLoss

import logging
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
torch_log = logging.getLogger("pytorch_lightning")
torch_log.propagate = False
torch_log.setLevel(logging.ERROR)


ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)


class MLP(pl.LightningModule):

    class Block(nn.Module):
        """The main building block of `MLP`."""

        def __init__(self, d_in: int, d_out: int, bias: bool, dropout: float) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    def __init__(self,
                 input_size,
                 layers_dimensions,
                 batch_size,
                 lr,
                 loss,
                 dropout):
        super(MLP, self).__init__()
        # Log args to Tensorboard.
        self.save_hyperparameters()

        # Net params
        self.input_size = input_size
        self.batch_size = batch_size
        self.learning_rate = lr
        self.dropout = dropout
        self.hidden_layers = len(layers_dimensions)

        if loss == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss == "mae":
            self.loss_fn = nn.L1Loss()
        elif loss == "mmd":
            self.loss_fn = MMDLoss(mmd_type="quadratic")
        else:
            raise ValueError("Unknown loss function.")

        if isinstance(self.dropout, float):
            dropouts = [self.dropout] * len(layers_dimensions)
        assert len(layers_dimensions) == len(dropouts)
        self.blocks = nn.Sequential(
            *[
                MLP.Block(
                    d_in=layers_dimensions[i - 1] if i else self.input_size,
                    d_out=d,
                    bias=True,
                    dropout=dropout,
                )
                for i, (d, dropout) in enumerate(zip(layers_dimensions, dropouts))
            ]
        )
        self.head = nn.Linear(layers_dimensions[-1] if layers_dimensions else input_size, 1)

    def forward(self, x):
        this_device = 'cuda:0' if self.on_gpu else 'cpu'
        noise = torch.randn(x.shape[0], 1, device=this_device)
        X = torch.cat([x, noise], dim=1)
        y = self.blocks(X)
        y = self.head(y)
        return y

    def predict_step(self, batch, batch_idx, **kwargs):
        x, y = batch
        y_hat = self(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.loss_fn(yhat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.loss_fn(yhat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class DFF(pl.LightningModule):
    def __init__(self, input_size, hidden_size, batch_size, lr, loss):
        super(DFF, self).__init__()
        # Log args to Tensorboard.
        self.save_hyperparameters()

        # Net params
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = lr

        # Create the net
        self.approximate = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size, 1),
        )

        # Loss
        if loss == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss == "mae":
            self.loss_fn = nn.L1Loss()
        elif loss == "mmd":
            self.loss_fn = MMDLoss(mmd_type="quadratic")
        else:
            raise ValueError("Unknown loss function.")

    def forward(self, x):
        this_device = 'cuda:0' if self.on_gpu else 'cpu'
        noise = torch.randn(x.shape[0], 1, device=this_device)
        X = torch.cat([x, noise], dim=1)
        y = self.approximate(X)
        return y

    def predict_step(self, batch, batch_idx, **kwargs):
        x, y = batch
        y_hat = self(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.loss_fn(yhat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.loss_fn(yhat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        # return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        return optimizer


class MDN(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_gaussians: int,
        lr: float,
        batch_size: int,
        loss_function: str = "loglikelihood",
    ):
        """
        Init function for the MDN

        Arguments:
            input_size (int): the number of dimensions in the input
            hidden_size (int): the number of dimensions in the hidden layer
            num_gaussians (int): the number of Gaussians per output dimensions
            lr (float): learning rate
            batch_size (int): Batch size.
            loss_function (str): Loss function can be either 'loglikelihood' or 'mmd'
                for Maximal Mean Discrepancy
        Input:
            minibatch (BxD): B is the batch size and D is the number of input
                dimensions.
        Output:
            (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
                number of Gaussians, and O is the number of dimensions for each
                Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
                is the standard deviation of each Gaussian. Mu is the mean of each
                Gaussian.
        """

        super(MDN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_gaussians = num_gaussians
        self.lr = lr
        self.batch_size = batch_size
        self.loss_fn = loss_function

        self.save_hyperparameters()

        self.mdn = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh())
        self.pi = nn.Linear(hidden_size, num_gaussians)
        self.sigma = nn.Sequential(nn.Linear(hidden_size, num_gaussians), nn.ELU())
        self.mu = nn.Linear(hidden_size, num_gaussians)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=(self.lr or self.learning_rate))
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch[0].size()[0] == self.batch_size:
            loss = self.common_step(batch)
        else:
            loss = 0.0
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def common_step(self, batch):
        x, y = batch
        # x = self.add_noise(x)
        pi, sigma, mu = self(x)
        if self.loss_fn == "loglikelihood":
            loss = self.mdn_loss(pi, sigma, mu, y)
        else:
            y_hat = Variable(self.g_sample(pi, sigma, mu), requires_grad=True)
            loss = self.mmd_loss(y.float(), y_hat.float(), "multiscale")
        return loss

    def forward(self, x):
        this_device = 'cuda:0' if self.on_gpu else 'cpu'
        noise = torch.randn(x.shape[0], 1, device=this_device)
        X = torch.cat([x, noise], dim=1)
        z_h = self.mdn(X)
        # z_h = self.mdn(x)
        pi = softmax(self.pi(z_h), -1)
        sigma = torch.exp(self.sigma(z_h))
        mu = self.mu(z_h)
        return pi, sigma, mu

    def mdn_loss(self, pi, sigma, mu, y):
        """
        Calculates the error, given the MoG parameters and the target
        The loss is the negative log likelihood of the data given the MoG
        parameters.
        """
        result = self.gaussian_probability(y, mu, sigma) * pi
        result = torch.sum(result, dim=1)
        result = -torch.log(result)
        return torch.mean(result)

    @staticmethod
    def gaussian_probability(y, mu, sigma):
        # make |mu|=K copies of y, subtract mu, divide by sigma
        result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
        result = -0.5 * (result * result)
        return (torch.exp(result) * torch.reciprocal(sigma)) * ONEOVERSQRT2PI

    @staticmethod
    def mmd_loss(x, y, kernel):
        """
        https://www.kaggle.com/onurtunali/maximum-mean-discrepancy

        Emprical maximum mean discrepancy. The lower the result,
        the more evidence that distributions are the same.

        Args:
            x: first sample, distribution P
            y: second sample, distribution Q
            kernel: kernel type such as "multiscale" or "rbf"
        """
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

        XX, YY, XY = (torch.zeros(xx.shape),
                      torch.zeros(xx.shape),
                      torch.zeros(xx.shape))

        if kernel == "multiscale":
            bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            for a in bandwidth_range:
                XX += a ** 2 * (a ** 2 + dxx) ** -1
                YY += a ** 2 * (a ** 2 + dyy) ** -1
                XY += a ** 2 * (a ** 2 + dxy) ** -1

        if kernel == "rbf":
            bandwidth_range = [10, 15, 20, 50]
            for a in bandwidth_range:
                XX += torch.exp(-0.5 * dxx / a)
                YY += torch.exp(-0.5 * dyy / a)
                XY += torch.exp(-0.5 * dxy / a)

        return torch.mean(XX + YY - 2. * XY)

    @staticmethod
    def add_noise(x):
        noise = torch.randn(x.shape[0], 1)
        X = torch.cat([x, noise], dim=1)
        return X

    @staticmethod
    def g_sample(pi: Tensor, sigma: Tensor, mu: Tensor) -> Tensor:
        """
        Gumbel sampling comes from here:
        https://github.com/hardmaru/pytorch_notebooks/blob/master/mixture_density_networks.ipynb
        """

        def gumbel_sample(x, axis=1):
            z = np.random.gumbel(loc=0, scale=1, size=x.shape)
            return (np.log(x) + z).argmax(axis=axis)

        # work with numpy arrays
        pi_data = pi.data.numpy()
        sigma_data = sigma.data.numpy()
        mu_data = mu.data.numpy()

        n_samples = pi.shape[0]
        k = gumbel_sample(pi_data)
        indices = (np.arange(n_samples), k)
        rn = np.random.randn(n_samples)
        # I added the multiplication by "pi", otherwise, the resulting
        # distribution is far shifted from the target one...
        # Also added the ABS() since we only want positive values.
        sampled = np.abs(
            pi_data[indices] * (rn * sigma_data[indices] + mu_data[indices])
        )
        sampled = sampled.reshape(-1, 1)

        return torch.from_numpy(sampled)

    @staticmethod
    def sample(pi, sigma, mu):
        """
        Draw samples from a MoG.
        """
        # Choose which gaussian we'll sample from
        pis = Categorical(pi).sample().view(pi.size(0), 1, 1)
        # Choose a random sample, one randn for batch X output dims
        # Do a (output dims)X(batch size) tensor here, so the broadcast works in
        # the next step, but we have to transpose back.
        gaussian_noise = torch.randn(
            (sigma.size(2), sigma.size(0)), requires_grad=False
        )
        variance_samples = sigma.gather(1, pis).detach().squeeze()
        mean_samples = mu.detach().gather(1, pis).squeeze()
        return (gaussian_noise * variance_samples + mean_samples).transpose(0, 1)
