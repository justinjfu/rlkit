"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch import autograd

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch import networks


class DynamicsModel(object):
    def __init__(self, obs_dim, action_dim):
        super(DynamicsModel, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def step(self, obs, act):
        raise NotImplementedError()


class TorchDynamicsModel(PyTorchModule):
    def __init__(self, obs_dim, action_dim, deterministic=False):
        super(TorchDynamicsModel, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.deterministic = deterministic

    def forward(self, obs, act):
        raise NotImplementedError()

    def step(self, obs, act):
        obs = ptu.from_numpy(obs).unsqueeze(0)
        act = ptu.from_numpy(act).unsqueeze(0)
        if self.deterministic:
            nobs = self.forward(obs, act)[0]
            nobs = ptu.get_numpy(nobs)
        else:
            nobs = self.sample(obs, act)[0]
        return nobs


class GaussianDynamicsModel(TorchDynamicsModel):
    def __init__(self, obs_dim, action_dim, deterministic=False, grad_penalty=1e-2, **mlp_args):
        super(GaussianDynamicsModel, self).__init__(obs_dim, action_dim, deterministic=deterministic)
        self.grad_penalty = grad_penalty
        self.mean_mlp = FlattenResMlp(
            output_size=obs_dim,
            input_size=obs_dim+action_dim,
            **mlp_args)
        if self.deterministic:
            self.var_mlp = networks.FlattenMlp(
                output_size=obs_dim,
                input_size=obs_dim+action_dim,
                **mlp_args)

    def forward(self, obs, act, return_var=False):
        delta = self.mean_mlp(obs, act)
        mean = obs + delta
        if return_var:
            if self.deterministic:
                logvar = torch.zeros_like(obs)
            else:
                logvar = self.var_mlp(obs, act)
            return mean, logvar
        else:
            return mean 

    def sample(self, obs, act):
        mean, logvar = self.forward(obs, act, return_var=True)
        mean = ptu.get_numpy(mean)
        std = ptu.get_numpy(torch.sqrt(torch.exp(logvar)))
        nobs = np.random.normal(size=mean.shape)
        nobs = nobs * std
        nobs = nobs + mean
        return nobs

    def loss(self, obs, act, nobs):
        obs.requires_grad = True
        pred_obs, pred_logvar = self.forward(obs, act, return_var=True)
        pred_var = torch.exp(pred_logvar)
        exp_loss = (pred_obs - nobs)**2 / (2*pred_var)
        norm_loss = pred_logvar

        smooth_pen = torch.mean(torch.abs(pred_obs))
        obs_grad = autograd.grad(smooth_pen, obs, create_graph=True)[0]
        grad_magnitude = torch.sum(torch.abs(obs_grad))

        grad_loss = 1e-1 * grad_magnitude
        return torch.mean(exp_loss + norm_loss) + grad_loss


class ResBlock(torch.nn.Module):
    def __init__(self, in_size, out_size, activation=F.relu):
        super(ResBlock, self).__init__()
        self.l1 = nn.Linear(in_size, in_size)
        self.act = activation
        self.l2 = nn.Linear(in_size, out_size)
        self.skip = nn.Linear(in_size, out_size)

    def forward(self, x):
        out = self.l1(x)
        out = self.act(out)
        out = self.l2(out)
        out = self.skip(x) + out
        return out


class ResMlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=None,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        if output_activation is None:
            output_activation = lambda x:x

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = ResBlock(in_size, next_size)
            in_size = next_size
            #hidden_init(fc.weight)
            #fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenResMlp(ResMlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)
