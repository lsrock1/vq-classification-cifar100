import torch
from torch import nn
from torch.nn import functional as F


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, on_training=True):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.on_training = on_training

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, input.shape[-1])
        if not self.on_training and input.shape[-1] != self.embed.shape[0]:
            assert self.embed.shape[0] % input.shape[-1] == 0
            
            proxy = self.embed.reshape(input.shape[-1], -1, self.n_embed)
            ld = proxy.shape[1]
            proxy = proxy.mean(dim=1)

            dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ proxy
                + proxy.pow(2).sum(0, keepdim=True)
            )
            _, embed_ind = (-dist).max(1)
            # new_input = torch.zeros([input.shape[0], input.shape[1], input.shape[2], input.shape[3], ld]).to(input.device)
            # new_input[:, :, :, :, 0] = input + new_input[:, :, :, :, 0]
            # input = input.reshape(input.shape[0], input.shape[1], input.shape[2], -1)
            embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
            embed_ind = embed_ind.view(*input.shape[:-1])
            quantize = self.embed_code(embed_ind)
            new_input = quantize.clone().detach().reshape(input.shape[0], input.shape[1], input.shape[2], input.shape[3], ld)
            # new_input[..., 0] = 0
            diff = (new_input[...,0] - input).pow(2).mean()
            new_input[..., 0] = input
            input = new_input.reshape(input.shape[0], input.shape[1], input.shape[2], -1)
            
        else:
            dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
            )
            _, embed_ind = (-dist).max(1)
            embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
            embed_ind = embed_ind.view(*input.shape[:-1])
            quantize = self.embed_code(embed_ind)
            diff = (quantize.detach() - input).pow(2).mean()
        if self.training and self.on_training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # dist_fn.all_reduce(embed_onehot_sum)
            # dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))