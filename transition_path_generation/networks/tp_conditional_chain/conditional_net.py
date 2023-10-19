import torch
from torch import nn
import sys
sys.path.append("transition_path_generation")
from networks.tp_conditional_chain.forward_module import ForwardModule
from networks.tp_conditional_chain.backward_module import BackwardModule
from networks.tp_conditional_chain.initial_point_module import InitialPointModule

class NeuralNet(nn.Module):
    def __init__(self, beta, mu):
        super(NeuralNet, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.beta = beta
        self.mu = mu.to(device)
        self.forward_score = ForwardModule()
        self.backward_score = BackwardModule()
        self.initial_score = InitialPointModule()

    def get_score(self, x, t):
        ts_len = x.shape[1] - 1
        x_n = torch.zeros((x.shape[0], ts_len, 2))
        x_n = x[:, :-1]
        path_t = x[:, -1]
        score = torch.zeros_like(x_n)
        for i in range(ts_len):
            n = i / ts_len
            if i == 0:
                score[:, i] += self.backward_score(x[:, i], t, x[:, i+1], n, path_t)
                score[:, i] += self.initial_score(x[:, i], t)
            elif i == ts_len - 1:
                score[:, i] += self.forward_score(x[:, i], t, x[:, i-1], n, path_t)
            else:
                score[:, i] += self.forward_score(x[:, i], t, x[:, i-1], n, path_t)
                score[:, i] += self.backward_score(x[:, i], t, x[:, i+1], n, path_t)
        return score

    def loss_func(self, x, t, h):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ts_len = (x.shape[1] - 2) // 2
        x_n = torch.zeros((x.shape[0], ts_len, 2), device=device)
        x_n[:, :, 0] = x[:, :ts_len]
        x_n[:, :, 1] = x[:, ts_len + 1:2*ts_len + 1]
        path_t = x[:, ts_len]
        
        score = torch.zeros_like(x_n, device=device)
        noise = torch.randn_like(x_n, device=device)
        mean = x_n * torch.exp(-self.beta * t) + self.mu.repeat(x_n.shape[0], x_n.shape[1],1) * (1 - torch.exp(-self.beta * t))
        x_tilde = mean + noise * torch.sqrt((1 - torch.exp(-2 * self.beta * t)) / (2 * self.beta))
        for i in range(ts_len):
            n = i / ts_len
            if i == 0:
                score[:, i] += self.backward_score(x_tilde[:, i], t, x_tilde[:, i+1], n, path_t)
                score[:, i] += self.initial_score(x_tilde[:, i], t, path_t)
            elif i == ts_len - 1:
                score[:, i] += self.forward_score(x_tilde[:, i], t, x_tilde[:, i-1], n, path_t)
            else:
                score[:, i] += self.forward_score(x_tilde[:, i], t, x_tilde[:, i-1], n, path_t)
                score[:, i] += self.backward_score(x_tilde[:, i], t, x_tilde[:, i+1], n, path_t)
        loss = torch.square(score - 2 * self.beta * (-x_tilde + mean)/(1 - torch.exp(-2 * self.beta * t))) * h
        loss = torch.mean(loss)
        return loss
