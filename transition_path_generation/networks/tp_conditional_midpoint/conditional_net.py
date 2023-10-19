import torch
from torch import nn
import sys
sys.path.append("transition_path_generation")
from networks.tp_conditional_midpoint.helper_modules.score_module_xxl import ScoreModule
from networks.tp_conditional_midpoint.helper_modules.initial_point_large import InitialPointModule
from networks.tp_conditional_midpoint.helper_modules.final_point_large import FinalPointModule

class NeuralNet(nn.Module):
    def __init__(self, beta, mu):
        super(NeuralNet, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.beta = beta
        self.mu = mu.to(device)
        self.score = ScoreModule()
        self.initial_score = InitialPointModule()
        self.final_score = FinalPointModule()

    def greatest_power_of_2(self, n):
        count = 0
        while n % 2 == 0:
            n //= 2
            count += 1
        return 2 ** count

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
            if i == 0:
                score[:,0] += self.initial_score(x_tilde[:,0], t, path_t)
            elif i == ts_len - 1:
                score[:,ts_len-1] += self.final_score(x_tilde[:, ts_len-1], x_n[:, 0], t, path_t)
            else:
                diff = self.greatest_power_of_2(i)
                score[:, i] += self.score(x_tilde[:, i], x_n[:,i-diff], x_n[:, i+diff], t, path_t * (2 * diff / ts_len - 1))
        loss = torch.square(score - 2 * self.beta * (-x_tilde + mean)/(1 - torch.exp(-2 * self.beta * t))) * h
        loss = torch.mean(loss)
        return loss
