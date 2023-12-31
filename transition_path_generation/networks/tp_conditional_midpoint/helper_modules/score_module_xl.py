import torch
from torch import nn

class ScoreModule(nn.Module):
    def __init__(self):
        super(ScoreModule, self).__init__()
        self.fc1 = nn.Linear(8, 60)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(60, 300)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.tanh2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(300, 800)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.tanh3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(800, 800)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        self.tanh4 = nn.LeakyReLU()
        self.fc5 = nn.Linear(800, 400)
        torch.nn.init.xavier_uniform_(self.fc5.weight)
        self.tanh5 = nn.LeakyReLU()
        self.fc6 = nn.Linear(400, 30)
        torch.nn.init.xavier_uniform_(self.fc6.weight)
        self.fin_relu = nn.LeakyReLU()
        self.fin = nn.Linear(30, 2)
        torch.nn.init.xavier_uniform_(self.fin.weight)

    def forward(self, x_next, x_cur, x_prev, t, path_t):
        y = torch.tensor([t]).float().repeat(x_cur.shape[0], 1)
        path_t = torch.unsqueeze(path_t, 1)
        out = torch.cat((x_cur, x_prev, x_next, path_t, y), dim=1)
        out = self.fc1(out)
        out = self.tanh1(out)
        out = self.fc2(out)
        out = self.tanh2(out)
        out = self.fc3(out)
        out = self.tanh3(out)
        out = self.fc4(out)
        out = self.tanh4(out)
        out = self.fc5(out)
        out = self.tanh5(out)
        out = self.fc6(out)
        out = self.fin_relu(out)
        out = self.fin(out)
        return out