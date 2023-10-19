import torch
from torch import nn

class FinalPointModule(nn.Module):
    def __init__(self):
        super(FinalPointModule, self).__init__()
        self.fc1 = nn.Linear(6, 20)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(20, 400)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.tanh2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(400, 400)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.tanh3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(400, 200)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        self.tanh4 = nn.Tanh()
        self.fc5 = nn.Linear(200, 20)
        torch.nn.init.xavier_uniform_(self.fc5.weight)
        self.fin_relu = nn.LeakyReLU()
        self.fin = nn.Linear(20, 2)
        torch.nn.init.xavier_uniform_(self.fin.weight)

    def forward(self, x_cur, x_start, t, path_t):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        path_t = torch.unsqueeze(path_t, 1)
        y = torch.tensor([t], device=device).float().repeat(x_cur.shape[0], 1)
        out = torch.cat((x_cur, x_start, y, path_t), dim=1)
        out = self.fc1(out)
        out = self.tanh1(out)
        out = self.fc2(out)
        out = self.tanh2(out)
        out = self.fc3(out)
        out = self.tanh3(out)
        out = self.fc4(out)
        out = self.tanh4(out)
        out = self.fc5(out)
        out = self.fin_relu(out)
        out = self.fin(out)
        return out