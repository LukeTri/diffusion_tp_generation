import sys
sys.path.append("transition_path_sampling")
sys.path.append("transition_path_generation")
print(sys.path)
import csv
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from networks.tp_conditional_midpoint.conditional_net import NeuralNet as net1
from networks.tp_conditional_chain.conditional_net import NeuralNet as net2
import helpers
import random
import muller
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-output_path')
parser.add_argument('-input_path')
parser.add_argument('-net', default="net1")
parser.add_argument('-lr', default=0.015)
parser.add_argument('-num_epochs', default=300)
parser.add_argument('-batch_size', default=600)
parser.add_argument('-tp_len', default=9)
parser.add_argument('-sig_min', default=0.005)
parser.add_argument('-const', default=2.2)
parser.add_argument('-n', default=105)

args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path
net = globals()[args.net]
lr = args.lr
num_epochs = int(args.num_epochs)
batch_size = int(args.batch_size)
ts_len = int(args.tp_len)
sig_min = float(args.sig_min)
const = float(args.const)
n = int(args.n)

print('Input file:', input_path)
print('Output file:', output_path)
print('Net:', net)
print('Learning rate:', lr)
print('Num epochs:', num_epochs)
print('Batch size:', batch_size)
print('TP length:', ts_len)
print('Starting noise level:', sig_min)
print('Exponential Integrator Constant:', const)
print('Number of discritezation points:', n)

beta = 0.5
mu = torch.tensor([0, 0])
gamma = 0.99

save_int = True
save_freq = 250

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file = open(input_path)
csvreader = csv.reader(file)

tps = []
for row in csvreader:
    arr = np.zeros((len(row), 2))
    for i in range(arr.shape[0]):
        str_list = row[i].strip('[]').split()
        arr[i] = np.array([float(num) for num in str_list])
    arr.astype(float)
    tps.append(arr)
file.close()

arr = np.array(tps).astype(float)

data = torch.tensor(arr).float()

data = helpers.reshape_ts(data, ts_len)
print(data.shape)

data_n = torch.zeros((data.shape[0], data.shape[1] * 2))
data_n[:, :ts_len+1] = data[:,:, 0]
data_n[:, ts_len+1:2*ts_len+2] = data[:,:, 1]
data = data_n

ind = torch.randperm(data.shape[0])
test_data = data[ind[:data.shape[0] // 50]]
train_data = data[ind[data.shape[0] // 50:]]

vals = helpers.get_times(sig_min, const, n, beta)
t_vals = vals[:,0].to(device)
h = vals[:, 1].to(device)
sigmas = vals[:, 2]
print(vals)

training_generator = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
model = net(beta, mu).to(device)
model = model.float()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

total_step = len(training_generator)
for epoch in tqdm(range(num_epochs)):
    print(epoch)
    for i, samples in enumerate(training_generator):
        optimizer.zero_grad()
        samples = samples.to(device)
        samples = samples.requires_grad_(True)
        loss_tot = torch.zeros(1, device=device)
        ind = torch.randperm(vals.shape[0])
        vals = vals[ind]
        for j in range(n):
            loss = model.loss_func(samples, t_vals[j], h[j])
            loss_tot += loss
        loss_tot.backward()
        optimizer.step()
    if i % 5 == 0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                .format(epoch + 1, num_epochs, i + 1, total_step, loss_tot.item()))
        if epoch > 0 and save_int and epoch % save_freq == 0:
            save_p = output_path + "_epoch_" + str(epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss, 'min_sigma': sig_min}, save_p)
    scheduler.step()

torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss, 'min_sigma': sig_min}, output_path)