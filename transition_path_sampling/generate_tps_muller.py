import sys
sys.path.append("transition_path_sampling")
import numpy as np
import trajectories
import transition_paths
import standardize_tps
import csv
import argparse

def gen_paths():
    traj = np.array(trajectories.gen_trajectory())
    tp = transition_paths.rec_transition_paths(data=traj)
    std_tp = standardize_tps.standardize(data=tp)
    return std_tp

parser = argparse.ArgumentParser()
parser.add_argument("-output_path", nargs=1, type=str, default="data/transition_paths/test2")
args = parser.parse_args()

output_path = args.output_path

tps = []
for i in range(20):
    print(i)
    arr = gen_paths()
    for j in range(arr.shape[0]):
        tps.append(arr[j])
print(tps)
with open(output_path, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(list(tps))