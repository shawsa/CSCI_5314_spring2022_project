#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle

with open(os.path.join('sim_data', 'sim1.pickle'), 'rb') as f:
    ts, spike_dict = pickle.load(f)

for i, (name, time_indices) in enumerate(spike_dict.items()):
    height = 2*i
    plt.plot(ts, height + 0*ts, 'k-', linewidth=0.1)
    plt.plot(ts[time_indices], [height]*len(time_indices), '.')
    plt.text(ts[-1], height - 1, name)

plt.show()
