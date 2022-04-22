#!/usr/bin/python3

import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np

with open(os.path.join('sim_data', 'sim1.pickle'), 'rb') as f:
    ts, spike_dict = pickle.load(f)

filtered_spike_dict = {
        name: time_indices
        for name, time_indices in spike_dict.items()
        if len(time_indices) != 0}
for height, (name, time_indices) in enumerate(filtered_spike_dict.items()):
    plt.plot(ts, height + 0*ts, 'k-', linewidth=0.1)
    plt.plot(ts[time_indices], [height]*len(time_indices), '.')
    plt.text(ts[-1], height, name)

plt.show()


zs = np.linspace(ts[0], ts[-1], 401)

def kern(t, scale=0.05):
    '''A smoothing kernel.'''
    return 1/scale*np.heaviside(t, .5)*np.exp(-t/scale)

max_rates = -1
for name, time_indices in filtered_spike_dict.items():
    rates = sum(kern(zs - ts[i]) for i in time_indices)
    max_rates = max(max_rates, np.max(rates))
    plt.semilogy(zs, rates, label=name)

plt.ylim(1, max_rates*1.1)
plt.grid(which='minor', axis='y', alpha=0.3)
plt.grid(which='major', axis='both', alpha=0.5, linestyle=':')
plt.legend()
plt.show()
