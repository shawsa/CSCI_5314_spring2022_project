#!/usr/bin/python3

import os.path
import pickle
import matplotlib.pyplot as plt
import numpy as np

from bio_neural_net.fruit_fly_network import EB_INNERVATION

pickle_dir = 'sim_data'
file_name = 'sim3.pickle'
image_dir = 'images'
image_prefix = 'sim3_'

with open(os.path.join(pickle_dir, file_name), 'rb') as f:
    ts, spike_dict = pickle.load(f)

img_path = os.path.join(image_dir, image_prefix + 'spikes.png')
print(f'Rendering {img_path}')
fig = plt.figure(num='Firing Rates', figsize=(20, 10))
for height, (name, time_indices) in enumerate(spike_dict.items()):
    plt.plot(ts, height + 0*ts, 'k-', linewidth=0.1)
    plt.plot(ts[time_indices], [height]*len(time_indices), '.')
    plt.text(ts[-1], height, name)

plt.title('Firing rates')

plt.savefig(img_path)
plt.close()

zs = np.linspace(ts[0], ts[-1], 1001)

def kern(t, scale=0.05):
    '''A smoothing kernel.'''
    return 1/scale*np.heaviside(t, .5)*np.exp(-t/scale)


rates_dict = {
        name: sum(kern(zs - ts[i]) for i in time_indices)
        for name, time_indices in spike_dict.items()
}

EB_rates = {
        region: sum(rates_dict[neuron]
                  for neuron in EB_INNERVATION.loc[EB_INNERVATION[region] == 1.0].index)
        for region in EB_INNERVATION.columns}

cue_labels = {
        0: 'cue on',
        1: 'cue off',
        4: 'CW rotation',
        5: 'CCW rotation',
        6: 'stop',
        7: 'forward walking'
}
x_tick_locs = list(range(11))
x_labels = [str(index) for index in range(11)]
for index, cue_str in cue_labels.items():
    x_labels[index] += '\n' + cue_str

X, Y = np.meshgrid(zs, np.arange(len(EB_rates)+1))
Z = np.empty_like(X)
for index, region in enumerate(EB_rates.keys()):
    Z[index] = EB_rates[region]

# plt.rcParams.update({'font.size': 22})
img_path = os.path.join(image_dir, image_prefix + 'EB_activity.png')
print(f'Rendering {img_path}')
plt.rcParams.update({'font.size': 22})
fig = plt.figure(num='EB Activity', figsize=(20, 10))
plt.pcolormesh(X, Y, Z, cmap='Blues')
for height in range(len(EB_rates) + 1):
    plt.plot([zs[0], zs[-1]], [height]*2, 'k-')
for loc in cue_labels.keys():
    plt.plot([loc]*2, [0, 16], 'k:')
plt.title('EB Activity')
plt.yticks([i+.5 for i in range(len(EB_rates))],
           EB_rates.keys())
plt.xticks(x_tick_locs, x_labels)
plt.xlabel('time (s)')
plt.colorbar(label='Hz')
plt.savefig(img_path)
plt.close()
