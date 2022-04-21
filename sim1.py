#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from tqdm import tqdm
from bio_neural_net import (
    Network,
    NeuronCluster,
    InputNeuronCluster,
    DEFAULT_NEURON_PARAMS,
    REIP_PARAMS,
    SynapseCluster,
    NMDASynapseCluster,
    NMDA_PARAMS,
    GABAA_PARAMS,
    ACETYLCHOLINE_PARAMS
)

from bio_neural_net.fruit_fly_network import get_fruit_fly_network

from itertools import product

######################################################################
# Experiment Parameters
######################################################################

# times in s
start_time = 0.0
end_time = .1  # 10.0  # adjusted to match step size
dt = 1e-6

pickle_dir = 'sim_data'
file_name = 'sim1.pickle'

# visual cues - stimulate the EB (EIP neurons)
EB_INPUT_FREQ = 50
cue_dict = {
    'EB-L1_input': [(0, end_time/2)],
    'RPEN_input': [(0, end_time)],
}

######################################################################
# End Experiment Parameters
######################################################################

steps = int((end_time - start_time)/dt)
ts = (np.arange(steps) - start_time) * dt

net = get_fruit_fly_network()

for name, intervals in cue_dict.items():
    net[name].intervals += intervals

net.set_time_params(start_time, dt, steps)
net.reset()

#######################
# main
#######################
if __name__ == '__main__':
    # net.print_full()
    voltage = []
    gating = []
    current = []
    neurons = ['EIP5', 'PEN5', 'RPEN']
    voltages = [[] for _ in neurons]
    for step in tqdm(range(steps)):
        for i, name in enumerate(neurons):
            voltages[i].append(net[name].V)
        net.update()

    neuron_dict = {neuron.name: neuron.firing_time_indices
                   for neuron in net.neurons.values()}

    with open(os.path.join(pickle_dir, file_name), 'wb') as f:
        pickle.dump((ts, neuron_dict), f)

    total_spikes = sum(len(lst) for lst in neuron_dict.values())

    print(f'Total spikes: {total_spikes}')

    for name, Vs in zip(neurons, voltages):
        plt.plot(ts, Vs, label=name)
    plt.ylabel('voltage (mV)')
    plt.legend()

    plt.show()
