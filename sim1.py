#!/usr/bin/python3
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from bio_neural_net.fruit_fly_network import (
        get_fruit_fly_network,
        INPUT_NEURONS,
        INPUT_SYNAPSE_CONDUCTANCE,
        CONDUCTANCE_DICT
)

######################################################################
# Experiment Parameters
######################################################################

# times in s
start_time = 0.0
end_time = 1.0  # adjusted to match step size
dt = 1e-4

pickle_dir = 'sim_data'
file_name = 'sim1.pickle'

# changed params
input_neurons = INPUT_NEURONS.copy()
input_synapse_conductance = INPUT_SYNAPSE_CONDUCTANCE.copy()
conductance_dict = CONDUCTANCE_DICT.copy()

# input_neurons['EB-L1_input']['freq'] = 20
# conductance_dict[('EIP', 'PEI')] = 10
# conductance_dict[('PEI', 'EIP')] = 10
# conductance_dict[('EIP', 'REIP')] = 1
# conductance_dict[('REIP', 'EIP')] = 5

conductance_dict[('EIP', 'PEI')] = 12
conductance_dict[('PEI', 'EIP')] = 8

conductance_dict[('EIP', 'REIP')] = 5
conductance_dict[('REIP', 'EIP')] = 40

conductance_dict[('EIP', 'PEN')] = 12
conductance_dict[('PEN', 'EIP')] = 8


input_neurons['EB-L1_input']['freq'] = 60

# visual cues - stimulate the EB (EIP neurons)
cue_dict = {
    'EB-L1_input': [(0, end_time/2)],
    'RPEN_input': [(0, end_time)],

    # 'rot_CW': [(end_time/2, end_time)],
    # 'RPEI_input': [(end_time/2, end_time)]
}

# record
neurons = ['EIP4', 'EIP4', 'EIP3', 'EIP12']

######################################################################
# End Experiment Parameters
######################################################################

steps = int((end_time - start_time)/dt)
ts = (np.arange(steps) - start_time) * dt

print('Constructing network . . . ', end='')
net = get_fruit_fly_network(
        input_neurons=input_neurons,
        conductance_dict=conductance_dict
    )
print('complete.')

for name, intervals in cue_dict.items():
    net[name].intervals += intervals

net.set_time_params(start_time, dt, steps)
net.reset()

#######################
# main
#######################
if __name__ == '__main__':
    # net.print_full()
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
        plt.plot(ts, Vs, '.-', label=name)
    plt.ylabel('voltage (mV)')
    plt.legend()

    plt.show()
