#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from tqdm import tqdm

from bio_neural_net.fruit_fly_network import (
        get_fruit_fly_network,
        INPUT_NEURONS,
        INPUT_SYNAPSE_CONDUCTANCE,
        CONDUCTANCE_DICT
)

from itertools import product

######################################################################
# Experiment Parameters
######################################################################

# times in s
start_time = 0.0
end_time = 10.0  # adjusted to match step size
dt = 1e-4

pickle_dir = 'sim_data'
file_name = 'sim2.pickle'

# changed params
input_neurons = INPUT_NEURONS.copy()
input_synapse_conductance = INPUT_SYNAPSE_CONDUCTANCE.copy()
conductance_dict = CONDUCTANCE_DICT.copy()

conductance_dict[('EIP', 'PEI')] = 11
conductance_dict[('PEI', 'EIP')] = 7

conductance_dict[('EIP', 'REIP')] = 5
conductance_dict[('REIP', 'EIP')] = 40

conductance_dict[('EIP', 'PEN')] = 12
conductance_dict[('PEN', 'EIP')] = 8


for rot in ['rot_CW', 'rot_CCW']:
    input_neurons[rot]['freq'] = 50
    input_neurons[rot]['size'] = 1
    input_synapse_conductance[rot] = 0.1

# Inptus per figure 5a.
cue_dict = {
    'EB-L1_input': [(0, 1.0)],       # cue onset to cue offset
    'RPEI_input':  [(0, 7.0)],      # cue onset to CW rotation
    # 'RPEI_input':  [(4.15, 10.0)],  # both rotations
    'rot_CW':      [(4.15, 5.0)],    # CW rotation
    'rot_CCW':     [(5.0, 6.0)],     # CWW rotation
    'RPEN_input':  [(7.03, 10)]
}

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
    for step in tqdm(range(steps)):
        net.update()

    neuron_dict = {neuron.name: neuron.firing_time_indices
                   for neuron in net.neurons.values()}

    with open(os.path.join(pickle_dir, file_name), 'wb') as f:
        pickle.dump((ts, neuron_dict), f)
