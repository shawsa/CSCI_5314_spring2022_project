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
end_time = 10  # 10.0  # adjusted to match step size
dt = 1e-5

pickle_dir = 'sim_data'
file_name = 'sim2.pickle'

# Inptus per figure 5a.
cue_dict = {
    'EB-L1_input': [(0, 1.0)],      # cue onset to cue offset
    'RPEN_input':  [(0, 4.0)],      # cue onset to CW rotation
    'RPEI_input':  [(4.0, 10.0)],   # both rotations
    'rot_CW':      [(4.0, 7.0)],    # CW rotation
    'rot_CCW':     [(7.0, 10.0)]    # CWW rotation
}

rot_freq_override = 3150

######################################################################
# End Experiment Parameters
######################################################################

steps = int((end_time - start_time)/dt)
ts = (np.arange(steps) - start_time) * dt

net = get_fruit_fly_network()
if rot_freq_override is not None:
    net['rot_CW'].freq = rot_freq_override
    net['rot_CCW'].freq = rot_freq_override

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
